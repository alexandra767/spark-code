"""Opt-in export of completed sessions into the user's training corpus.

Phase 4 Task 6. Mirrors the shape of the existing Claude UI / JARVIS corpus
pipeline (``~/training-corpus/claude-ui-conversations.jsonl``: one JSON
object per line, keyed by ``source``/``model``/``exported_at``/``messages``
— confirmed by reading a live line from that file) so a downstream LoRA
training run can mix spark-code sessions in without a separate loader. Adds
two spark-code-specific fields the plan calls for: ``system_prompt_hash``
(so training code can group/filter by system-prompt version without storing
the — large, static, and not itself training signal — prompt text) and
``files_changed`` (the session's edit footprint).

Privacy is opt-in and defense-in-depth:

- ``corpus.export_enabled`` defaults to FALSE (see config.py's
  ``DEFAULT_CONFIG``) — nothing is ever written without the user turning
  this on. The check happens INSIDE :func:`export_session` (via
  ``meta["enabled"]``) rather than trusting every call site to gate
  correctly, so it's directly unit-testable and can't be bypassed by a
  future call site that forgets the ``if``.
- Sessions that errored mid-way are never exported (``meta["error"]``
  truthy -> skip). Only clean completions teach the model good behavior.
- Message content is scrubbed for API-key/token-shaped strings before
  writing (see :func:`_scrub`).
- This module never makes a network call — ``os.makedirs`` plus a local
  file append are the only side effects.
- No wall-clock reads here: the timestamp comes from ``meta`` (the caller,
  which runs in a context where ``time``/``datetime`` are actually
  available across both the interactive and headless entry points) so this
  module stays a pure function of its inputs and is trivial to unit test.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading

DEFAULT_CORPUS_DIR = "~/training-corpus/spark-code"

# Serializes the append below so two sessions ending near-simultaneously (e.g.
# a headless run and an interactive one sharing the same corpus dir) can't
# interleave partial writes into the same JSONL line.
_APPEND_LOCK = threading.Lock()

# API-key/token/credential-shaped strings. Aggressively pattern-based rather
# than exhaustive: this is a *record of behavior* for training, not a
# secrets-management surface — a false positive (redacting a long git SHA)
# just costs a token, not the other kind of cost. No existing scrub utility
# was found (grepped for scrub/redact/secret across spark_code/ — the only
# hit, cli.py's ``_redacted_config``, masks config dict values by KEY name
# and doesn't apply to free-text message content), so this is a small
# purpose-built regex set.
_SECRET_PATTERNS = [
    # OpenAI ("sk-proj-...")/Anthropic ("sk-ant-api03-...")/OpenRouter
    # ("sk-or-v1-...") all share the "sk-" prefix, so this one generic
    # pattern already catches all three cloud-key shapes Phase 5 Task 2
    # (/setkey) adds — verified via test_cloud_key_shapes_are_scrubbed rather
    # than duplicated as separate patterns.
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),  # OpenAI/Anthropic/OpenRouter-style API keys
    re.compile(r"\b[sr]k_live_[A-Za-z0-9]{16,}"),  # Stripe live secret/restricted keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"\bnpm_[A-Za-z0-9]{20,}"),  # npm access tokens
    re.compile(r"\bxox[bpars]-[A-Za-z0-9-]{10,}"),  # Slack tokens
    re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"),  # Google OAuth access tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    # Phase 5 Task 2: Google/Gemini API keys ("AIzaSy..."), the one
    # PROVIDER_PRESETS shape NOT already covered by the "sk-" pattern above.
    re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    # Generic "Bearer <opaque>" auth header — whole match redacted (drops the
    # token; the literal word "Bearer" going too is harmless).
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    # PEM private-key blocks — the exact shape of an Apple ASC .p8 signing key
    # ("BEGIN PRIVATE KEY") and any RSA/EC/OPENSSH variant. DOTALL so the
    # multi-line base64 body between the markers is swallowed whole. Also
    # catches the value of a Play service-account JSON `private_key` (whose
    # BEGIN/END markers survive whether the newlines are real or backslash-n
    # escaped) — without this a .p8 or SA JSON pasted into a session survived
    # into the training-corpus export intact.
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
               re.DOTALL),
    # Belt-and-suspenders for the JSON form specifically: redact the whole
    # `"private_key": "..."` field even if its BEGIN/END markers were stripped
    # or truncated before this ran.
    re.compile(r'"private_key"\s*:\s*"[^"]*"'),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),  # long hex strings (tokens/hashes-as-secrets)
]

_REDACTED = "[REDACTED]"

_ENABLED_TRUE_STRINGS = {"true", "1", "yes"}


def _is_enabled(value) -> bool:
    """Strictly coerce the opt-in ``export_enabled`` flag to a bool.

    Privacy-critical: a hand-edited quoted ``export_enabled: "false"`` in
    YAML loads as the (truthy!) STRING ``"false"`` — a plain ``if value:``
    would silently turn export ON. Only real ``True`` or an explicit
    affirmative string (case-insensitive ``true``/``1``/``yes``) enables;
    everything else (any other string, 0, None, "") is OFF.
    """
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _ENABLED_TRUE_STRINGS
    return False


def _scrub(value):
    """Recursively scrub secret-shaped strings out of a message value.

    Message ``content``/``tool_calls`` can be a plain string, a list of
    ``{"type": ..., ...}`` blocks (multimodal), or nested dicts (tool-call
    arguments) — this walks all three so nothing slips through unscrubbed.
    """
    if isinstance(value, str):
        scrubbed = value
        for pattern in _SECRET_PATTERNS:
            scrubbed = pattern.sub(_REDACTED, scrubbed)
        return scrubbed
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    return value


def _session_messages(context) -> list[dict]:
    """Return the session's non-system messages, orphaned-tool_call-repaired.

    Prefers ``Context.get_messages()`` over a raw ``.messages`` read: that
    method runs ``sanitize_orphaned_tool_calls()`` first, so an interrupted
    turn's assistant ``tool_calls`` message left without its matching
    ``role:"tool"`` reply is backfilled into a VALID sequence before it's
    serialized (an unsanitized orphan would be a malformed OpenAI transcript
    that no consumer — training or otherwise — should ingest). The
    system message ``get_messages`` prepends is dropped here: its content is
    captured as ``system_prompt_hash`` instead of stored inline (it's large,
    static, and not itself training signal). Falls back to a bare ``.messages``
    read for stand-ins without ``get_messages``.
    """
    getter = getattr(context, "get_messages", None)
    if callable(getter):
        return [m for m in getter() if m.get("role") != "system"]
    return list(getattr(context, "messages", None) or [])


def _scrub_messages(messages: list[dict]) -> list[dict]:
    """Deep-copy + scrub a session's message list for export.

    Embedded image data (``add_user_with_image``'s base64 data URLs) is
    dropped in favor of a lightweight marker — a text training corpus has no
    use for multi-megabyte inline image blobs, and leaving them in would
    bloat every export. Never mutates the live session's ``context.messages``.
    """
    cleaned = []
    for msg in copy.deepcopy(messages):
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    new_content.append({"type": "image_url", "image_url": "[image omitted]"})
                else:
                    new_content.append(_scrub(block))
            msg["content"] = new_content
        else:
            msg["content"] = _scrub(content)
        if "tool_calls" in msg:
            msg["tool_calls"] = _scrub(msg["tool_calls"])
        cleaned.append(msg)
    return cleaned


def export_session(context, path_dir: str, meta: dict) -> str | None:
    """Append the completed session in ``context`` as one JSONL record.

    ``meta`` carries everything this module deliberately does not compute
    itself:

    - ``enabled`` (bool): the resolved ``corpus.export_enabled`` config
      value. Missing/false -> no-op (opt-in default-off).
    - ``error`` (str | None): set -> the session errored mid-way, skip
      export (only clean completions are exported).
    - ``model`` (str): model name/id the session ran under.
    - ``timestamp`` (str): export time, ISO8601 recommended. Caller-supplied
      — see the module docstring for why this module never touches
      wall-clock time itself.
    - ``files_changed`` (list[str], optional): paths written this session.
    - ``session_id`` (str, optional): stable id for the record.

    Returns the path written, or ``None`` if disabled, errored, or there
    were no messages to export (nothing happened -> nothing worth writing).
    """
    if not _is_enabled(meta.get("enabled")):
        return None
    if meta.get("error"):
        return None
    messages = _session_messages(context)
    if not messages:
        return None

    system_prompt = getattr(context, "system_prompt", "") or ""
    record = {
        "source": "spark-code",
        "session_id": meta.get("session_id", ""),
        "system_prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "model": meta.get("model"),
        "exported_at": meta.get("timestamp"),
        "messages": _scrub_messages(messages),
        "files_changed": list(meta.get("files_changed") or []),
    }

    expanded_dir = os.path.expanduser(path_dir or DEFAULT_CORPUS_DIR)
    os.makedirs(expanded_dir, exist_ok=True)
    out_path = os.path.join(expanded_dir, "spark-code-sessions.jsonl")
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _APPEND_LOCK, open(out_path, "a", encoding="utf-8") as f:
        f.write(line)
    return out_path
