"""Cloud provider API key setup + storage (Phase 5 Task 2).

Layers a guided ``/setkey`` flow + a small preset table over the existing
``${ENV}``-interpolated provider config (config.py's
``expand_env_vars``/``resolve_provider``) so plugging in a public cloud key
(Anthropic, OpenAI, Gemini, OpenRouter) doesn't require hand-editing
``config.yaml`` or exporting a shell variable — this is the "primary/utility/
fallback model when the DGX is down or she's away from home" feature.

Storage: ``~/.spark/keys`` — a small JSON object (``{"openrouter": "sk-or-...",
...}``), created on first :func:`save_key` and ``chmod 0o600`` on every write.
``~/.spark`` is not a git repository, so this file is never at risk of being
committed; see :func:`keys_path`'s docstring for the belt-and-suspenders
``.gitignore`` entry this module adds if a ``.gitignore`` already exists there
(e.g. if the user ever turns ``~/.spark`` into a repo for dotfile syncing).

SECURITY: nothing in this module ever logs, prints, or returns a key in full.
:func:`mask` is the only display-safe form — every caller that shows a key to
the user (the ``/setkey`` confirmation, any future ``/doctor`` row) must go
through it.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

# Each preset is exactly the fields needed to (a) prompt for the right env var
# name / doc link and (b) seed a new `providers.<name>` config block (endpoint
# + model only — the key itself lives in the keys file, never in config.yaml).
PROVIDER_PRESETS: dict[str, dict] = {
    "anthropic": {
        "endpoint": "https://api.anthropic.com/v1/",
        "model": "claude-sonnet-4-5",
        "key_env": "ANTHROPIC_API_KEY",
        "doc_url": "https://console.anthropic.com/settings/keys",
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "key_env": "OPENAI_API_KEY",
        "doc_url": "https://platform.openai.com/api-keys",
    },
    "gemini": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "key_env": "GEMINI_API_KEY",
        "doc_url": "https://aistudio.google.com/apikey",
    },
    "openrouter": {
        "endpoint": "https://openrouter.ai/api/v1",
        "model": "openrouter/auto",
        "key_env": "OPENROUTER_API_KEY",
        "doc_url": "https://openrouter.ai/keys",
    },
}

_KEYS_RELATIVE_PATH = os.path.join(".spark", "keys")

# An unresolved `${VAR}` placeholder (env var wasn't set at expand_env_vars
# time — see config.py's _expand_env_str, which leaves it literal rather than
# blanking it). resolve_provider_key must NOT treat this as "an explicit key
# is configured" or the keys-file fallback would never kick in for a provider
# whose config.yaml references an env var that isn't actually exported.
_UNRESOLVED_ENV_RE = re.compile(r"\$\{[^}]+\}")


def keys_path() -> str:
    """Absolute path to the keys store: ``~/.spark/keys``.

    Never committed: ``~/.spark`` isn't a git repository. If the user ever
    does turn it into one (e.g. dotfile syncing), :func:`save_key` also drops
    a ``.gitignore`` entry for the bare ``keys`` filename as a second line of
    defense — see :func:`_ensure_gitignored`.
    """
    return os.path.expanduser(os.path.join("~", _KEYS_RELATIVE_PATH))


def _ensure_gitignored(directory: str) -> None:
    """If ``directory`` already has a ``.gitignore``, make sure it covers ``keys``.

    No-op (and never raises) if there's no ``.gitignore`` there — ``~/.spark``
    isn't a repo by default, so this is only a safety net for the case where
    it becomes one. Best-effort: any I/O error is swallowed, since a missing
    gitignore entry is a soft nice-to-have, not the actual security boundary
    (chmod 600 + never-in-config-file are).
    """
    gitignore_path = os.path.join(directory, ".gitignore")
    try:
        if not os.path.exists(gitignore_path):
            return
        with open(gitignore_path, encoding="utf-8") as f:
            existing = f.read()
        if "keys" in existing.splitlines():
            return
        with open(gitignore_path, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("keys\n")
    except OSError:
        logger.warning("Could not update %s to gitignore the keys file", gitignore_path)


def load_keys() -> dict[str, str]:
    """Read the keys file. Returns ``{}`` if absent/unreadable/malformed.

    Never raises — a corrupt or missing keys file must degrade to "no cloud
    keys configured", not crash config loading (which calls this on every
    session start via config.resolve_provider).
    """
    path = keys_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, str) and v}


def save_key(provider: str, value: str) -> str:
    """Write/update ``provider``'s key in the keys file; chmod 600; return the path.

    Creates ``~/.spark/`` if absent. Preserves any other providers' keys
    already in the file (read-modify-write). Raises on genuine I/O failure
    (e.g. permission denied creating the directory) — callers (the ``/setkey``
    command) are expected to catch and report that, rather than this module
    silently swallowing a save the user explicitly asked for.
    """
    path = keys_path()
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    keys = load_keys()
    keys[provider] = value

    # Create with 0o600 from the start (umask-independent) rather than
    # chmod-after-write, so the key is never briefly world/group-readable on
    # disk between write and chmod.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2, sort_keys=True)
        f.write("\n")
    # Belt-and-suspenders: an existing file (e.g. pre-dating this feature, or
    # with a looser umask) might not already be 0o600 — force it explicitly.
    os.chmod(path, 0o600)

    _ensure_gitignored(directory)
    return path


def mask(value: str) -> str:
    """Display-safe form of an API key: never the full value.

    Long keys (the normal case — every real provider key is well over 8
    chars): first 3 + ``…`` + last 4, e.g. ``sk-…wxyz``. Short/edge-case
    values are fully masked (no characters revealed) rather than risk
    overlap between the "first 3" and "last 4" windows leaking the whole
    thing.
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "…" * min(len(value), 4)
    return f"{value[:3]}…{value[-4:]}"


def resolve_provider_key(provider_name: str, provider_conf: dict, keys: dict) -> str:
    """Resolve the API key to use for ``provider_name``.

    Precedence:
    1. An explicit, already-``${ENV}``-resolved ``api_key`` in
       ``provider_conf`` (config.py's ``expand_env_vars`` has already run by
       the time ``resolve_provider`` calls this — see config.py). A literal
       unresolved placeholder (the env var wasn't set) does NOT count as
       explicit — it falls through to the keys file instead of being sent to
       the provider as-is.
    2. This provider's entry in the keys file (``~/.spark/keys``, set via
       ``/setkey``).
    3. ``""`` — no key found anywhere; same as today's "no auth header sent".
    """
    explicit = (provider_conf or {}).get("api_key", "")
    if isinstance(explicit, str) and explicit and not _UNRESOLVED_ENV_RE.search(explicit):
        return explicit
    return (keys or {}).get(provider_name, "") or ""
