"""Tests for opt-in session -> training corpus export (Phase 4 Task 6).

No network, no wall-clock reads inside corpus.py — every test builds its own
``meta`` dict (including ``timestamp``) and a minimal stand-in for
``Context`` (a plain object with ``.system_prompt``/``.messages``) so these
stay fast, deterministic unit tests.
"""

import json
import os

import spark_code.cli as cli_mod
from spark_code.context import Context
from spark_code.corpus import export_session


class FakeContext:
    """Minimal stand-in for spark_code.context.Context — only the two
    attributes export_session actually reads."""

    def __init__(self, system_prompt="You are Spark.", messages=None):
        self.system_prompt = system_prompt
        self.messages = messages if messages is not None else []


def _meta(**overrides):
    base = {
        "enabled": True,
        "error": None,
        "model": "qwen3.5:122b",
        "timestamp": "2026-07-07T12:00:00+00:00",
        "files_changed": ["a.py"],
        "session_id": "sess-1",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Step 1 (plan-specified): failing tests
# ---------------------------------------------------------------------------


def test_export_writes_parseable_jsonl_with_expected_keys(tmp_path):
    context = FakeContext(messages=[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ])
    path = export_session(context, str(tmp_path), _meta())
    assert path is not None
    assert os.path.exists(path)

    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["source"] == "spark-code"
    assert record["model"] == "qwen3.5:122b"
    assert record["exported_at"] == "2026-07-07T12:00:00+00:00"
    assert record["files_changed"] == ["a.py"]
    assert record["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert "system_prompt_hash" in record
    assert isinstance(record["system_prompt_hash"], str)
    assert len(record["system_prompt_hash"]) == 64  # sha256 hex digest
    assert record["session_id"] == "sess-1"


def test_disabled_returns_none_writes_nothing(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    path = export_session(context, str(tmp_path), _meta(enabled=False))
    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_errored_session_is_skipped(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    path = export_session(context, str(tmp_path), _meta(error="stream error: 503"))
    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_no_messages_is_skipped(tmp_path):
    context = FakeContext(messages=[])
    path = export_session(context, str(tmp_path), _meta())
    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_secret_shaped_string_is_scrubbed(tmp_path):
    fake_key = "sk-" + ("a" * 40)
    fake_hex = "d41d8cd98f00b204e9800998ecf8427e" * 2  # 64 hex chars
    context = FakeContext(messages=[
        {"role": "user", "content": f"my key is {fake_key} and hash {fake_hex}"},
        {"role": "assistant", "content": "got it, redacting"},
    ])
    path = export_session(context, str(tmp_path), _meta())
    with open(path) as f:
        record = json.loads(f.readline())

    blob = json.dumps(record)
    assert fake_key not in blob
    assert fake_hex not in blob
    assert "[REDACTED]" in record["messages"][0]["content"]


def test_secret_scrub_covers_tool_call_arguments(tmp_path):
    fake_key = "sk-" + ("b" * 40)
    context = FakeContext(messages=[
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "bash",
                          "arguments": json.dumps({"command": f"curl -H 'Authorization: {fake_key}'"})}},
        ]},
    ])
    path = export_session(context, str(tmp_path), _meta())
    with open(path) as f:
        record = json.loads(f.readline())
    assert fake_key not in json.dumps(record)


def test_dir_created_if_absent(tmp_path):
    target = tmp_path / "nested" / "corpus-dir"
    assert not target.exists()
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    path = export_session(context, str(target), _meta())
    assert path is not None
    assert target.exists()
    assert os.path.exists(path)


def test_expands_user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    path = export_session(context, "~/training-corpus/spark-code", _meta())
    assert path is not None
    assert path.startswith(str(tmp_path))


def test_appends_multiple_sessions_as_separate_lines(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "one"}])
    export_session(context, str(tmp_path), _meta(session_id="a"))
    export_session(context, str(tmp_path), _meta(session_id="b"))
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    with open(files[0]) as f:
        lines = [json.loads(ln) for ln in f]
    assert len(lines) == 2
    assert [ln["session_id"] for ln in lines] == ["a", "b"]


def test_image_content_is_not_embedded_raw(tmp_path):
    context = FakeContext(messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "what's in this screenshot?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ])
    path = export_session(context, str(tmp_path), _meta())
    with open(path) as f:
        record = json.loads(f.readline())
    blob = json.dumps(record)
    assert "AAAA" not in blob
    assert "base64" not in blob


# ---------------------------------------------------------------------------
# Fix 2 — sanitize orphaned tool_calls via context.get_messages()
# ---------------------------------------------------------------------------


def _orphaned_tool_call_ids(messages: list[dict]) -> set:
    """Return assistant tool_call ids with no matching role:tool reply."""
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    ids = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            ids.add(tc.get("id"))
    return ids - answered


def test_orphaned_tool_calls_are_sanitized_before_export(tmp_path):
    # A real Context left in the interrupted state: an assistant tool_calls
    # message with NO following role:tool reply. export_session must go
    # through get_messages() (which runs sanitize_orphaned_tool_calls) so the
    # serialized sequence is valid — every tool_call id answered.
    context = Context(system_prompt="sys")
    context.add_user("do the thing")
    context.add_assistant_tool_calls(
        [{"id": "call-1", "name": "bash", "arguments": {"command": "ls"}}],
        content="running it",
    )
    # deliberately NO add_tool_result — this is the orphaned/interrupted state
    path = export_session(context, str(tmp_path), _meta())
    with open(path) as f:
        record = json.loads(f.readline())
    assert record["messages"], "expected messages exported"
    # No system message leaked inline (hash carries it instead).
    assert all(m.get("role") != "system" for m in record["messages"])
    # Sanitize backfilled a synthetic tool result → no orphans.
    assert _orphaned_tool_call_ids(record["messages"]) == set()


def test_export_does_not_inline_system_message(tmp_path):
    context = Context(system_prompt="the big system prompt")
    context.add_user("hi")
    context.add_assistant("hello")
    path = export_session(context, str(tmp_path), _meta())
    with open(path) as f:
        record = json.loads(f.readline())
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["user", "assistant"]
    # The system prompt text is NOT stored inline, only its hash.
    assert "the big system prompt" not in json.dumps(record["messages"])


# ---------------------------------------------------------------------------
# Fix 3 — strict boolean coercion for the opt-in flag
# ---------------------------------------------------------------------------


def test_enabled_string_false_does_not_export(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    # A hand-edited quoted "false" is a truthy STRING — must NOT enable export.
    assert export_session(context, str(tmp_path), _meta(enabled="false")) is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_various_falsey_strings_do_not_export(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    for val in ("false", "no", "off", "0", "", 0, None):
        assert export_session(context, str(tmp_path), _meta(enabled=val)) is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_true_variants_do_export(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    for val in (True, "true", "TRUE", "1", "yes"):
        assert export_session(context, str(tmp_path), _meta(enabled=val)) is not None


# ---------------------------------------------------------------------------
# Fix 4 — wider secret scrub
# ---------------------------------------------------------------------------


def test_wider_secret_shapes_are_scrubbed(tmp_path):
    secrets = [
        "ya29.z_-" + "AbCdEf0" * 6,               # Google OAuth access token
        "xoxb-" + "zzz1234567890abc",             # Slack bot token
        "npm_" + "zZ" + "AbCdEfGhIjKlMnOpQrSt",   # npm token (non-hex chars)
        "sk_live_" + "zABC1234DEFG5678HIJK",      # Stripe secret key
        "Bearer " + "zzzzABCDefgh1234ijkl5678mnop",  # generic bearer token
        "ghp_" + "zZ" + "AbCdEfGhIjKlMnOpQrSt",   # GitHub token (existing pattern)
    ]
    for secret in secrets:
        context = FakeContext(messages=[
            {"role": "user", "content": f"here it is: {secret} <-- please store"},
        ])
        path = export_session(context, str(tmp_path), _meta())
        with open(path) as f:
            lines = f.readlines()
        blob = lines[-1]
        # strip the 'Bearer '/'ya29.' non-secret prefixes when asserting the
        # opaque part is gone
        opaque = secret.split(" ")[-1]
        assert opaque not in blob, f"unscrubbed secret: {secret!r}"
        assert "[REDACTED]" in blob


# ---------------------------------------------------------------------------
# Fix 1 — interrupted sessions must NOT export as "clean" (CLI wiring)
# ---------------------------------------------------------------------------


class FakeAgent:
    """Minimal stand-in for Agent — only the attrs _export_corpus_session reads."""

    def __init__(self, cancelled=False, interrupted=False, stream_error=None,
                 written=None):
        self._cancelled = cancelled
        self._session_interrupted = interrupted
        self._last_stream_error = stream_error
        self._verify_written_paths = written or []


def _corpus_config(tmp_path, enabled=True):
    return {"corpus": {"export_enabled": enabled, "dir": str(tmp_path)},
            "model": {"name": "qwen3.5:122b"}}


def test_cli_hook_skips_cancelled_session(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "partial work"}])
    agent = FakeAgent(cancelled=True, stream_error=None)  # Ctrl+C: no stream error
    cli_mod._export_corpus_session(_corpus_config(tmp_path), context, agent,
                                   error=None)
    assert list(tmp_path.iterdir()) == []


def test_cli_hook_skips_session_interrupted_on_earlier_turn(tmp_path):
    # Last turn finished clean (_cancelled False, no error) but an EARLIER turn
    # was interrupted → the persistent flag must still block export.
    context = FakeContext(messages=[{"role": "user", "content": "later clean turn"}])
    agent = FakeAgent(cancelled=False, interrupted=True, stream_error=None)
    cli_mod._export_corpus_session(_corpus_config(tmp_path), context, agent,
                                   error=None)
    assert list(tmp_path.iterdir()) == []


def test_cli_hook_skips_stream_error_session(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    agent = FakeAgent(stream_error="API error (503)")
    cli_mod._export_corpus_session(_corpus_config(tmp_path), context, agent,
                                   error="API error (503)")
    assert list(tmp_path.iterdir()) == []


def test_cli_hook_exports_clean_session(tmp_path):
    context = FakeContext(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    agent = FakeAgent(written=["a.py"])
    cli_mod._export_corpus_session(_corpus_config(tmp_path), context, agent,
                                   error=None)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    with open(files[0]) as f:
        record = json.loads(f.readline())
    assert record["files_changed"] == ["a.py"]
    assert record["model"] == "qwen3.5:122b"


def test_cli_hook_respects_string_false_config(tmp_path):
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    agent = FakeAgent()
    cli_mod._export_corpus_session(_corpus_config(tmp_path, enabled="false"),
                                   context, agent, error=None)
    assert list(tmp_path.iterdir()) == []


def test_cli_hook_never_raises_on_bad_agent(tmp_path):
    # Defensive: a totally malformed agent must not crash teardown.
    context = FakeContext(messages=[{"role": "user", "content": "hi"}])
    cli_mod._export_corpus_session(_corpus_config(tmp_path), context, object(),
                                   error=None)
