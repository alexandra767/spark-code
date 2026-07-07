"""Tests for opt-in session -> training corpus export (Phase 4 Task 6).

No network, no wall-clock reads inside corpus.py — every test builds its own
``meta`` dict (including ``timestamp``) and a minimal stand-in for
``Context`` (a plain object with ``.system_prompt``/``.messages``) so these
stay fast, deterministic unit tests.
"""

import json
import os

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
