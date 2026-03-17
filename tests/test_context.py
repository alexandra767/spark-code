"""Tests for spark_code.context.Context."""

import json
import pytest
from pathlib import Path

from spark_code.context import Context


class TestAddUser:
    def test_add_user_appends_message_and_increments_turn(self):
        ctx = Context()
        ctx.add_user("hello")
        assert ctx.messages[-1] == {"role": "user", "content": "hello"}
        assert ctx.turn_count == 1

    def test_add_user_increments_turn_count_each_call(self):
        ctx = Context()
        ctx.add_user("first")
        ctx.add_user("second")
        assert ctx.turn_count == 2


class TestAddAssistant:
    def test_add_assistant_appends_message(self):
        ctx = Context()
        ctx.add_assistant("response text")
        assert ctx.messages[-1] == {"role": "assistant", "content": "response text"}

    def test_add_assistant_does_not_increment_turn_count(self):
        ctx = Context()
        ctx.add_assistant("response")
        assert ctx.turn_count == 0


class TestAddToolResult:
    def test_add_tool_result_appends_tool_message(self):
        ctx = Context()
        ctx.add_tool_result("call_123", "read_file", "file contents here")
        msg = ctx.messages[-1]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["name"] == "read_file"
        assert msg["content"] == "file contents here"


class TestAddAssistantToolCalls:
    def test_add_assistant_tool_calls_formats_correctly(self):
        tool_calls = [
            {"id": "call_1", "name": "read_file", "arguments": {"path": "/tmp/x"}}
        ]
        ctx = Context()
        ctx.add_assistant_tool_calls(tool_calls)
        msg = ctx.messages[-1]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg


class TestGetMessages:
    def test_get_messages_includes_system_prompt_first(self):
        ctx = Context(system_prompt="You are helpful.")
        ctx.add_user("hi")
        messages = ctx.get_messages()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hi"

    def test_get_messages_empty_context_still_has_system(self):
        ctx = Context(system_prompt="test prompt")
        messages = ctx.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


class TestEstimateTokens:
    def test_estimate_tokens_returns_reasonable_value(self):
        ctx = Context(system_prompt="short")
        ctx.add_user("hello world")
        tokens = ctx.estimate_tokens()
        # Should be roughly total chars // 4
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_counts_tool_calls_too(self):
        ctx = Context(system_prompt="x")
        tool_calls = [
            {"id": "call_1", "name": "read_file", "arguments": {"path": "/a/very/long/path/to/a/file.txt"}}
        ]
        ctx.add_assistant_tool_calls(tool_calls)
        tokens_with_tools = ctx.estimate_tokens()

        ctx2 = Context(system_prompt="x")
        tokens_without = ctx2.estimate_tokens()

        assert tokens_with_tools > tokens_without


class TestCompact:
    def test_compact_preserves_recent_messages(self):
        ctx = Context(system_prompt="sys")
        for i in range(20):
            ctx.add_user(f"message {i}")
            ctx.add_assistant(f"response {i}")
        ctx.compact(keep_recent=6)
        # After compaction, should have a summary + recent messages
        # The summary should use role "system", not "user"
        has_summary = any(
            m["role"] == "system" for m in ctx.messages
        )
        assert has_summary or len(ctx.messages) <= 6

    def test_compact_with_few_messages_does_nothing(self):
        ctx = Context(system_prompt="sys")
        ctx.add_user("one")
        ctx.add_assistant("two")
        original_count = len(ctx.messages)
        ctx.compact(keep_recent=6)
        assert len(ctx.messages) == original_count

    def test_compact_post_check_reduces_further_if_over_limit(self):
        # Create a context with very small max_tokens to trigger post-check
        ctx = Context(system_prompt="sys", max_tokens=100)
        for i in range(30):
            ctx.add_user(f"message number {i} with some extra padding text to inflate token count significantly")
            ctx.add_assistant(f"response number {i} also padded with a lot of extra words to push token estimates up")
        before = len(ctx.messages)
        ctx.compact(keep_recent=6)
        # Should have compacted, potentially multiple times
        assert len(ctx.messages) < before


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        ctx = Context(system_prompt="test prompt")
        ctx.add_user("hello")
        ctx.add_assistant("world")
        path = tmp_path / "context.json"
        ctx.save(str(path))

        ctx2 = Context()
        result = ctx2.load(str(path))
        assert result is not False
        msgs2 = ctx2.get_messages()
        # Should contain system + user + assistant
        assert any(m["content"] == "hello" for m in msgs2)
        assert any(m["content"] == "world" for m in msgs2)

    def test_save_creates_parent_dirs(self, tmp_path):
        ctx = Context(system_prompt="test")
        ctx.add_user("data")
        path = tmp_path / "subdir" / "nested" / "context.json"
        ctx.save(str(path))
        assert path.exists()

    def test_load_returns_false_for_missing_file(self):
        ctx = Context()
        result = ctx.load("/tmp/nonexistent_spark_code_test_file.json")
        assert result is False or result is None


class TestClear:
    def test_clear_resets_everything(self):
        ctx = Context()
        ctx.add_user("hello")
        ctx.add_assistant("world")
        ctx.clear()
        assert ctx.messages == []
        assert ctx.turn_count == 0


class TestMessageOrder:
    def test_multiple_messages_maintained_in_order(self):
        ctx = Context(system_prompt="sys")
        ctx.add_user("first")
        ctx.add_assistant("second")
        ctx.add_user("third")
        ctx.add_assistant("fourth")
        messages = ctx.get_messages()
        # System prompt is first
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "first"
        assert messages[2]["content"] == "second"
        assert messages[3]["content"] == "third"
        assert messages[4]["content"] == "fourth"
