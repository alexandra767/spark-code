"""Tests for spark_code.context.Context."""


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
        before = len(ctx.messages)
        ctx.compact(keep_recent=6)

        # compact() injects the summary as a "user" message (not "system") because
        # get_messages() already prepends the system prompt — see context.py:307.
        has_summary = any(
            "Conversation Summary" in (m.get("content") or "") for m in ctx.messages
        )
        assert has_summary, "compact() should leave a summary marker in messages"
        assert len(ctx.messages) < before, "compact() should reduce message count"
        # The last 6 messages should be the most recent originals (preserved verbatim)
        assert ctx.messages[-1]["content"] == "response 19"
        assert ctx.messages[-2]["content"] == "message 19"

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


class TestCompactionSafety:
    """Regression: compaction must not orphan tool results (invalid history)."""

    def _msg_tool_call(self, cid, name):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": cid, "type": "function",
                                "function": {"name": name, "arguments": "{}"}}]}

    def test_recent_never_starts_with_orphan_tool_result(self):
        ctx = Context(max_tokens=0)
        # Build a history where the naive keep_recent boundary would land
        # between an assistant tool_calls message and its tool result.
        ctx.messages = [
            {"role": "user", "content": "do a thing"},
            self._msg_tool_call("c1", "read_file"),
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "data"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "another"},
            self._msg_tool_call("c2", "bash"),
            {"role": "tool", "tool_call_id": "c2", "name": "bash", "content": "ok"},
        ]
        ctx.compact(keep_recent=1)  # naive slice would keep only the last tool msg
        # After compaction, no tool message may be preceded by a non-tool_calls msg
        for i, m in enumerate(ctx.messages):
            if m.get("role") == "tool":
                prev = ctx.messages[i - 1]
                assert prev.get("role") == "assistant" and prev.get("tool_calls"), (
                    f"orphaned tool result at index {i}")


class TestSafeSplitIndexAllToolTail:
    """Audit 2026-07-06 item 9: _safe_split_index only stopped its backward walk
    on a non-tool role, so a (hand-corrupted / pathological) history whose head
    is all tool messages walked idx to -1 and beyond — Python negative indexing
    reads from the TAIL and the loop eventually raises IndexError. The walk must
    clamp at 0."""

    def _tool(self, cid):
        return {"role": "tool", "tool_call_id": cid, "name": "bash",
                "content": "out"}

    def test_all_tool_list_returns_zero_no_exception(self):
        ctx = Context(max_tokens=0)
        ctx.messages = [self._tool("a"), self._tool("b"), self._tool("c")]
        assert ctx._safe_split_index(2) == 0

    def test_tool_head_clamps_at_zero(self):
        ctx = Context(max_tokens=0)
        ctx.messages = [self._tool("a"),
                        {"role": "assistant", "content": "hi"}]
        assert ctx._safe_split_index(1) == 1  # stops on the non-tool message
        assert ctx._safe_split_index(0) == 0  # never goes negative


class TestNarrationPreserved:
    def test_tool_calls_keep_narration_content(self):
        ctx = Context()
        ctx.add_assistant_tool_calls(
            [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
            content="Let me list the files.")
        assert ctx.messages[-1]["content"] == "Let me list the files."
        assert ctx.messages[-1]["tool_calls"][0]["function"]["name"] == "bash"

    def test_tool_calls_without_content_is_none(self):
        ctx = Context()
        ctx.add_assistant_tool_calls(
            [{"id": "c1", "name": "bash", "arguments": {}}])
        assert ctx.messages[-1]["content"] is None


class TestSchemaReserve:
    """Audit 2026-07-06 item 4: estimate_tokens ignored the tool-schema JSON
    sent on EVERY request (2-4K tokens) and used an optimistic 4 chars/token, so
    the 0.8/0.9 auto-compact thresholds fired too late and long sessions 400'd on
    context length. The estimate now adds a schema reserve plus a tighter ratio."""

    def test_schema_reserve_added_to_estimate(self):
        without = Context(system_prompt="x")
        without.add_user("hello world")
        base = without.estimate_tokens()

        reserved = Context(system_prompt="x")
        reserved.add_user("hello world")
        reserved.schema_reserve_tokens = 900
        assert reserved.estimate_tokens() >= base + 900

    def test_reserve_defaults_to_zero(self):
        ctx = Context(system_prompt="x")
        assert ctx.schema_reserve_tokens == 0


class TestOrphanedToolCallSanitize:
    """Audit 2026-07-06 item 2: an interrupt (Ctrl+C) between an assistant
    tool_calls message and its tool results — or reloading a session saved in
    that state — leaves tool_call ids with no matching role:"tool" reply, which
    OpenAI-compatible servers reject with a 400. get_messages() and load() must
    backfill synthetic results so the message sequence stays valid."""

    def _assistant_tc(self, *ids_names):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": cid, "type": "function",
                                "function": {"name": name, "arguments": "{}"}}
                               for cid, name in ids_names]}

    def _assert_valid(self, messages):
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                answered = set()
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    answered.add(messages[j].get("tool_call_id"))
                    j += 1
                for tc in m["tool_calls"]:
                    assert tc["id"] in answered, f"orphaned tool_call {tc['id']}"

    def test_get_messages_backfills_orphaned_tail(self):
        ctx = Context()
        ctx.messages = [
            {"role": "user", "content": "read three files"},
            self._assistant_tc(("a", "read_file"), ("b", "read_file"),
                               ("c", "read_file")),
            {"role": "tool", "tool_call_id": "a", "name": "read_file", "content": "A"},
            {"role": "tool", "tool_call_id": "b", "name": "read_file", "content": "B"},
            # "c" was interrupted before its result was recorded.
        ]
        msgs = ctx.get_messages()
        self._assert_valid(msgs[1:])  # skip the prepended system message
        c_results = [m for m in msgs if m.get("role") == "tool"
                     and m.get("tool_call_id") == "c"]
        assert len(c_results) == 1
        assert "interrupted" in c_results[0]["content"].lower()

    def test_get_messages_backfills_zero_result_orphan(self):
        # The whole tool batch was interrupted before ANY result was recorded.
        ctx = Context()
        ctx.messages = [
            {"role": "user", "content": "run it"},
            self._assistant_tc(("x", "bash")),
        ]
        msgs = ctx.get_messages()
        self._assert_valid(msgs[1:])

    def test_load_sanitizes_orphaned_session(self, tmp_path):
        ctx = Context()
        ctx.messages = [
            {"role": "user", "content": "do it"},
            self._assistant_tc(("x", "bash")),
        ]
        path = tmp_path / "orphan.json"
        ctx.save(str(path))
        ctx2 = Context()
        assert ctx2.load(str(path)) is True
        self._assert_valid(ctx2.messages)
        assert any(m.get("role") == "tool" and m.get("tool_call_id") == "x"
                   for m in ctx2.messages)

    def test_sanitize_is_idempotent(self):
        ctx = Context()
        ctx.messages = [
            {"role": "user", "content": "go"},
            self._assistant_tc(("q", "glob")),
        ]
        ctx.get_messages()
        n_after_first = len(ctx.messages)
        ctx.get_messages()
        assert len(ctx.messages) == n_after_first


class TestContextLeft:
    """Real (server-reported) usage should drive the context meter instead of
    the char-based estimate, which drifts from the model's actual tokenizer."""

    def test_context_left_uses_server_usage(self):
        ctx = Context()
        ctx.record_usage(16384)
        assert abs(ctx.context_left(32768) - 0.5) < 0.01

    def test_context_left_falls_back_to_estimate(self):
        ctx = Context()
        ctx.add_user("hi")
        assert ctx.context_left(32768) > 0.9

    def test_recorded_usage_cleared_on_compact_and_clear(self):
        ctx = Context()
        ctx.record_usage(30000)
        ctx.clear()
        assert ctx.last_prompt_tokens is None

    def test_recorded_usage_cleared_on_compact(self):
        ctx = Context(system_prompt="sys")
        for i in range(20):
            ctx.add_user(f"message {i}")
            ctx.add_assistant(f"response {i}")
        ctx.record_usage(30000)
        ctx.compact(keep_recent=6)
        assert ctx.last_prompt_tokens is None

    def test_context_left_handles_zero_window(self):
        ctx = Context()
        ctx.record_usage(100)
        assert ctx.context_left(0) == 0.0


class TestEstimateTokensImages:
    def test_image_payload_counted(self):
        ctx = Context(system_prompt="")
        big = "A" * 40000  # ~fake base64 image payload
        ctx.add_user_with_image("look", big)
        # Must be counted (not ~0), so auto-compaction can trigger on images.
        assert ctx.estimate_tokens() > 5000


class TestEstimateCompactableTokens:
    """Anti-thrash support (review round 1): _maybe_compact must know how much
    compact() could actually remove — the system prompt and the keep_recent
    tail survive compaction, so only the removable slice counts."""

    def test_zero_when_history_at_or_under_keep_recent(self):
        ctx = Context()
        for i in range(3):
            ctx.add_user(f"m{i}")
        assert ctx.estimate_compactable_tokens(6) == 0

    def test_counts_only_the_removable_slice(self):
        # Giant system prompt must NOT count — it survives compaction.
        ctx = Context(system_prompt="S" * 100000)
        for i in range(10):
            ctx.add_user("x" * 350)  # ~100 tokens each
        est = ctx.estimate_compactable_tokens(6)
        # 10 messages, keep 6 → 4 removable * ~100 tokens each.
        assert 300 <= est <= 500

    def test_respects_safe_split_boundary(self):
        # The removable slice ends at the SAFE split, exactly like compact().
        ctx = Context()
        ctx.messages = [
            {"role": "user", "content": "u" * 700},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "bash",
             "content": "t" * 700},
        ]
        # keep_recent=1 → naive split lands ON the tool result; the safe split
        # walks back past its assistant parent → only message 0 is removable.
        est = ctx.estimate_compactable_tokens(1)
        assert 150 <= est <= 250  # ~700 chars / 3.5


class TestPrefixCacheOrdering:
    """T2 Step 7: system-prompt content must be ordered static-first so the
    server's prefix cache keeps the longest stable prefix. The volatile
    platform slot (cwd now, git status / todo state later) goes LAST."""

    def test_static_precedes_volatile_platform(self):
        ctx = Context(system_prompt="STATIC_BASE_PROMPT",
                      platform_prompt="CWD: /tmp/x (volatile)",
                      provider_prompt="PROVIDER_STATIC")
        combined = ctx.get_messages()[0]["content"]
        i_sys = combined.index("STATIC_BASE_PROMPT")
        i_prov = combined.index("PROVIDER_STATIC")
        i_plat = combined.index("CWD: /tmp/x (volatile)")
        assert i_sys < i_prov < i_plat, "static base + provider must precede volatile platform"

    def test_reorder_preserves_all_content_and_tokens(self):
        ctx = Context(system_prompt="BASE", platform_prompt="PLAT",
                      provider_prompt="PROV")
        combined = ctx.get_messages()[0]["content"]
        assert "BASE" in combined and "PLAT" in combined and "PROV" in combined
        # Reordering a join changes order, not content length → no token delta.
        assert ctx.estimate_tokens() == Context(
            system_prompt="BASE", platform_prompt="PLAT",
            provider_prompt="PROV").estimate_tokens()


class TestCompactSummaryParam:
    """T2: compact() accepts a model-generated summary that replaces the
    mechanical digest, while the mechanical digest stays as the offline/test
    fallback and all boundary/pinned safety still holds."""

    def _long_history(self, ctx):
        for i in range(20):
            ctx.add_user(f"user message {i}")
            ctx.add_assistant(f"assistant reply {i} with some detail")

    def test_provided_summary_used_verbatim(self):
        ctx = Context(max_tokens=0)
        self._long_history(ctx)
        marker = "CUSTOM_SUMMARY_MARKER_XYZ preserve the DB migration decision"
        ctx.compact(keep_recent=6, summary=marker)
        # The first (summary) message must carry the provided text verbatim.
        assert ctx.messages[0]["role"] == "user"
        assert marker in ctx.messages[0]["content"]

    def test_provided_summary_skips_mechanical_digest(self):
        ctx = Context(max_tokens=0)
        # Give the history a distinctive tool result the mechanical digest would
        # surface under "### Tools used". A model summary must NOT include that.
        ctx.add_user("do a thing")
        ctx.add_assistant_tool_calls(
            [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}])
        ctx.add_tool_result("c1", "bash", "ok")
        self._long_history(ctx)
        ctx.compact(keep_recent=6, summary="MODEL SUMMARY ONLY")
        assert "### Tools used" not in ctx.messages[0]["content"]
        assert "MODEL SUMMARY ONLY" in ctx.messages[0]["content"]

    def test_none_summary_uses_mechanical_digest(self):
        ctx = Context(max_tokens=0)
        self._long_history(ctx)
        ctx.compact(keep_recent=6, summary=None)
        # Mechanical digest header still present when no model summary supplied.
        assert "Conversation Summary" in ctx.messages[0]["content"]

    def test_model_summary_preserves_tool_boundary(self):
        """Boundary safety (Phase 0) must still hold when a MODEL summary is
        used: recent[] may never begin with an orphaned tool result."""
        ctx = Context(max_tokens=0)
        ctx.messages = [
            {"role": "user", "content": "do a thing"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file",
             "content": "data"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "another"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c2", "type": "function",
                             "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c2", "name": "bash", "content": "ok"},
        ]
        ctx.compact(keep_recent=1, summary="MODEL SUMMARY TEXT")
        assert "MODEL SUMMARY TEXT" in ctx.messages[0]["content"]
        for i, m in enumerate(ctx.messages):
            if m.get("role") == "tool":
                prev = ctx.messages[i - 1]
                assert prev.get("role") == "assistant" and prev.get("tool_calls"), (
                    f"orphaned tool result at index {i}")

    def test_compact_leaves_pinned_system_prompt_intact(self):
        """Pinned files live in system_prompt (not messages), so compaction —
        which only touches messages — must leave them untouched (survival)."""
        pinned_block = "<!--spark:pinned-->\nimportant pinned file\n<!--/spark:pinned-->"
        ctx = Context(system_prompt=f"base prompt\n\n{pinned_block}", max_tokens=0)
        self._long_history(ctx)
        ctx.compact(keep_recent=6, summary="SUMMARY")
        assert pinned_block in ctx.system_prompt
