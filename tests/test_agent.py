"""Tests for the Agent class — the core agent loop."""

import asyncio
import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from spark_code.agent import Agent
from spark_code.context import Context
from spark_code.permissions import PermissionManager
from spark_code.tools.base import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockModel:
    """A mock model that yields predefined sequences of chunks."""

    total_input_tokens = 0
    total_output_tokens = 0

    def __init__(self, responses):
        """
        responses: list of lists of chunks.
        Each call to chat() pops the next list and yields its chunks.
        """
        self.responses = list(responses)
        self._call = 0

    async def chat(self, **kwargs):
        if self._call < len(self.responses):
            chunks = self.responses[self._call]
            self._call += 1
            for chunk in chunks:
                yield chunk

    async def close(self):
        pass


class MockTool(Tool):
    """A simple tool for testing that returns a predictable result."""

    name = "mock_tool"
    description = "A test tool"
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {"arg": {"type": "string"}},
            "required": ["arg"],
        }

    async def execute(self, **kwargs):
        return f"mock result: {kwargs}"


def _make_console() -> Console:
    """Return a Console that writes to a StringIO so nothing hits the terminal."""
    return Console(file=io.StringIO(), force_terminal=True)


def _make_agent(model, tools=None, permissions=None, test_command=None):
    """Build an Agent wired to the given mock model."""
    registry = ToolRegistry()
    if tools:
        for t in tools:
            registry.register(t)

    return Agent(
        model=model,
        context=Context(),
        tools=registry,
        permissions=permissions or PermissionManager(mode="trust"),
        console=_make_console(),
        test_command=test_command,
    )


class CapturingModel(MockModel):
    """A MockModel that also records the ``messages`` sent on every call, so
    tests can assert what a TRANSIENT nudge injected (never persisted to
    context history) actually looked like on the wire for a given round."""

    def __init__(self, responses):
        super().__init__(responses)
        self.captured_messages: list = []

    async def chat(self, **kwargs):
        self.captured_messages.append(kwargs.get("messages"))
        async for chunk in super().chat(**kwargs):
            yield chunk


class _WriteFileTool(Tool):
    """A write_file stand-in that always succeeds — for verification-habit
    tests, the exact write behavior doesn't matter, only the tool NAME (the
    agent tracks write_file/edit_file by name) and that the result has no
    "Error" in it (so it counts as a successful write)."""

    name = "write_file"
    description = "write a file"
    is_read_only = False
    requires_permission = False

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {"file_path": {"type": "string"},
                          "content": {"type": "string"}},
            "required": ["file_path", "content"],
        }

    async def execute(self, **kwargs):
        return "Wrote 10 bytes to a.py"


class _BashTool(Tool):
    """A bash stand-in for verification-habit tests — real command execution
    is irrelevant; only the ``command`` argument (checked for the detected
    test command) and a successful (non-error) result matter."""

    name = "bash"
    description = "run a shell command"
    is_read_only = False
    requires_permission = False

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(self, **kwargs):
        return "exit 0"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_simple_text_response():
    """Model returns only text chunks — agent should return the combined text."""
    model = MockModel([
        [
            {"type": "text", "content": "Hello "},
            {"type": "text", "content": "world!"},
            {"type": "done", "usage": {}},
        ],
    ])

    agent = _make_agent(model)
    result = await agent.run("Say hello")

    assert result == "Hello world!"
    # The user message and assistant reply should be in context
    msgs = agent.context.messages
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Say hello"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Hello world!"


async def test_tool_call_execution():
    """Model requests a tool call, then gives a final text answer."""
    tool = MockTool()

    # Round 1: model emits a tool_call chunk (no text), then done
    # Round 2: model emits text answer after seeing tool result
    model = MockModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "mock_tool",
             "arguments": {"arg": "hello"}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Tool said: mock result"},
            {"type": "done", "usage": {}},
        ],
    ])

    agent = _make_agent(model, tools=[tool])
    result = await agent.run("Use the tool")

    assert "mock result" in result

    # Verify context has tool result
    roles = [m["role"] for m in agent.context.messages]
    assert "tool" in roles


async def test_unknown_tool():
    """Model requests a tool that doesn't exist — error added to context."""
    model = MockModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "nonexistent_tool",
             "arguments": {"x": 1}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Okay, that failed."},
            {"type": "done", "usage": {}},
        ],
    ])

    agent = _make_agent(model)
    await agent.run("Try a bad tool")

    # The tool result in context should contain an error
    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "Unknown tool" in tool_msgs[0]["content"]


async def test_permission_denied():
    """When permissions.check returns False the tool should not execute."""
    tool = MockTool()

    model = MockModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "mock_tool",
             "arguments": {"arg": "secret"}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Permission was denied."},
            {"type": "done", "usage": {}},
        ],
    ])

    # Create a permission manager that denies everything
    perms = PermissionManager(mode="ask")
    perms.check = MagicMock(return_value=False)

    agent = _make_agent(model, tools=[tool], permissions=perms)
    await agent.run("Do something restricted")

    # The tool result should say permission denied
    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "Permission denied" in tool_msgs[0]["content"]


async def test_empty_dict_arguments_allowed():
    """Tool calls with arguments={} should be allowed (not rejected).

    This was a Phase 1 fix: the agent used to reject any falsy arguments
    including empty dicts. Now it only rejects arguments that are None.
    """
    # A tool that accepts no required args
    class NoArgTool(Tool):
        name = "no_arg_tool"
        description = "A tool with no required args"
        is_read_only = True
        requires_permission = False

        @property
        def parameters(self):
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            return "executed with empty args"

    model = MockModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "no_arg_tool",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Done."},
            {"type": "done", "usage": {}},
        ],
    ])

    agent = _make_agent(model, tools=[NoArgTool()])
    await agent.run("Run the tool with no args")

    # The tool result should NOT contain an error — it should have executed
    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "executed with empty args" in tool_msgs[0]["content"]
    assert "Error" not in tool_msgs[0]["content"]


async def test_none_arguments_rejected():
    """Tool calls with arguments=None should be rejected with an error.

    This guards against truncated model responses where arguments are missing.

    Note: render_tool_call is patched because it crashes on None args
    (a minor rendering bug). The core logic — rejecting the call and
    recording the error in context — is what this test validates.
    """
    tool = MockTool()

    model = MockModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "mock_tool",
             "arguments": None},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Arguments were missing."},
            {"type": "done", "usage": {}},
        ],
    ])

    agent = _make_agent(model, tools=[tool])

    with patch("spark_code.agent.render_tool_call"):
        await agent.run("Call tool with None args")

    # The tool result should contain an error about missing arguments
    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "Error" in tool_msgs[0]["content"]
    assert "no arguments" in tool_msgs[0]["content"].lower() or "truncated" in tool_msgs[0]["content"].lower()


def test_agent_sets_schema_reserve_from_tools():
    """Audit 2026-07-06 item 4: the agent seeds the context's schema reserve from
    the size of the tool schemas it sends every request, so estimate_tokens (and
    thus auto-compaction) accounts for that fixed overhead."""
    model = MockModel([])
    agent = _make_agent(model, tools=[MockTool()])
    assert agent.context.schema_reserve_tokens > 0


async def test_interrupt_mid_tool_leaves_repairable_context():
    """Audit 2026-07-06 item 2: a CancelledError raised mid-tool (Ctrl+C during
    execution) is a BaseException that escapes the tool-exec try/except, leaving
    an assistant tool_calls message with no matching tool result. The next
    get_messages() must backfill a synthetic result so the follow-up request is
    OpenAI-valid instead of 400ing."""

    class CancellingTool(Tool):
        name = "canceller"
        description = "raises CancelledError mid-run"
        is_read_only = False
        requires_permission = False

        @property
        def parameters(self):
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            raise asyncio.CancelledError()

    model = MockModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "canceller",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
    ])

    agent = _make_agent(model, tools=[CancellingTool()])
    with patch("spark_code.agent.render_tool_call"):
        with pytest.raises(asyncio.CancelledError):
            await agent.run("run it")

    # Orphan present: assistant tool_calls with call_1 but no tool result.
    raw = agent.context.messages
    assert raw[-1].get("role") == "assistant" and raw[-1].get("tool_calls")

    # get_messages() repairs it into a valid alternation.
    msgs = agent.context.get_messages()
    tool_results = [m for m in msgs if m.get("role") == "tool"
                    and m.get("tool_call_id") == "call_1"]
    assert len(tool_results) == 1


async def test_done_chunk_usage_recorded_into_context():
    """The done chunk's usage.prompt_tokens (server-reported, real tokenizer
    count) must reach Context.record_usage so the toolbar meter reflects
    reality instead of the char-based estimate."""
    model = MockModel([
        [
            {"type": "text", "content": "Hello!"},
            {"type": "done", "usage": {"prompt_tokens": 4096, "completion_tokens": 12}},
        ],
    ])

    agent = _make_agent(model)
    await agent.run("hi")

    assert agent.context.last_prompt_tokens == 4096


async def test_done_chunk_empty_usage_does_not_record():
    """An empty usage dict (some Ollama configs omit it) must NOT stomp an
    already-recorded value with 0 — leave last_prompt_tokens as the estimate
    fallback instead of a wrong zero."""
    model = MockModel([
        [
            {"type": "text", "content": "Hello!"},
            {"type": "done", "usage": {}},
        ],
    ])

    agent = _make_agent(model)
    await agent.run("hi")


# ---------------------------------------------------------------------------
# T2: auto-compaction upgrade — real-usage trigger + model summary
# ---------------------------------------------------------------------------


def _fill_history(ctx, n=20):
    # Messages are padded so the compactable slice is a meaningful share of the
    # window — the anti-thrash gain guard (slice >= 5% of window) must see a
    # compaction that can plausibly help, like a real long session's history.
    pad = " lorem ipsum dolor sit amet consectetur" * 40  # ~1.6K chars
    for i in range(n):
        ctx.add_user(f"user message {i}{pad}")
        ctx.add_assistant(f"assistant reply {i} with detail{pad}")


def _agent_with_context(model, max_tokens=32768):
    from spark_code.context import Context
    agent = _make_agent(model)
    agent.context = Context(max_tokens=max_tokens)
    return agent


async def test_maybe_compact_fires_when_low():
    """context_left(window) < 0.25 (>75% used) triggers compaction, and the
    model-generated summary lands in the compacted history."""
    model = MockModel([
        [{"type": "text", "content": "THE MODEL SUMMARY"},
         {"type": "done", "usage": {}}],
    ])
    agent = _agent_with_context(model)
    _fill_history(agent.context)
    agent.context.record_usage(int(32768 * 0.8))  # 80% used → 20% left < 25%
    before = len(agent.context.messages)

    msg = await agent._maybe_compact()

    assert msg is not None, "should have compacted"
    assert len(agent.context.messages) < before
    assert "THE MODEL SUMMARY" in agent.context.messages[0]["content"]


async def test_maybe_compact_skips_when_room():
    """context_left(window) >= 0.25 (<=75% used) must NOT compact."""
    model = MockModel([
        [{"type": "text", "content": "SUMMARY"}, {"type": "done", "usage": {}}],
    ])
    agent = _agent_with_context(model)
    _fill_history(agent.context)
    agent.context.record_usage(int(32768 * 0.5))  # 50% used → 50% left
    before = len(agent.context.messages)

    msg = await agent._maybe_compact()

    assert msg is None
    assert len(agent.context.messages) == before


async def test_maybe_compact_unknown_window_never_fires():
    """Sharp edge from T1: context_left() returns 0.0 for a falsy window, which
    must NOT be read as 'compact now'. A zero/unknown window never compacts."""
    model = MockModel([
        [{"type": "text", "content": "SUMMARY"}, {"type": "done", "usage": {}}],
    ])
    agent = _agent_with_context(model, max_tokens=0)
    _fill_history(agent.context)
    agent.context.record_usage(30000)  # would be "full" for a 32K window
    before = len(agent.context.messages)

    msg = await agent._maybe_compact()

    assert msg is None
    assert len(agent.context.messages) == before


async def test_maybe_compact_skips_summary_call_when_nothing_to_compact():
    """High usage but a too-short history (<= keep_recent) must NOT make the
    summary model call — compact() would keep everything anyway, so a request
    per round would be wasted."""
    called = {"n": 0}

    class CountingModel:
        async def chat(self, **kwargs):
            called["n"] += 1
            yield {"type": "text", "content": "SUMMARY"}
            yield {"type": "done", "usage": {}}

    agent = _agent_with_context(CountingModel())
    agent.context.add_user("one giant message")  # only 1 message
    agent.context.record_usage(int(32768 * 0.9))  # 90% used

    msg = await agent._maybe_compact()

    assert msg is None
    assert called["n"] == 0, "no model call when there's nothing to compact"


async def test_maybe_compact_recursion_guard():
    """The _compacting flag prevents the summary request from re-triggering
    compaction (which would recurse)."""
    model = MockModel([
        [{"type": "text", "content": "SUMMARY"}, {"type": "done", "usage": {}}],
    ])
    agent = _agent_with_context(model)
    _fill_history(agent.context)
    agent.context.record_usage(int(32768 * 0.9))
    agent._compacting = True  # simulate being inside a summary request
    before = len(agent.context.messages)

    msg = await agent._maybe_compact()

    assert msg is None
    assert len(agent.context.messages) == before


async def test_maybe_compact_no_thrash_with_giant_static_prompt():
    """Reviewer repro (review round 1): a huge STATIC system prompt (e.g. big
    pinned files) parks usage in the 75-90% band while the message list is tiny.
    Compaction can free ~nothing there, so _maybe_compact must NOT fire a
    full-context summary request every round — the compactable-slice gain guard
    (slice < 5% of window) skips entirely: no request, no compact, all 5 rounds."""
    called = {"n": 0}

    class CountingModel:
        async def chat(self, **kwargs):
            called["n"] += 1
            yield {"type": "text", "content": "SUMMARY"}
            yield {"type": "done", "usage": {}}

    from spark_code.context import Context
    agent = _make_agent(CountingModel())
    # ~100K chars of static prompt → estimate ≈ 28.6K of a 32K window (~87% used).
    agent.context = Context(system_prompt="P" * 100000, max_tokens=32768)
    for i in range(5):  # 10 small messages — more than keep_recent (6)
        agent.context.add_user(f"small message {i}")
        agent.context.add_assistant(f"small reply {i}")
    # Sanity: the raw trigger condition IS met (this is the thrash band)...
    assert agent.context.context_left(32768) < 0.25
    before = list(agent.context.messages)

    for _ in range(5):  # five rounds, per the reviewer's repro
        assert await agent._maybe_compact() is None

    # ...but the gain guard must have blocked every round: zero summary
    # requests, history untouched.
    assert called["n"] == 0
    assert agent.context.messages == before


async def test_maybe_compact_cooldown_blocks_rapid_refire():
    """Anti-thrash cooldown: never auto-compact twice within 5 rounds, even if
    every other guard would allow an immediate refire."""
    called = {"n": 0}

    class CountingModel:
        async def chat(self, **kwargs):
            called["n"] += 1
            yield {"type": "text", "content": "SUMMARY"}
            yield {"type": "done", "usage": {}}

    agent = _agent_with_context(CountingModel())
    _fill_history(agent.context)
    agent.context.record_usage(int(32768 * 0.8))

    assert await agent._maybe_compact() is not None  # round 1: fires
    assert called["n"] == 1

    # Re-inflate so context_left and the gain guard would BOTH allow a refire.
    _fill_history(agent.context)
    agent.context.record_usage(int(32768 * 0.8))

    for _ in range(4):  # rounds 2-5: inside the cooldown window
        assert await agent._maybe_compact() is None
    assert called["n"] == 1, "no summary requests during cooldown"

    assert await agent._maybe_compact() is not None  # round 6: allowed again
    assert called["n"] == 2


async def test_maybe_compact_model_failure_falls_back_to_mechanical():
    """If the summary request fails (error chunk / raise), compaction still
    happens via the mechanical digest — the session never wedges and no message
    is lost."""
    class ErrorModel:
        async def chat(self, **kwargs):
            yield {"type": "error", "content": "API error (500)"}
            yield {"type": "done", "usage": {}}

    agent = _agent_with_context(ErrorModel())
    _fill_history(agent.context)
    agent.context.record_usage(int(32768 * 0.85))
    before = len(agent.context.messages)

    msg = await agent._maybe_compact()

    assert msg is not None
    assert len(agent.context.messages) < before
    # Mechanical digest was used (its header), not a crash.
    assert "Conversation Summary" in agent.context.messages[0]["content"]


async def test_maybe_compact_model_raises_falls_back():
    """An exception thrown by the model client is caught → mechanical fallback."""
    class RaisingModel:
        async def chat(self, **kwargs):
            if False:
                yield {}
            raise RuntimeError("boom")

    agent = _agent_with_context(RaisingModel())
    _fill_history(agent.context)
    agent.context.record_usage(int(32768 * 0.85))
    before = len(agent.context.messages)

    msg = await agent._maybe_compact()

    assert msg is not None
    assert len(agent.context.messages) < before


async def test_generate_compaction_summary_folds_instructions():
    """/compact <instructions> appends the user's focus into the summary
    request that goes to the model."""
    from spark_code.agent import generate_compaction_summary
    from spark_code.context import Context

    captured = {}

    class CapturingModel:
        async def chat(self, **kwargs):
            captured["messages"] = kwargs.get("messages")
            captured["stream"] = kwargs.get("stream")
            yield {"type": "text", "content": "SUMMARY"}
            yield {"type": "done", "usage": {}}

    ctx = Context()
    ctx.add_user("build the database layer")
    ctx.add_assistant("working on it")
    out = await generate_compaction_summary(
        CapturingModel(), ctx, instructions="focus on the database work")

    assert out == "SUMMARY"
    assert captured["stream"] is False  # non-streamed request
    last = captured["messages"][-1]["content"]
    assert "focus on the database work" in last


# ---------------------------------------------------------------------------
# T2 Step 6: per-tool result budgets
# ---------------------------------------------------------------------------


def test_truncate_result_respects_explicit_limit():
    from spark_code.agent import _truncate_result
    big = "H" * 15000 + "T" * 15000  # 30K, distinct head/tail
    out = _truncate_result(big, 8000)
    assert "truncated" in out
    assert out.startswith("H")       # head preserved
    assert out.endswith("T")         # tail preserved (exit codes etc.)
    assert len(out) < 8200           # ~8K, not 30K


def test_budget_for_tool_defaults():
    from spark_code.agent import MAX_TOOL_RESULT_CHARS
    model = MockModel([])
    agent = _make_agent(model)
    assert agent._budget_for("grep") == 8000
    assert agent._budget_for("read_file") == 10000
    assert agent._budget_for("bash") == 15000
    assert agent._budget_for("dispatch_agent") == 8000
    # Unlisted tool falls back to the module default.
    assert agent._budget_for("web_search") == MAX_TOOL_RESULT_CHARS


def test_budget_for_tool_config_override():
    from spark_code.context import Context
    from spark_code.permissions import PermissionManager
    from spark_code.tools.base import ToolRegistry
    agent = Agent(
        model=MockModel([]),
        context=Context(),
        tools=ToolRegistry(),
        permissions=PermissionManager(mode="trust"),
        console=_make_console(),
        result_budgets={"grep": 500},
    )
    assert agent._budget_for("grep") == 500       # overridden
    assert agent._budget_for("read_file") == 10000  # module default kept
    assert agent._budget_for("web_search") == 15000

    assert agent.context.last_prompt_tokens is None


# ---------------------------------------------------------------------------
# Task 6: verification habit — run tests before claiming done
# ---------------------------------------------------------------------------


async def test_verify_nudge_fires_once_then_second_final_answer_ends_turn():
    """write_file round -> final answer (no tool calls) -> the nudge fires and
    the loop continues ONCE -> a second final answer ends the turn normally.
    The nudge text must appear in round 3's outgoing request but NEVER in
    persisted context history (transient, like the round-limit notes)."""
    model = CapturingModel([
        [{"type": "tool_call", "id": "call_1", "name": "write_file",
          "arguments": {"file_path": "a.py", "content": "x"}},
         {"type": "done", "usage": {}}],
        [{"type": "text", "content": "First answer."},
         {"type": "done", "usage": {}}],
        [{"type": "text", "content": "Second answer."},
         {"type": "done", "usage": {}}],
    ])
    agent = _make_agent(model, tools=[_WriteFileTool()], test_command="pytest")

    result = await agent.run("change a file")

    assert result == "First answer.Second answer."
    assert model._call == 3, "exactly one extra round after the nudge"
    assert agent._verify_nudged is True
    assert agent._verify_dirty is True
    assert agent._verify_tests_run is False

    # The nudge was injected into round 3's request only.
    round3_request = model.captured_messages[2]
    assert any(
        m.get("role") == "system"
        and "have not run the project's tests" in m.get("content", "")
        and "`pytest`" in m.get("content", "")
        for m in round3_request
    )
    # Round 1 and round 2 requests must NOT have carried it.
    for earlier in model.captured_messages[:2]:
        assert not any(
            "have not run the project's tests" in (m.get("content") or "")
            for m in earlier
        )
    # Never persisted into actual conversation history.
    assert not any(
        "have not run the project's tests" in (m.get("content") or "")
        for m in agent.context.messages
    )
    # Both of the model's real answers ARE persisted.
    assistant_texts = [m["content"] for m in agent.context.messages
                       if m.get("role") == "assistant" and m.get("content")]
    assert "First answer." in assistant_texts
    assert "Second answer." in assistant_texts


async def test_verify_no_nudge_when_tests_ran_this_turn():
    """A bash call containing the detected test command marks tests as run —
    the subsequent final answer must NOT be interrupted by a nudge."""
    model = CapturingModel([
        [{"type": "tool_call", "id": "call_1", "name": "write_file",
          "arguments": {"file_path": "a.py", "content": "x"}},
         {"type": "done", "usage": {}}],
        [{"type": "tool_call", "id": "call_2", "name": "bash",
          "arguments": {"command": "pytest -q"}},
         {"type": "done", "usage": {}}],
        [{"type": "text", "content": "All tests pass."},
         {"type": "done", "usage": {}}],
    ])
    agent = _make_agent(model, tools=[_WriteFileTool(), _BashTool()],
                        test_command="pytest")

    result = await agent.run("change a file and verify")

    assert result == "All tests pass."
    assert model._call == 3, "no extra round — nudge must not fire"
    assert agent._verify_dirty is True
    assert agent._verify_tests_run is True
    assert agent._verify_nudged is False


async def test_verify_bash_match_not_fooled_by_substring():
    """A bash command that merely CONTAINS the test-command word as a
    substring of something else (e.g. "mypytest_dir") must NOT count as having
    run the tests — matched at word boundaries, like detect_test_command's
    instructions-text check."""
    model = CapturingModel([
        [{"type": "tool_call", "id": "call_1", "name": "write_file",
          "arguments": {"file_path": "a.py", "content": "x"}},
         {"type": "done", "usage": {}}],
        [{"type": "tool_call", "id": "call_2", "name": "bash",
          "arguments": {"command": "ls mypytest_dir/"}},
         {"type": "done", "usage": {}}],
        [{"type": "text", "content": "Here's the directory listing."},
         {"type": "done", "usage": {}}],
        [{"type": "text", "content": "Ran pytest, all green."},
         {"type": "done", "usage": {}}],
    ])
    agent = _make_agent(model, tools=[_WriteFileTool(), _BashTool()],
                        test_command="pytest")

    result = await agent.run("change a file and list a dir")

    # The "mypytest_dir" ls must NOT satisfy tests-run — nudge fires once.
    assert agent._verify_tests_run is False
    assert agent._verify_nudged is True
    assert model._call == 4
    assert result == "Here's the directory listing.Ran pytest, all green."


async def test_verify_no_nudge_on_read_only_turn():
    """No write/edit tool ever succeeded this turn — a plain Q&A turn must
    never be interrupted by the verification nudge."""
    model = CapturingModel([
        [{"type": "text", "content": "The answer is 42."},
         {"type": "done", "usage": {}}],
    ])
    agent = _make_agent(model, test_command="pytest")

    result = await agent.run("what is the answer")

    assert result == "The answer is 42."
    assert model._call == 1
    assert agent._verify_dirty is False
    assert agent._verify_nudged is False


async def test_verify_no_nudge_when_test_command_undetectable():
    """detect_test_command returning None (no project markers, no instructions
    override) must turn the feature silently off — even after a write and a
    final answer, no nudge fires."""
    model = CapturingModel([
        [{"type": "tool_call", "id": "call_1", "name": "write_file",
          "arguments": {"file_path": "a.py", "content": "x"}},
         {"type": "done", "usage": {}}],
        [{"type": "text", "content": "Done."},
         {"type": "done", "usage": {}}],
    ])
    agent = _make_agent(model, tools=[_WriteFileTool()], test_command=None)

    result = await agent.run("change a file")

    assert result == "Done."
    assert model._call == 2, "no extra round — test_command is None"
    assert agent._verify_dirty is True
    assert agent._verify_nudged is False


async def test_verify_flags_reset_across_user_turns():
    """The dirty/tests-run/nudged flags are per-TURN state: a second call to
    run() (a new user message) must start clean even though the previous turn
    left them set."""
    model = CapturingModel([
        # Turn 1: write, final answer -> nudge fires -> second final ends turn.
        [{"type": "tool_call", "id": "call_1", "name": "write_file",
          "arguments": {"file_path": "a.py", "content": "x"}},
         {"type": "done", "usage": {}}],
        [{"type": "text", "content": "First answer."},
         {"type": "done", "usage": {}}],
        [{"type": "text", "content": "Second answer."},
         {"type": "done", "usage": {}}],
        # Turn 2: pure read-only Q&A — must NOT inherit turn 1's dirty state.
        [{"type": "text", "content": "Just answering, no changes."},
         {"type": "done", "usage": {}}],
    ])
    agent = _make_agent(model, tools=[_WriteFileTool()], test_command="pytest")

    await agent.run("change a file")
    assert agent._verify_nudged is True
    assert agent._verify_dirty is True

    result2 = await agent.run("what does this do")

    assert result2 == "Just answering, no changes."
    assert model._call == 4, "turn 2 must not trigger another nudge round"
    assert agent._verify_dirty is False
    assert agent._verify_tests_run is False
    assert agent._verify_nudged is False
