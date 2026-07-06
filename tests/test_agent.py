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


def _make_agent(model, tools=None, permissions=None):
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
    )


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
    for i in range(n):
        ctx.add_user(f"user message {i}")
        ctx.add_assistant(f"assistant reply {i} with detail")


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
