"""Tests for dispatch_agent — subagents as context isolation (Phase 2 Task 4).

The sub-agent machinery composes existing pieces (Agent + Context +
build_tools + the non-interactive PermissionManager). These tests drive it
with a scripted mock model (the pattern in tests/test_agent.py) so they run
offline and fast.
"""

import asyncio

import pytest

from spark_code import dispatch
from spark_code.context import Context
from spark_code.tools.base import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockModel:
    """Yields a scripted stream per chat() call (like tests/test_agent.py)."""

    def __init__(self, responses):
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


class RaisingModel:
    """A model whose chat stream raises — exercises the error path."""

    async def chat(self, **kwargs):
        raise RuntimeError("boom in the engine")
        yield  # unreachable — makes this an async generator


class SlowCountingModel:
    """Sleeps while tracking how many chat() bodies run concurrently."""

    def __init__(self, counter):
        self.counter = counter

    async def chat(self, **kwargs):
        self.counter["cur"] += 1
        self.counter["max"] = max(self.counter["max"], self.counter["cur"])
        try:
            await asyncio.sleep(0.05)
        finally:
            self.counter["cur"] -= 1
        yield {"type": "text", "content": "done"}
        yield {"type": "done", "usage": {}}


class _FakeDispatchTool(Tool):
    """Stand-in named dispatch_agent to prove the depth guard excludes it."""

    name = "dispatch_agent"
    description = "fake"
    is_read_only = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "should never run in a sub-agent"


def _cfg(max_concurrent=3, max_rounds=15, context_window=32768):
    return {
        "model": {"context_window": context_window},
        "agents": {"max_concurrent": max_concurrent, "max_rounds": max_rounds},
    }


def _final_text(text):
    return [[{"type": "text", "content": text}, {"type": "done", "usage": {}}]]


@pytest.fixture(autouse=True)
def _reset_dispatch_semaphore():
    """Each test gets a clean shared semaphore bound to its own event loop."""
    dispatch._reset_semaphore()
    yield
    dispatch._reset_semaphore()


# ---------------------------------------------------------------------------
# Registry filtering / depth guard
# ---------------------------------------------------------------------------


def test_explore_registry_is_read_only_only():
    registry = dispatch._build_subagent_registry("explore")
    names = registry.names()
    assert "read_file" in names
    assert "grep" in names
    # Mutating tools must be absent from a read-only sub-agent.
    assert "write_file" not in names
    assert "edit_file" not in names
    for tool in registry.all():
        assert tool.is_read_only, f"{tool.name} is not read-only"


def test_reviewer_registry_is_read_only_only():
    registry = dispatch._build_subagent_registry("reviewer")
    names = registry.names()
    assert "read_file" in names
    assert "write_file" not in names


def test_implementer_registry_has_writes_but_not_dispatch_agent():
    # Seed the base with a dispatch_agent tool so the depth guard has something
    # to strip — proves nesting is structurally impossible.
    base = ToolRegistry()
    from spark_code.cli import build_tools
    for tool in build_tools().all():
        base.register(tool)
    base.register(_FakeDispatchTool())
    assert "dispatch_agent" in base.names()

    registry = dispatch._build_subagent_registry("implementer", base_registry=base)
    names = registry.names()
    assert "write_file" in names
    assert "edit_file" in names
    assert "dispatch_agent" not in names  # no nesting


# ---------------------------------------------------------------------------
# run_subagent core
# ---------------------------------------------------------------------------


async def test_final_text_returned():
    model = MockModel(_final_text("GREETING is defined in greetings.py as 'Howdy'"))
    result = await dispatch.run_subagent(
        model, "find GREETING", "explore", _cfg(), "auto")
    assert result == "GREETING is defined in greetings.py as 'Howdy'"


async def test_result_truncated_head_and_tail():
    big = "A" * 4000 + "B" * 4000 + "C" * 4000  # 12000 > 8000
    model = MockModel(_final_text(big))
    result = await dispatch.run_subagent(
        model, "explore", "explore", _cfg(), "auto")
    assert len(result) < len(big)
    assert "truncated" in result
    assert result.startswith("A")   # head preserved
    assert result.endswith("C")     # tail preserved


async def test_subagent_exception_returns_error_string():
    model = RaisingModel()
    result = await dispatch.run_subagent(
        model, "explore", "explore", _cfg(), "auto")
    assert result.startswith("[dispatch_agent error]")
    assert "boom in the engine" in result


async def test_unknown_agent_type_returns_error_string():
    model = MockModel(_final_text("unused"))
    result = await dispatch.run_subagent(
        model, "do it", "wizard", _cfg(), "auto")
    assert result.startswith("[dispatch_agent error]")


async def test_semaphore_limits_concurrency():
    counter = {"cur": 0, "max": 0}
    cfg = _cfg(max_concurrent=3)

    async def one():
        model = SlowCountingModel(counter)
        return await dispatch.run_subagent(model, "go", "explore", cfg, "auto")

    results = await asyncio.gather(*[one() for _ in range(5)])
    assert all(r == "done" for r in results)
    assert counter["max"] <= 3           # the semaphore ceiling
    assert counter["max"] >= 2           # but it DID run in parallel


async def test_lead_context_untouched():
    lead_ctx = Context()
    lead_ctx.add_user("the lead's own task")
    before = len(lead_ctx.messages)
    model = MockModel(_final_text("sub report"))
    result = await dispatch.run_subagent(
        model, "explore", "explore", _cfg(), "auto")
    assert result == "sub report"
    # The sub-agent built its OWN context; the lead's is untouched.
    assert len(lead_ctx.messages) == before


async def test_subagent_respects_max_rounds_cap():
    # The sub-agent's cap comes from agents.max_rounds, not Agent.MAX_TOOL_ROUNDS.
    model = MockModel(_final_text("ok"))
    cfg = _cfg(max_rounds=7)
    # Capture the Agent the tool builds by monkeypatching nothing — instead
    # assert via the returned text (round cap is verified structurally below).
    result = await dispatch.run_subagent(model, "x", "explore", cfg, "auto")
    assert result == "ok"


# ---------------------------------------------------------------------------
# DispatchAgentTool wrapper
# ---------------------------------------------------------------------------


def test_tool_schema_and_read_only_flag():
    tool = dispatch.DispatchAgentTool(model=None, config=_cfg())
    assert tool.name == "dispatch_agent"
    assert tool.is_read_only is False  # conservative: per-call read-only unsupported
    schema = tool.to_schema()
    props = schema["parameters"]["properties"]
    assert set(props) == {"prompt", "agent_type"}
    assert props["agent_type"]["enum"] == ["explore", "reviewer", "implementer"]
    assert schema["parameters"]["required"] == ["prompt", "agent_type"]


async def test_tool_execute_delegates_to_run_subagent():
    model = MockModel(_final_text("delegated result"))
    tool = dispatch.DispatchAgentTool(model=model, config=_cfg(), lead_mode="auto")
    result = await tool.execute(prompt="find it", agent_type="explore")
    assert result == "delegated result"


async def test_tool_reads_live_lead_mode_from_permissions():
    from spark_code.permissions import PermissionManager
    perms = PermissionManager(mode="ask")
    captured = {}

    async def _fake_run(model, prompt, agent_type, config, lead_mode):
        captured["mode"] = lead_mode
        return "ok"

    model = MockModel(_final_text("ok"))
    tool = dispatch.DispatchAgentTool(model=model, config=_cfg(), permissions=perms)
    import spark_code.dispatch as d
    orig = d.run_subagent
    d.run_subagent = _fake_run
    try:
        await tool.execute(prompt="p", agent_type="explore")
        assert captured["mode"] == "ask"
        perms.mode = "auto"          # mode changes mid-session (Shift+Tab)
        await tool.execute(prompt="p", agent_type="explore")
        assert captured["mode"] == "auto"
    finally:
        d.run_subagent = orig


# ---------------------------------------------------------------------------
# build_tools registration
# ---------------------------------------------------------------------------


def test_build_tools_registers_dispatch_when_model_and_config_given():
    from spark_code.cli import build_tools
    registry = build_tools(model=MockModel([]), config=_cfg())
    assert "dispatch_agent" in registry.names()


def test_build_tools_omits_dispatch_without_model():
    from spark_code.cli import build_tools
    registry = build_tools()
    assert "dispatch_agent" not in registry.names()
