"""Tests for true plan mode — enforced read-only + exit_plan_mode approval gate.

Plan mode used to be a loose ``config["_plan_mode"]`` flag that merely
prompt-wrapped the user's message. It is now real enforcement at
tool-execution time (``Agent._plan_denies`` — shared by BOTH the sequential
and parallel paths) plus an ``exit_plan_mode`` approval tool. These tests cover:

  * the enforcement matrix (writes blocked, reads allowed, exit_plan_mode
    allowed, dispatch_agent allowed for explore/reviewer but blocked for
    implementer — the Task-4 carry-forward);
  * a write is actually blocked end-to-end (file not created, denial text
    returned) while a read still executes;
  * the exit_plan_mode gate: approve flips state + drops to auto, reject keeps
    plan mode + returns feedback, and it is a no-op outside plan mode;
  * Shift+Tab into plan then approve leaves ``permissions.mode == "auto"``.
"""

import io
from unittest.mock import patch

from rich.console import Console

from spark_code.agent import Agent
from spark_code.context import Context
from spark_code.permissions import PermissionManager
from spark_code.plan_mode import (
    PLAN_DENIAL,
    ExitPlanModeTool,
    PlanState,
    cycle_mode,
)
from spark_code.tools.base import Tool, ToolRegistry
from spark_code.tools.read_file import ReadFileTool
from spark_code.tools.write_file import WriteFileTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel:
    """Yields predefined chunk sequences, one list per chat() call."""

    total_input_tokens = 0
    total_output_tokens = 0

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


class _FakeDispatch(Tool):
    """Stand-in with dispatch_agent's shape: name + conservative is_read_only."""

    name = "dispatch_agent"
    description = "fake"
    is_read_only = False  # matches DispatchAgentTool (conservative)

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "ok"


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True)


def _agent(model, tools=None, permissions=None, plan_state=None):
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    return Agent(
        model=model,
        context=Context(),
        tools=registry,
        permissions=permissions or PermissionManager(mode="auto"),
        console=_console(),
        plan_state=plan_state,
    )


# ---------------------------------------------------------------------------
# Enforcement matrix (Agent._plan_denies — shared by both tool paths)
# ---------------------------------------------------------------------------


def test_plan_denies_matrix_when_active():
    ps = PlanState(active=True)
    agent = _agent(MockModel([]), plan_state=ps)

    write = WriteFileTool()
    read = ReadFileTool()
    exit_tool = ExitPlanModeTool(ps)
    dispatch = _FakeDispatch()

    # write is blocked
    assert agent._plan_denies(
        {"name": "write_file", "arguments": {"file_path": "x", "content": "y"}},
        write) is True
    # read is allowed (read-only)
    assert agent._plan_denies(
        {"name": "read_file", "arguments": {"file_path": "x"}}, read) is False
    # exit_plan_mode is always allowed — it IS the gate
    assert agent._plan_denies(
        {"name": "exit_plan_mode", "arguments": {"plan": "do it"}},
        exit_tool) is False
    # dispatch_agent explore / reviewer allowed (read-only sub-registries)
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": {"agent_type": "explore"}},
        dispatch) is False
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": {"agent_type": "reviewer"}},
        dispatch) is False
    # dispatch_agent implementer blocked (can edit files)
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": {"agent_type": "implementer"}},
        dispatch) is True


def test_plan_denies_dispatch_with_non_dict_args_blocks_gracefully():
    """_parse_tool_arguments doesn't guarantee a dict — a bare-scalar payload
    (e.g. the json string '"explore"') must be BLOCKED (fail closed), not raise
    AttributeError on args.get()."""
    ps = PlanState(active=True)
    agent = _agent(MockModel([]), plan_state=ps)
    dispatch = _FakeDispatch()

    # String, list, int — none of them can prove a read-only agent_type.
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": "explore"}, dispatch) is True
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": ["explore"]}, dispatch) is True
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": 7}, dispatch) is True


async def test_dispatch_string_args_in_plan_mode_denied_end_to_end():
    """Full loop: a dispatch_agent tool_call whose arguments is the STRING
    "explore" gets the denial result — no exception, no orphaned tool_call."""
    model = MockModel([
        [
            {"type": "tool_call", "id": "c1", "name": "dispatch_agent",
             "arguments": "explore"},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Understood."},
            {"type": "done", "usage": {}},
        ],
    ])
    ps = PlanState(active=True)
    agent = _agent(model, tools=[_FakeDispatch()],
                   permissions=PermissionManager(mode="auto"), plan_state=ps)

    await agent.run("dispatch something")  # must not raise

    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert tool_msgs, "tool_call was orphaned — no tool result recorded"
    assert PLAN_DENIAL in tool_msgs[-1]["content"]


def test_plan_denies_nothing_when_inactive():
    ps = PlanState(active=False)
    agent = _agent(MockModel([]), plan_state=ps)
    assert agent._plan_denies(
        {"name": "write_file", "arguments": {"file_path": "x", "content": "y"}},
        WriteFileTool()) is False


def test_plan_denies_nothing_when_no_plan_state():
    agent = _agent(MockModel([]), plan_state=None)
    assert agent._plan_denies(
        {"name": "write_file", "arguments": {"file_path": "x", "content": "y"}},
        WriteFileTool()) is False


# ---------------------------------------------------------------------------
# End-to-end: a write is blocked, a read still runs
# ---------------------------------------------------------------------------


async def test_write_blocked_end_to_end(tmp_path):
    target = tmp_path / "should_not_exist.txt"
    model = MockModel([
        [
            {"type": "tool_call", "id": "c1", "name": "write_file",
             "arguments": {"file_path": str(target), "content": "nope"}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Blocked, presenting a plan."},
            {"type": "done", "usage": {}},
        ],
    ])
    ps = PlanState(active=True)
    agent = _agent(model, tools=[WriteFileTool(), ReadFileTool()],
                   permissions=PermissionManager(mode="auto"), plan_state=ps)

    await agent.run("write the file")

    # The write never happened.
    assert not target.exists()
    # The tool result carries the denial text.
    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert tool_msgs
    assert PLAN_DENIAL in tool_msgs[-1]["content"]


async def test_read_allowed_in_plan_mode(tmp_path):
    src = tmp_path / "hello.txt"
    src.write_text("GREETING = 'hi there'\n")
    model = MockModel([
        [
            {"type": "tool_call", "id": "c1", "name": "read_file",
             "arguments": {"file_path": str(src)}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Read it."},
            {"type": "done", "usage": {}},
        ],
    ])
    ps = PlanState(active=True)
    agent = _agent(model, tools=[WriteFileTool(), ReadFileTool()],
                   permissions=PermissionManager(mode="auto"), plan_state=ps)

    await agent.run("read the file")

    tool_msgs = [m for m in agent.context.messages if m["role"] == "tool"]
    assert tool_msgs
    assert "GREETING" in tool_msgs[-1]["content"]
    assert PLAN_DENIAL not in tool_msgs[-1]["content"]


# ---------------------------------------------------------------------------
# exit_plan_mode approval gate
# ---------------------------------------------------------------------------


async def test_exit_plan_mode_approval_flips_state_and_mode():
    ps = PlanState(active=True)
    perms = PermissionManager(mode="auto")
    tool = ExitPlanModeTool(ps, permissions=perms, console=_console())

    with patch("spark_code.plan_mode.Prompt.ask", return_value="y"):
        result = await tool.execute(plan="1. do a thing\n2. do another")

    assert result == "Plan approved. Proceed."
    assert ps.active is False
    assert perms.mode == "auto"


async def test_exit_plan_mode_rejection_keeps_mode_and_returns_feedback():
    ps = PlanState(active=True)
    perms = PermissionManager(mode="auto")
    tool = ExitPlanModeTool(ps, permissions=perms, console=_console())

    # First ask → reject ("n"); second ask → the feedback text.
    with patch("spark_code.plan_mode.Prompt.ask",
               side_effect=["n", "add tests first"]):
        result = await tool.execute(plan="1. ship it")

    assert result.startswith("Plan rejected: add tests first")
    assert "Revise and present again" in result
    assert ps.active is True  # still in plan mode


async def test_exit_plan_mode_outside_plan_mode_is_noop():
    ps = PlanState(active=False)
    perms = PermissionManager(mode="ask")
    tool = ExitPlanModeTool(ps, permissions=perms, console=_console())

    with patch("spark_code.plan_mode.Prompt.ask") as ask:
        result = await tool.execute(plan="whatever")

    assert result == "Not in plan mode."
    ask.assert_not_called()
    assert perms.mode == "ask"  # untouched


# ---------------------------------------------------------------------------
# Shift+Tab into plan, then approve → mode stays auto
# ---------------------------------------------------------------------------


async def test_shift_tab_into_plan_then_approve_leaves_mode_auto():
    ps = PlanState(active=False)
    perms = PermissionManager(mode="auto")

    # Shift+Tab from auto → plan (the mode cycle). Entering plan drops to auto.
    next_mode = cycle_mode("auto", ps, perms)
    assert next_mode == "plan"
    assert ps.active is True
    assert perms.mode == "auto"

    # Present a plan and approve it.
    tool = ExitPlanModeTool(ps, permissions=perms, console=_console())
    with patch("spark_code.plan_mode.Prompt.ask", return_value="y"):
        result = await tool.execute(plan="1. implement")

    assert result == "Plan approved. Proceed."
    assert ps.active is False
    assert perms.mode == "auto"


def test_cycle_mode_full_cycle():
    """ask → auto → plan → ask, and plan entry drops perms to auto."""
    ps = PlanState(active=False)
    perms = PermissionManager(mode="ask")

    assert cycle_mode("ask", ps, perms) == "auto"
    assert ps.active is False and perms.mode == "auto"

    assert cycle_mode("auto", ps, perms) == "plan"
    assert ps.active is True and perms.mode == "auto"

    assert cycle_mode("plan", ps, perms) == "ask"
    assert ps.active is False and perms.mode == "ask"
