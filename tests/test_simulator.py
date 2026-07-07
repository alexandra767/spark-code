"""Tests for the iOS vision loop (Phase 4 Task 4):

  * SimulatorScreenshotTool itself — subprocess argv, the booted-check-first
    guard, timeouts, non-vision graceful degradation, and the sentinel it
    returns for a vision-capable model (spark_code/tools/simulator.py).
  * build_tools()'s Darwin+xcrun registration guard (spark_code/cli.py).
  * Agent._store_tool_result — the image-injection mechanism that turns the
    tool's sentinel into an add_user_with_image-shaped multimodal turn
    (spark_code/agent.py), exercised through both the sequential AND
    parallel tool-execution paths.

Every subprocess call is patched — none of these tests require an actual
booted simulator or even macOS.
"""

import base64
import os
import platform
import shutil
import subprocess

from spark_code.tools.base import Tool, ToolRegistry
from spark_code.tools.simulator import (
    SCREENSHOT_SENTINEL_PREFIX,
    SimulatorScreenshotTool,
)

FAKE_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-simulator-screenshot-bytes"

BOOTED_LISTING = (
    "== Devices ==\n"
    "-- iOS 26.2 --\n"
    "    iPhone 17 Pro (ABCDEF-1234-5678) (Booted)\n"
)
NO_BOOTED_LISTING = (
    "== Devices ==\n"
    "-- iOS 26.2 --\n"
    "    iPhone 17 Pro (ABCDEF-1234-5678) (Shutdown)\n"
)


class FakeModel:
    def __init__(self, supports_vision: bool):
        self.supports_vision = supports_vision


def _happy_run(write_png: bool = True):
    """A subprocess.run replacement answering BOTH calls the tool makes:
    the booted-check (list devices) and the screenshot capture itself —
    mirrors `xcrun simctl io booted screenshot <path>` by writing
    FAKE_PNG_BYTES to the requested path, exactly like the real command
    would write a real PNG there."""
    def _run(argv, capture_output=True, text=True, timeout=None):
        if argv[:4] == ["xcrun", "simctl", "list", "devices"]:
            return subprocess.CompletedProcess(argv, 0, stdout=BOOTED_LISTING, stderr="")
        if argv[:4] == ["xcrun", "simctl", "io", "booted"]:
            path = argv[-1]
            if write_png:
                with open(path, "wb") as f:
                    f.write(FAKE_PNG_BYTES)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected simctl invocation: {argv}")
    return _run


# ---------------------------------------------------------------------------
# SimulatorScreenshotTool
# ---------------------------------------------------------------------------


async def test_booted_check_runs_before_screenshot_with_right_argv(monkeypatch):
    calls = []

    def _run(argv, **kw):
        calls.append(argv)
        return _happy_run()(argv, **kw)

    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _run)
    tool = SimulatorScreenshotTool(model=FakeModel(True))
    result = await tool.execute()

    assert calls[0] == ["xcrun", "simctl", "list", "devices"]
    assert calls[1][:5] == ["xcrun", "simctl", "io", "booted", "screenshot"]
    assert result.startswith(SCREENSHOT_SENTINEL_PREFIX)
    # Cleanup the real tmp file the tool wrote.
    os.remove(result[len(SCREENSHOT_SENTINEL_PREFIX):])


async def test_no_booted_simulator_returns_clear_message_not_exception(monkeypatch):
    calls = []

    def _run(argv, **kw):
        calls.append(argv)
        assert argv == ["xcrun", "simctl", "list", "devices"]
        return subprocess.CompletedProcess(argv, 0, stdout=NO_BOOTED_LISTING, stderr="")

    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _run)
    tool = SimulatorScreenshotTool(model=FakeModel(True))
    result = await tool.execute()

    assert "No booted" in result
    assert "Traceback" not in result
    # Screenshot must never be attempted once the booted-check comes up empty.
    assert len(calls) == 1


async def test_booted_check_subprocess_error_is_not_an_exception(monkeypatch):
    def _run(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="simctl: daemon unreachable")

    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _run)
    tool = SimulatorScreenshotTool(model=FakeModel(True))
    result = await tool.execute()

    assert "Error" in result
    assert "daemon unreachable" in result


async def test_booted_check_timeout(monkeypatch):
    def _run(argv, **kw):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kw.get("timeout"))

    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _run)
    tool = SimulatorScreenshotTool(model=FakeModel(True), timeout=1.0)
    result = await tool.execute()

    assert "timed out" in result.lower()


async def test_screenshot_step_timeout(monkeypatch):
    def _run(argv, **kw):
        if argv[:4] == ["xcrun", "simctl", "list", "devices"]:
            return subprocess.CompletedProcess(argv, 0, stdout=BOOTED_LISTING, stderr="")
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kw.get("timeout"))

    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _run)
    tool = SimulatorScreenshotTool(model=FakeModel(True), timeout=1.0)
    result = await tool.execute()

    assert "timed out" in result.lower()


async def test_screenshot_command_failure_returns_message_not_exception(monkeypatch):
    def _run(argv, **kw):
        if argv[:4] == ["xcrun", "simctl", "list", "devices"]:
            return subprocess.CompletedProcess(argv, 0, stdout=BOOTED_LISTING, stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="simctl: could not connect to device")

    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _run)
    tool = SimulatorScreenshotTool(model=FakeModel(True))
    result = await tool.execute()

    assert "Error" in result
    assert "could not connect to device" in result


async def test_screenshot_reports_success_but_writes_no_file(monkeypatch):
    monkeypatch.setattr(
        "spark_code.tools.simulator.subprocess.run", _happy_run(write_png=False))
    tool = SimulatorScreenshotTool(model=FakeModel(True))
    result = await tool.execute()

    assert "Error" in result
    assert "no image" in result.lower()


async def test_non_vision_model_saves_file_and_returns_text_message(monkeypatch):
    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _happy_run())
    tool = SimulatorScreenshotTool(model=FakeModel(False))
    result = await tool.execute()

    assert result.startswith("current model has no vision; screenshot saved to ")
    saved_path = result.split("screenshot saved to ", 1)[1]
    assert os.path.exists(saved_path)
    with open(saved_path, "rb") as f:
        assert f.read() == FAKE_PNG_BYTES
    os.remove(saved_path)


async def test_no_model_at_all_treated_as_non_vision(monkeypatch):
    """Sub-agent tool registries pass model=None — must degrade gracefully,
    never raise on the getattr(self.model, "supports_vision", False) check."""
    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _happy_run())
    tool = SimulatorScreenshotTool(model=None)
    result = await tool.execute()

    assert result.startswith("current model has no vision; screenshot saved to ")
    os.remove(result.split("screenshot saved to ", 1)[1])


async def test_vision_model_sentinel_and_png_readback(monkeypatch):
    """PNG readback: the file at the sentinel's path is exactly the bytes
    the (mocked) screenshot command wrote — proves the tool's contract with
    the agent's image-injection step (which reads this same path)."""
    monkeypatch.setattr("spark_code.tools.simulator.subprocess.run", _happy_run())
    tool = SimulatorScreenshotTool(model=FakeModel(True))
    result = await tool.execute()

    assert result.startswith(SCREENSHOT_SENTINEL_PREFIX)
    path = result[len(SCREENSHOT_SENTINEL_PREFIX):]
    with open(path, "rb") as f:
        data = f.read()
    assert data == FAKE_PNG_BYTES
    os.remove(path)


def test_schema_takes_no_arguments():
    tool = SimulatorScreenshotTool()
    assert tool.parameters == {"type": "object", "properties": {}}
    assert tool.name == "simulator_screenshot"
    assert tool.is_read_only is True


# ---------------------------------------------------------------------------
# build_tools() registration guard (Darwin + xcrun only)
# ---------------------------------------------------------------------------


class TestRegistrationGuard:
    def test_registered_on_darwin_with_xcrun_present(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            shutil, "which", lambda name: "/usr/bin/xcrun" if name == "xcrun" else None)
        from spark_code.cli import build_tools
        registry = build_tools()
        assert "simulator_screenshot" in registry.names()

    def test_absent_on_non_darwin_no_error(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            shutil, "which", lambda name: "/usr/bin/xcrun" if name == "xcrun" else None)
        from spark_code.cli import build_tools
        # Must not raise even though this "would" have xcrun on PATH — the
        # platform check alone must gate it out, with no import error either.
        registry = build_tools()
        assert "simulator_screenshot" not in registry.names()

    def test_absent_on_darwin_without_xcrun(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(shutil, "which", lambda name: None)
        from spark_code.cli import build_tools
        registry = build_tools()
        assert "simulator_screenshot" not in registry.names()


# ---------------------------------------------------------------------------
# Agent-level image injection (the "subtle part") — mirrors context.py:207's
# add_user_with_image shape exactly, through BOTH the sequential and
# parallel tool-execution paths.
# ---------------------------------------------------------------------------


class _FakeMockChatModel:
    """Drives Agent.run() with a scripted sequence of chunk lists — same
    shape as test_agent.py's MockModel, kept local so this file has no
    cross-module test dependency."""

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


class _ScriptedScreenshotTool(Tool):
    """A fake simulator_screenshot tool that returns a pre-canned result —
    lets the agent-level tests drive Agent._store_tool_result without going
    through the real subprocess-calling tool."""

    name = "simulator_screenshot"
    description = "fake"
    is_read_only = True
    requires_permission = False

    def __init__(self, canned_result: str):
        self._result = canned_result

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return self._result


class _OtherReadOnlyTool(Tool):
    """A second auto-allowed read-only tool, used to force two tool_calls
    into one round so the PARALLEL execution path (agent.py's
    _execute_parallel) is exercised too, not just the sequential one."""

    name = "peek_tool"
    description = "fake"
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return "peeked"


class _FakeReadFileTool(Tool):
    """A read-only stand-in for read_file so a mixed round can pair a genuine
    non-screenshot read tool with simulator_screenshot."""

    name = "read_file"
    description = "fake"
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self):
        return {"type": "object",
                "properties": {"file_path": {"type": "string"}}}

    async def execute(self, **kwargs) -> str:
        return "file contents here"


def _assert_valid_tool_transcript(messages):
    """Assert every assistant tool_calls message is followed IMMEDIATELY by a
    contiguous run of role:"tool" results answering exactly its ids (each
    once, in order, with no other role — e.g. an injected user image —
    interleaved), and that no stray/orphaned role:"tool" message exists
    outside such a run. Mirrors what a strict OpenAI-compatible server
    enforces; the pre-fix bug (image spliced into the middle of a multi-tool
    round) violates it."""
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            expected = [tc["id"] for tc in m["tool_calls"]]
            j = i + 1
            got = []
            while j < n and messages[j]["role"] == "tool":
                got.append(messages[j]["tool_call_id"])
                j += 1
            assert got == expected, (
                f"tool-result run {got} does not match tool_calls {expected} "
                f"at index {i} — run was broken (image spliced in?)")
            i = j
        else:
            assert m["role"] != "tool", (
                f"stray/orphaned tool message at index {i}: {m}")
            i += 1


def _make_agent(model, tools, tmp_dir_console=None):
    from rich.console import Console

    from spark_code.agent import Agent
    from spark_code.context import Context
    from spark_code.permissions import PermissionManager

    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    import io
    return Agent(
        model=model,
        context=Context(),
        tools=registry,
        permissions=PermissionManager(mode="trust"),
        console=Console(file=io.StringIO(), force_terminal=True),
    )


def _make_png_file(tmp_dir) -> str:
    path = os.path.join(tmp_dir, "screenshot.png")
    with open(path, "wb") as f:
        f.write(FAKE_PNG_BYTES)
    return path


async def test_sequential_path_injects_image_content_block(tmp_dir):
    png_path = _make_png_file(tmp_dir)
    canned = f"{SCREENSHOT_SENTINEL_PREFIX}{png_path}"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "simulator_screenshot",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "I can see the app."},
            {"type": "done", "usage": {}},
        ],
    ])
    agent = _make_agent(model, [_ScriptedScreenshotTool(canned)])
    await agent.run("Screenshot the simulator and describe it")

    messages = agent.context.messages
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    # The tool-role reply must be short plain text — never the sentinel or
    # the raw base64 payload.
    assert tool_msgs[0]["content"].startswith("Screenshot captured")
    assert SCREENSHOT_SENTINEL_PREFIX not in tool_msgs[0]["content"]

    image_msgs = [
        m for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(image_msgs) == 1
    parts = image_msgs[0]["content"]
    assert parts[0]["type"] == "text"
    image_part = parts[1]
    assert image_part["type"] == "image_url"
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    b64_payload = url.split("base64,", 1)[1]
    assert base64.b64decode(b64_payload) == FAKE_PNG_BYTES


async def test_non_vision_result_never_injects_an_image(tmp_dir):
    """When the tool itself decided the model has no vision, its result is
    plain text (no sentinel) — the agent must store it as-is and must NOT
    synthesize an image turn."""
    canned = "current model has no vision; screenshot saved to /tmp/whatever.png"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "simulator_screenshot",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Noted, no vision available."},
            {"type": "done", "usage": {}},
        ],
    ])
    agent = _make_agent(model, [_ScriptedScreenshotTool(canned)])
    await agent.run("Screenshot the simulator")

    messages = agent.context.messages
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == canned

    image_msgs = [
        m for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert image_msgs == []


async def test_missing_file_at_sentinel_path_degrades_without_crashing(tmp_dir):
    missing_path = os.path.join(tmp_dir, "does-not-exist.png")
    canned = f"{SCREENSHOT_SENTINEL_PREFIX}{missing_path}"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "simulator_screenshot",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Something went wrong."},
            {"type": "done", "usage": {}},
        ],
    ])
    agent = _make_agent(model, [_ScriptedScreenshotTool(canned)])
    await agent.run("Screenshot the simulator")

    messages = agent.context.messages
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "could not be read back" in tool_msgs[0]["content"]
    image_msgs = [
        m for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert image_msgs == []


async def test_parallel_path_also_injects_image_content_block(tmp_dir):
    """Screenshot called ALONGSIDE another tool in one round (parallel path,
    screenshot NOT last) — the image must be injected AFTER the round's
    complete tool-result run, never spliced into the middle of it. The
    pre-fix bug corrupted the transcript here: sanitize_orphaned_tool_calls
    saw a broken run, backfilled a synthetic '[interrupted]', and stranded
    the real second result in an invalid position."""
    png_path = _make_png_file(tmp_dir)
    canned = f"{SCREENSHOT_SENTINEL_PREFIX}{png_path}"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "simulator_screenshot",
             "arguments": {}},
            {"type": "tool_call", "id": "call_2", "name": "peek_tool",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Done."},
            {"type": "done", "usage": {}},
        ],
    ])
    agent = _make_agent(
        model, [_ScriptedScreenshotTool(canned), _OtherReadOnlyTool()])
    await agent.run("Screenshot and peek")

    messages = agent.context.messages
    # Full-sequence validity: the tool-result run must be intact (no image
    # spliced in) and there must be no stray/synthetic tool message.
    _assert_valid_tool_transcript(messages)
    # And get_messages()'s sanitize pass must have nothing to repair.
    assert agent.context.sanitize_orphaned_tool_calls() == 0

    # Exactly ONE result per tool call, each answered once (no duplicates).
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert set(by_id) == {"call_1", "call_2"}
    assert by_id["call_2"] == "peeked"
    assert by_id["call_1"].startswith("Screenshot captured")

    # Exactly one image message, and it lands AFTER the last tool result.
    image_indices = [
        idx for idx, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(image_indices) == 1
    last_tool_idx = max(
        idx for idx, m in enumerate(messages) if m["role"] == "tool")
    assert image_indices[0] > last_tool_idx
    url = messages[image_indices[0]]["content"][1]["image_url"]["url"]
    assert base64.b64decode(url.split("base64,", 1)[1]) == FAKE_PNG_BYTES


async def test_mixed_round_read_file_and_screenshot_keeps_transcript_valid(tmp_dir):
    """The exact mixed-round case: model calls [read_file, simulator_screenshot,
    peek_tool] in ONE round, with the screenshot in the MIDDLE (so a
    naive immediate-inject would splice the image between two real tool
    results). Result must be a valid transcript with the image present."""
    png_path = _make_png_file(tmp_dir)
    canned = f"{SCREENSHOT_SENTINEL_PREFIX}{png_path}"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "read_file",
             "arguments": {"file_path": "App.swift"}},
            {"type": "tool_call", "id": "call_2", "name": "simulator_screenshot",
             "arguments": {}},
            {"type": "tool_call", "id": "call_3", "name": "peek_tool",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Compared code to the running UI."},
            {"type": "done", "usage": {}},
        ],
    ])
    agent = _make_agent(
        model,
        [_FakeReadFileTool(), _ScriptedScreenshotTool(canned), _OtherReadOnlyTool()])
    await agent.run("Read the file and screenshot the simulator")

    messages = agent.context.messages
    _assert_valid_tool_transcript(messages)
    assert agent.context.sanitize_orphaned_tool_calls() == 0

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert set(by_id) == {"call_1", "call_2", "call_3"}
    assert by_id["call_2"].startswith("Screenshot captured")

    # Image present, decodes to the real PNG, and sits after the last tool result.
    image_indices = [
        idx for idx, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(image_indices) == 1
    last_tool_idx = max(
        idx for idx, m in enumerate(messages) if m["role"] == "tool")
    assert image_indices[0] > last_tool_idx
    url = messages[image_indices[0]]["content"][1]["image_url"]["url"]
    assert base64.b64decode(url.split("base64,", 1)[1]) == FAKE_PNG_BYTES
