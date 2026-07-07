"""Regression tests for the MCP transport + client.

These prove the audit fixes:
  * stdio: id-matching, interleaved notifications don't corrupt responses, large
    results survive, a chatty stderr never deadlocks, a timed-out request never
    poisons the stream, env expansion reaches the spawned subprocess, and the
    initialize -> notifications/initialized -> tools/list lifecycle runs.
  * http: minimal Streamable-HTTP client negotiates Mcp-Session-Id and matches
    responses by id.

Also (Phase 5 Task 5) MCP image-content routing:
  * MCPTool.execute() — an ``image`` content item is decoded to a temp file
    and returned as a MCP_IMAGE_SENTINEL_PREFIX sentinel line (not the old
    inert "[Image: <mime>]" text), including multiple images in one result
    and a missing/malformed ``data`` field degrading gracefully.
  * Agent._store_tool_result / _resolve_mcp_images — the sentinel reaches a
    vision model as a buffered add_user_with_image turn (flushed AFTER the
    round's tool-result run, exactly like Phase 4 Task 4's
    simulator_screenshot mechanism — including the mixed-round transcript
    validity invariant), while a non-vision model gets the original
    placeholder text back with no image block.
"""

import asyncio
import base64
import os
import sys

import pytest

from spark_code.mcp.client import MCP_IMAGE_SENTINEL_PREFIX, MCPClient, MCPTool
from spark_code.mcp.registry import expand_mcp_env
from spark_code.mcp.transport import HTTPTransport, StdioTransport
from spark_code.tools.base import Tool, ToolRegistry

FAKE_SERVER = os.path.join(os.path.dirname(__file__), "_mcp_fake_server.py")

INIT_PARAMS = {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "spark-test", "version": "0.0.0"},
}


async def _started_transport(env=None):
    t = StdioTransport(sys.executable, [FAKE_SERVER], env=env or {})
    await t.start()
    return t


async def _handshake(t):
    await t.send("initialize", INIT_PARAMS)
    await t.notify("notifications/initialized")
    await t.send("tools/list")


# --------------------------------------------------------------------------- #
# stdio transport
# --------------------------------------------------------------------------- #
async def test_lifecycle_and_tool_discovery():
    """initialize -> initialized -> tools/list discovers the tools."""
    client = MCPClient()
    tools = await client.connect("fake", {
        "transport": "stdio", "command": sys.executable, "args": [FAKE_SERVER],
    })
    try:
        assert sorted(t._display_name for t in tools) == ["big", "echo", "getenv"]
        # Names are namespaced by server for the tool registry.
        assert all(t.name.startswith("fake__") for t in tools)
    finally:
        await client.disconnect_all()


async def test_interleaved_notifications_do_not_corrupt():
    t = await _started_transport()
    try:
        await _handshake(t)
        # The server emits a notification right before this response.
        r = await t.send("tools/call",
                         {"name": "echo", "arguments": {"text": "hello world"}})
        assert r["content"][0]["text"] == "hello world"
    finally:
        await t.stop()


async def test_large_result_survives():
    """A >64 KiB result must not raise / truncate."""
    t = await _started_transport()
    try:
        await _handshake(t)
        size = 2_000_000  # 2 MB, far past the old 64 KiB default limit
        r = await t.send("tools/call",
                         {"name": "big", "arguments": {"size": size}})
        assert len(r["content"][0]["text"]) == size
    finally:
        await t.stop()


async def test_out_of_order_id_matching():
    """Parallel calls resolving out of order must not swap results."""
    t = await _started_transport()
    try:
        await _handshake(t)
        slow = asyncio.create_task(t.send(
            "tools/call", {"name": "echo", "arguments": {"text": "SLOW", "delay": 0.4}}))
        fast = asyncio.create_task(t.send(
            "tools/call", {"name": "echo", "arguments": {"text": "FAST", "delay": 0.05}}))
        r_slow, r_fast = await asyncio.gather(slow, fast)
        assert r_slow["content"][0]["text"] == "SLOW"
        assert r_fast["content"][0]["text"] == "FAST"
    finally:
        await t.stop()


async def test_stderr_noise_does_not_deadlock():
    """A server that floods stderr must not block requests."""
    t = await _started_transport()
    try:
        # initialize alone dumps ~48 KB to stderr (past the 64 KiB pipe would
        # block the old code); several calls add more.
        await _handshake(t)
        for i in range(5):
            r = await t.send("tools/call",
                             {"name": "echo", "arguments": {"text": f"n{i}"}})
            assert r["content"][0]["text"] == f"n{i}"
    finally:
        await t.stop()


async def test_timeout_does_not_poison_stream():
    """A timed-out request's late response must not become the next result."""
    t = await _started_transport()
    try:
        await _handshake(t)
        with pytest.raises(RuntimeError):
            await t.send("tools/call",
                         {"name": "echo", "arguments": {"text": "LATE", "delay": 0.5}},
                         timeout=0.1)
        # The LATE response arrives ~0.4s later; it must be dropped, and this
        # fresh request must receive its OWN result.
        r = await t.send("tools/call",
                         {"name": "echo", "arguments": {"text": "FRESH"}})
        assert r["content"][0]["text"] == "FRESH"
    finally:
        await t.stop()


async def test_env_expansion_reaches_subprocess(monkeypatch):
    """expand_mcp_env is wired into connect(): ${VAR} reaches the spawned server."""
    monkeypatch.setenv("SPARK_TEST_MCP_TOKEN", "sekret-42")
    client = MCPClient()
    tools = await client.connect("fake", {
        "transport": "stdio",
        "command": sys.executable,
        "args": [FAKE_SERVER],
        "env": {"MY_TOKEN": "${SPARK_TEST_MCP_TOKEN}"},
    })
    try:
        getenv = next(t for t in tools if t._display_name == "getenv")
        out = await getenv.execute(key="MY_TOKEN")
        assert out == "sekret-42"
    finally:
        await client.disconnect_all()


async def test_send_before_start_raises():
    t = StdioTransport(sys.executable, [FAKE_SERVER])
    with pytest.raises(RuntimeError):
        await t.send("tools/list")


async def test_stop_is_idempotent_and_clean():
    t = await _started_transport()
    await _handshake(t)
    await t.stop()
    await t.stop()  # second call must be a no-op, not raise
    assert t.process is not None and t.process.returncode is not None


# --------------------------------------------------------------------------- #
# registry env expansion (unit)
# --------------------------------------------------------------------------- #
def test_expand_mcp_env_sets_defined_and_leaves_undefined(monkeypatch):
    monkeypatch.setenv("SPARK_DEFINED", "value-1")
    monkeypatch.delenv("SPARK_UNDEFINED", raising=False)
    cfg = {"command": "srv", "env": {
        "A": "${SPARK_DEFINED}",
        "B": "prefix-${SPARK_DEFINED}-suffix",
        "C": "$SPARK_DEFINED",
        "D": "${SPARK_UNDEFINED}",
        "E": "plain",
    }}
    out = expand_mcp_env(cfg)
    assert out["env"]["A"] == "value-1"
    assert out["env"]["B"] == "prefix-value-1-suffix"
    assert out["env"]["C"] == "value-1"
    assert out["env"]["D"] == "${SPARK_UNDEFINED}"  # left literal
    assert out["env"]["E"] == "plain"
    # original config is not mutated
    assert cfg["env"]["A"] == "${SPARK_DEFINED}"


# --------------------------------------------------------------------------- #
# Streamable-HTTP transport
# --------------------------------------------------------------------------- #
async def test_http_transport_session_and_id_matching():
    import httpx

    state = {"session": "sess-abc"}

    def handler(request: "httpx.Request") -> "httpx.Response":
        import json as _json
        body = _json.loads(request.content)
        method = body.get("method")
        rid = body.get("id")
        if method == "initialize":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": rid,
                "result": {"protocolVersion": "2024-11-05",
                           "capabilities": {"tools": {}}},
            }, headers={"Mcp-Session-Id": state["session"],
                        "Content-Type": "application/json"})
        # every non-initialize request must carry the negotiated session id
        assert request.headers.get("Mcp-Session-Id") == state["session"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": rid, "result": {"tools": [
                    {"name": "ping", "description": "ping",
                     "inputSchema": {"type": "object", "properties": {}}}]},
            })
        if method == "tools/call":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": "http-ok"}]}})
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": "no"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = HTTPTransport("http://mock.local/mcp", client=client)
    await t.start()
    try:
        init = await t.send("initialize", INIT_PARAMS)
        assert init["capabilities"]["tools"] == {}
        assert t._session_id == "sess-abc"
        await t.notify("notifications/initialized")
        listed = await t.send("tools/list")
        assert listed["tools"][0]["name"] == "ping"
        called = await t.send("tools/call", {"name": "ping", "arguments": {}})
        assert called["content"][0]["text"] == "http-ok"
    finally:
        await t.stop()


async def test_http_transport_parses_sse_framed_response():
    import httpx

    def handler(request: "httpx.Request") -> "httpx.Response":
        import json as _json
        rid = _json.loads(request.content).get("id")
        # A notification (no id) then the real response, SSE-framed.
        body = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","method":"notifications/message",'
            '"params":{"data":"noise"}}\n\n'
            "event: message\n"
            'data: {"jsonrpc":"2.0","id":' + str(rid) +
            ',"result":{"tools":[]}}\n\n'
        )
        return httpx.Response(200, text=body,
                              headers={"Content-Type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    t = HTTPTransport("http://mock.local/mcp", client=client)
    await t.start()
    try:
        result = await t.send("tools/list")
        assert result == {"tools": []}
    finally:
        await t.stop()


# --------------------------------------------------------------------------- #
# MCP image-content routing (Phase 5 Task 5)
# --------------------------------------------------------------------------- #
# MCPTool.execute() — a fake transport stub (a "fake result dict") stands in
# for a real server here; the stdio round trip is already proven above, this
# section is purely about content-array -> sentinel handling.
# --------------------------------------------------------------------------- #


class _FakeToolCallTransport:
    """Transport stub whose ``send`` returns a pre-canned ``tools/call``
    result dict, so MCPTool.execute()'s content-array handling can be
    exercised directly without a real subprocess."""

    def __init__(self, result: dict):
        self._result = result

    async def send(self, method, params=None):
        return self._result


TINY_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-mcp-image-bytes"


def _image_item(data: bytes = TINY_PNG_BYTES, mime: str = "image/png") -> dict:
    return {"type": "image", "data": base64.b64encode(data).decode("ascii"),
            "mimeType": mime}


async def test_mcptool_execute_returns_sentinel_for_image_item():
    tool = MCPTool("srv", {"name": "screenshot", "description": "d"},
                   _FakeToolCallTransport({"content": [_image_item()]}))
    result = await tool.execute()

    assert result.startswith(MCP_IMAGE_SENTINEL_PREFIX)
    mime, _, path = result[len(MCP_IMAGE_SENTINEL_PREFIX):].partition(":")
    assert mime == "image/png"
    with open(path, "rb") as f:
        assert f.read() == TINY_PNG_BYTES
    os.remove(path)


async def test_mcptool_execute_mixed_text_and_image_content():
    tool = MCPTool("srv", {"name": "screenshot", "description": "d"},
                   _FakeToolCallTransport({"content": [
                       {"type": "text", "text": "Captured the page."},
                       _image_item(),
                   ]}))
    result = await tool.execute()

    lines = result.split("\n")
    assert lines[0] == "Captured the page."
    assert lines[1].startswith(MCP_IMAGE_SENTINEL_PREFIX)
    path = lines[1][len(MCP_IMAGE_SENTINEL_PREFIX):].split(":", 1)[1]
    os.remove(path)


async def test_mcptool_execute_multiple_images_all_get_sentinels():
    tool = MCPTool("srv", {"name": "multi_shot", "description": "d"},
                   _FakeToolCallTransport({"content": [
                       _image_item(b"first-image-bytes"),
                       _image_item(b"second-image-bytes"),
                   ]}))
    result = await tool.execute()

    lines = [ln for ln in result.split("\n")
             if ln.startswith(MCP_IMAGE_SENTINEL_PREFIX)]
    assert len(lines) == 2
    paths = [ln.split(":", 2)[-1] for ln in lines]
    assert paths[0] != paths[1]
    with open(paths[0], "rb") as f:
        assert f.read() == b"first-image-bytes"
    with open(paths[1], "rb") as f:
        assert f.read() == b"second-image-bytes"
    for p in paths:
        os.remove(p)


async def test_mcptool_execute_image_missing_data_falls_back_to_placeholder():
    """No ``data`` field (a non-compliant/odd server) must degrade to the
    original inert placeholder, never crash and never emit a sentinel that
    points nowhere."""
    tool = MCPTool("srv", {"name": "screenshot", "description": "d"},
                   _FakeToolCallTransport({"content": [
                       {"type": "image", "mimeType": "image/png"}]}))
    result = await tool.execute()

    assert result == "[Image: image/png]"
    assert MCP_IMAGE_SENTINEL_PREFIX not in result


async def test_mcptool_execute_image_malformed_base64_falls_back_to_placeholder():
    tool = MCPTool("srv", {"name": "screenshot", "description": "d"},
                   _FakeToolCallTransport({"content": [
                       {"type": "image", "data": "not-valid-base64!!!",
                        "mimeType": "image/jpeg"}]}))
    result = await tool.execute()

    assert result == "[Image: image/jpeg]"
    assert MCP_IMAGE_SENTINEL_PREFIX not in result


# --------------------------------------------------------------------------- #
# Agent-level image injection — mirrors tests/test_simulator.py's Agent-level
# section exactly (same buffer-and-flush-after-round mechanism, same
# transcript-validity invariant), but driving an MCP-shaped tool result
# instead of simulator_screenshot's.
# --------------------------------------------------------------------------- #


class _FakeMockChatModel:
    """Drives Agent.run() with a scripted sequence of chunk lists — same
    shape as test_agent.py's/test_simulator.py's MockModel, kept local so
    this file has no cross-module test dependency."""

    total_input_tokens = 0
    total_output_tokens = 0

    def __init__(self, responses, supports_vision: bool = True):
        self.responses = list(responses)
        self._call = 0
        self.supports_vision = supports_vision

    async def chat(self, **kwargs):
        if self._call < len(self.responses):
            chunks = self.responses[self._call]
            self._call += 1
            for chunk in chunks:
                yield chunk

    async def close(self):
        pass


class _ScriptedMCPImageTool(Tool):
    """A fake MCP tool (e.g. ``browser__browser_take_screenshot``) that
    returns a pre-canned result — lets the agent-level tests drive
    Agent._store_tool_result/_resolve_mcp_images without going through a
    real transport."""

    name = "browser__browser_take_screenshot"
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
    """A read-only stand-in for read_file so a mixed round can pair a
    genuine non-screenshot read tool with the MCP image tool."""

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
    outside such a run. Mirrors tests/test_simulator.py's helper of the same
    name — the pre-Phase-4-Task-4-fix bug (image spliced into the middle of
    a multi-tool round) violates this."""
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


def _make_agent(model, tools):
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


def _make_mcp_image_file(tmp_dir, data: bytes = TINY_PNG_BYTES) -> str:
    path = os.path.join(tmp_dir, "mcp_screenshot.png")
    with open(path, "wb") as f:
        f.write(data)
    return path


async def test_vision_model_buffers_and_injects_mcp_image(tmp_dir):
    png_path = _make_mcp_image_file(tmp_dir)
    canned = f"{MCP_IMAGE_SENTINEL_PREFIX}image/png:{png_path}"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1",
             "name": "browser__browser_take_screenshot", "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "I can see the page."},
            {"type": "done", "usage": {}},
        ],
    ], supports_vision=True)
    agent = _make_agent(model, [_ScriptedMCPImageTool(canned)])
    await agent.run("Take a screenshot of the page and describe it")

    messages = agent.context.messages
    _assert_valid_tool_transcript(messages)
    assert agent.context.sanitize_orphaned_tool_calls() == 0

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"].startswith("Image captured")
    assert MCP_IMAGE_SENTINEL_PREFIX not in tool_msgs[0]["content"]

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
    assert base64.b64decode(b64_payload) == TINY_PNG_BYTES

    # Image lands AFTER the (only) tool result.
    last_tool_idx = max(
        idx for idx, m in enumerate(messages) if m["role"] == "tool")
    image_idx = next(
        idx for idx, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], list))
    assert image_idx > last_tool_idx


async def test_non_vision_model_keeps_placeholder_no_image_injected(tmp_dir):
    """A non-vision model must never see a buffered image turn — it gets
    back the same inert placeholder MCPTool always produced before this
    task, even though the tool itself always emits a sentinel now (MCPTool
    has no model reference; the gate lives in Agent._resolve_mcp_images)."""
    png_path = _make_mcp_image_file(tmp_dir)
    canned = f"{MCP_IMAGE_SENTINEL_PREFIX}image/png:{png_path}"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1",
             "name": "browser__browser_take_screenshot", "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "No vision available."},
            {"type": "done", "usage": {}},
        ],
    ], supports_vision=False)
    agent = _make_agent(model, [_ScriptedMCPImageTool(canned)])
    await agent.run("Take a screenshot of the page")

    messages = agent.context.messages
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "[Image: image/png]"
    assert MCP_IMAGE_SENTINEL_PREFIX not in tool_msgs[0]["content"]

    image_msgs = [
        m for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert image_msgs == []


async def test_multiple_images_in_one_mcp_result_all_land(tmp_dir):
    path_a = _make_mcp_image_file(tmp_dir, b"image-A-bytes")
    path_b = os.path.join(tmp_dir, "second.png")
    with open(path_b, "wb") as f:
        f.write(b"image-B-bytes")
    canned = "\n".join([
        "Captured 2 screenshots.",
        f"{MCP_IMAGE_SENTINEL_PREFIX}image/png:{path_a}",
        f"{MCP_IMAGE_SENTINEL_PREFIX}image/png:{path_b}",
    ])
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1",
             "name": "browser__browser_take_screenshot", "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Compared both."},
            {"type": "done", "usage": {}},
        ],
    ], supports_vision=True)
    agent = _make_agent(model, [_ScriptedMCPImageTool(canned)])
    await agent.run("Take two screenshots")

    messages = agent.context.messages
    _assert_valid_tool_transcript(messages)
    assert agent.context.sanitize_orphaned_tool_calls() == 0

    image_msgs = [
        m for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(image_msgs) == 2
    payloads = {
        base64.b64decode(m["content"][1]["image_url"]["url"].split("base64,", 1)[1])
        for m in image_msgs
    }
    assert payloads == {b"image-A-bytes", b"image-B-bytes"}


async def test_parallel_path_also_injects_mcp_image(tmp_dir):
    """MCP screenshot tool called ALONGSIDE another read-only tool in one
    round (parallel path) — the image must land AFTER the round's complete
    tool-result run, never spliced into the middle of it."""
    png_path = _make_mcp_image_file(tmp_dir)
    canned = f"{MCP_IMAGE_SENTINEL_PREFIX}image/png:{png_path}"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1",
             "name": "browser__browser_take_screenshot", "arguments": {}},
            {"type": "tool_call", "id": "call_2", "name": "peek_tool",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Done."},
            {"type": "done", "usage": {}},
        ],
    ], supports_vision=True)
    agent = _make_agent(
        model, [_ScriptedMCPImageTool(canned), _OtherReadOnlyTool()])
    await agent.run("Screenshot and peek")

    messages = agent.context.messages
    _assert_valid_tool_transcript(messages)
    assert agent.context.sanitize_orphaned_tool_calls() == 0

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert set(by_id) == {"call_1", "call_2"}
    assert by_id["call_2"] == "peeked"
    assert by_id["call_1"].startswith("Image captured")

    image_indices = [
        idx for idx, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(image_indices) == 1
    last_tool_idx = max(
        idx for idx, m in enumerate(messages) if m["role"] == "tool")
    assert image_indices[0] > last_tool_idx


async def test_mixed_round_read_file_and_mcp_image_keeps_transcript_valid(tmp_dir):
    """The exact mixed-round case Phase 4 Task 4 guarded against, replayed
    for an MCP tool: model calls [read_file, browser_take_screenshot,
    peek_tool] in ONE round, with the screenshot in the MIDDLE (so a naive
    immediate-inject would splice the image between two real tool
    results). Result must be a valid transcript with the image present,
    landing after the last tool result — no orphaned tool_calls."""
    png_path = _make_mcp_image_file(tmp_dir)
    canned = f"{MCP_IMAGE_SENTINEL_PREFIX}image/png:{png_path}"
    model = _FakeMockChatModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "read_file",
             "arguments": {"file_path": "App.tsx"}},
            {"type": "tool_call", "id": "call_2",
             "name": "browser__browser_take_screenshot", "arguments": {}},
            {"type": "tool_call", "id": "call_3", "name": "peek_tool",
             "arguments": {}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Compared code to the rendered page."},
            {"type": "done", "usage": {}},
        ],
    ], supports_vision=True)
    agent = _make_agent(
        model,
        [_FakeReadFileTool(), _ScriptedMCPImageTool(canned), _OtherReadOnlyTool()])
    await agent.run("Read the file and screenshot the page")

    messages = agent.context.messages
    _assert_valid_tool_transcript(messages)
    assert agent.context.sanitize_orphaned_tool_calls() == 0

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert set(by_id) == {"call_1", "call_2", "call_3"}
    assert by_id["call_2"].startswith("Image captured")

    image_indices = [
        idx for idx, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(image_indices) == 1
    last_tool_idx = max(
        idx for idx, m in enumerate(messages) if m["role"] == "tool")
    assert image_indices[0] > last_tool_idx
    url = messages[image_indices[0]]["content"][1]["image_url"]["url"]
    assert base64.b64decode(url.split("base64,", 1)[1]) == TINY_PNG_BYTES
