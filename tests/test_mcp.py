"""Regression tests for the MCP transport + client.

These prove the audit fixes:
  * stdio: id-matching, interleaved notifications don't corrupt responses, large
    results survive, a chatty stderr never deadlocks, a timed-out request never
    poisons the stream, env expansion reaches the spawned subprocess, and the
    initialize -> notifications/initialized -> tools/list lifecycle runs.
  * http: minimal Streamable-HTTP client negotiates Mcp-Session-Id and matches
    responses by id.
"""

import asyncio
import os
import sys

import pytest

from spark_code.mcp.client import MCPClient
from spark_code.mcp.registry import expand_mcp_env
from spark_code.mcp.transport import HTTPTransport, StdioTransport

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
