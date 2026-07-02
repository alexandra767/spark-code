"""A tiny fake stdio MCP server for tests.

Speaks JSON-RPC 2.0 over stdin/stdout. It deliberately:
  * spews a lot of noise to *stderr* (to prove the transport drains it and never
    deadlocks on a full pipe),
  * emits an interleaved *notification* immediately before every response (to
    prove notifications don't corrupt id-matched responses),
  * handles each ``tools/call`` in a background thread with an optional ``delay``
    so parallel calls can resolve out of order (to prove id-matching),
  * can echo back an arbitrarily large blob (to prove big results survive the
    64 KiB default StreamReader limit).

Run as: ``python _mcp_fake_server.py``
"""

import json
import os
import sys
import threading
import time

_write_lock = threading.Lock()


def write_msg(obj):
    data = json.dumps(obj) + "\n"
    with _write_lock:
        sys.stdout.write(data)
        sys.stdout.flush()


def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _noise(prefix, n=60, width=800):
    for i in range(n):
        log(f"{prefix} {i} " + "x" * width)


def handle(req):
    method = req.get("method")
    rid = req.get("id")

    if method == "initialize":
        _noise("[fake-mcp] init noise")  # ~48 KB of stderr up front
        write_msg({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "1.0"},
        }})

    elif method == "notifications/initialized":
        log("[fake-mcp] received notifications/initialized")
        # notification: no response

    elif method == "tools/list":
        # interleaved notification BEFORE the response
        write_msg({"jsonrpc": "2.0", "method": "notifications/message",
                   "params": {"level": "info", "data": "listing tools"}})
        write_msg({"jsonrpc": "2.0", "id": rid, "result": {"tools": [
            {"name": "echo", "description": "Echo text back",
             "inputSchema": {"type": "object",
                             "properties": {"text": {"type": "string"}}}},
            {"name": "big", "description": "Return a big blob",
             "inputSchema": {"type": "object",
                             "properties": {"size": {"type": "integer"}}}},
            {"name": "getenv", "description": "Return an env var value",
             "inputSchema": {"type": "object",
                             "properties": {"key": {"type": "string"}}}},
        ]}})

    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        delay = float(args.get("delay", 0) or 0)

        def respond():
            if delay:
                time.sleep(delay)
            # emit a notification just before the response to test interleaving
            write_msg({"jsonrpc": "2.0", "method": "notifications/progress",
                       "params": {"progress": 1, "total": 1}})
            _noise("[fake-mcp] call noise", n=10)  # more stderr under load
            if name == "big":
                size = int(args.get("size", 1000))
                text = "A" * size
            elif name == "getenv":
                text = os.environ.get(args.get("key", ""), "")
            else:  # echo
                text = str(args.get("text", ""))
            write_msg({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": text}]}})

        threading.Thread(target=respond, daemon=True).start()

    else:
        if rid is not None:
            write_msg({"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32601, "message": f"Method not found: {method}"}})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        handle(req)


if __name__ == "__main__":
    main()
