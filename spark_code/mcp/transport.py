"""MCP transport layer — stdio and SSE."""

import asyncio
import json
import os


class StdioTransport:
    """Communicate with MCP server via stdin/stdout."""

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0

    async def start(self):
        """Start the MCP server process."""
        full_env = {**os.environ, **self.env}
        self.process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )

    async def send(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and wait for response."""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Transport not started")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        payload = json.dumps(request) + "\n"
        self.process.stdin.write(payload.encode())
        await self.process.stdin.drain()

        # Read response line
        line = await asyncio.wait_for(
            self.process.stdout.readline(), timeout=30.0
        )
        if not line:
            raise RuntimeError("MCP server closed connection")

        response = json.loads(line.decode())
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        return response.get("result", {})

    async def stop(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()


class SSETransport:
    """Communicate with MCP server via Server-Sent Events (HTTP)."""

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self._client = None

    async def start(self):
        """Initialize HTTP client."""
        import httpx
        self._client = httpx.AsyncClient(timeout=30.0, headers=self.headers)

    async def send(self, method: str, params: dict | None = None) -> dict:
        """Send request via HTTP POST."""
        if not self._client:
            raise RuntimeError("Transport not started")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
        }
        if params:
            payload["params"] = params

        response = await self._client.post(
            f"{self.url}/message",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    async def stop(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
