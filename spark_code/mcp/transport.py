"""MCP transport layer — stdio (hand-rolled JSON-RPC) and Streamable HTTP.

The stdio transport is a correct async JSON-RPC client:

  * a single background *reader* task consumes stdout line-by-line, parses each
    JSON-RPC message and dispatches it to the waiting request by ``id`` via an
    ``id -> asyncio.Future`` map;
  * a background *stderr drain* task continuously reads stderr so a chatty
    server can never fill (and block on) the 64 KiB pipe buffer;
  * ``send()`` registers a Future for its request id, writes the request and
    awaits the Future with a timeout — a timed-out request removes its Future so
    a late response can never be mistaken for the next request's result.

The HTTP transport is a minimal but correct Streamable-HTTP client: it POSTs a
JSON-RPC request, reads the JSON (or SSE-framed) response, matches it by id and
negotiates the ``Mcp-Session-Id`` header.
"""

import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

# asyncio's default StreamReader limit is 64 KiB; an MCP tool result larger than
# that makes readline() raise. Use a generous limit so big results survive.
_STREAM_LIMIT = 10 * 1024 * 1024  # 10 MB

_DEFAULT_TIMEOUT = 30.0


class StdioTransport:
    """Communicate with an MCP server via stdin/stdout using JSON-RPC.

    Public interface (used by MCPClient):
        await start()
        await send(method, params=None, timeout=...) -> result dict
        await notify(method, params=None)            -> None
        await stop()
    """

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: asyncio.subprocess.Process | None = None
        self.default_timeout = _DEFAULT_TIMEOUT

        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._closed = False

    async def start(self):
        """Start the MCP server process and its background reader tasks."""
        full_env = {**os.environ, **{k: str(v) for k, v in self.env.items()}}
        self.process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
            limit=_STREAM_LIMIT,
        )
        self._reader_task = asyncio.ensure_future(self._reader_loop())
        self._stderr_task = asyncio.ensure_future(self._stderr_drain())

    # ------------------------------------------------------------------ #
    # Background tasks
    # ------------------------------------------------------------------ #
    async def _reader_loop(self):
        """Read stdout line-by-line and dispatch each JSON-RPC message by id."""
        assert self.process is not None and self.process.stdout is not None
        stream = self.process.stdout
        try:
            while True:
                try:
                    line = await stream.readline()
                except (asyncio.LimitOverrunError, ValueError) as e:
                    # A single line exceeded _STREAM_LIMIT. The stream can no
                    # longer be framed reliably; treat as fatal.
                    logger.error("MCP stdout line exceeded limit: %s", e)
                    self._fail_all_pending(
                        RuntimeError("MCP response exceeded stream limit"))
                    break
                if not line:
                    break  # EOF — server exited / closed stdout
                text = line.strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    # Some servers print a banner / non-JSON noise to stdout.
                    logger.debug("Ignoring non-JSON stdout line: %r", text[:200])
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("MCP reader loop error: %s", e)
        finally:
            self._fail_all_pending(RuntimeError("MCP server closed connection"))

    def _dispatch(self, msg):
        """Route one parsed JSON-RPC message to its waiting Future (by id)."""
        if not isinstance(msg, dict):
            return
        mid = msg.get("id")
        if mid is not None and mid in self._pending:
            fut = self._pending.pop(mid)
            if not fut.done():
                if "error" in msg:
                    fut.set_exception(RuntimeError(f"MCP error: {msg['error']}"))
                else:
                    fut.set_result(msg.get("result", {}))
        elif mid is None:
            # Notification (logging / progress / etc.) — no id, nothing awaits it.
            logger.debug("MCP notification: %s", msg.get("method"))
        else:
            # Response for an unknown / already-timed-out id — drop it so it can
            # never poison a subsequent request.
            logger.debug("Dropping MCP response for unmatched id %r", mid)

    async def _stderr_drain(self):
        """Continuously drain stderr so a chatty server never blocks on a full pipe."""
        assert self.process is not None
        stream = self.process.stderr
        if stream is None:
            return
        try:
            while True:
                try:
                    line = await stream.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    # Overly long stderr line — skip and keep draining.
                    continue
                if not line:
                    break
                logger.debug("[MCP %s stderr] %s", self.command,
                             line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("MCP stderr drain error: %s", e)

    def _fail_all_pending(self, exc: Exception):
        for _rid, fut in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    # ------------------------------------------------------------------ #
    # Requests / notifications
    # ------------------------------------------------------------------ #
    async def send(self, method: str, params: dict | None = None,
                   timeout: float | None = None) -> dict:
        """Send a JSON-RPC request and await the matching response."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Transport not started")
        if self.process.returncode is not None:
            raise RuntimeError("MCP server process has exited")

        timeout = self.default_timeout if timeout is None else timeout
        self._request_id += 1
        req_id = self._request_id
        request = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params:
            request["params"] = params

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        try:
            self.process.stdin.write((json.dumps(request) + "\n").encode())
            await self.process.stdin.drain()
        except Exception:
            self._pending.pop(req_id, None)
            raise

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            # Remove the Future so a late response is dropped, not misdelivered.
            self._pending.pop(req_id, None)
            raise RuntimeError(
                f"MCP request '{method}' timed out after {timeout}s")
        finally:
            self._pending.pop(req_id, None)

    async def notify(self, method: str, params: dict | None = None):
        """Send a JSON-RPC notification (no id, no response expected)."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Transport not started")
        message = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def stop(self):
        """Cancel background tasks, terminate the subprocess and close streams."""
        if self._closed:
            return
        self._closed = True

        for task in (self._reader_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._reader_task = None
        self._stderr_task = None

        self._fail_all_pending(RuntimeError("Transport stopped"))

        if self.process:
            proc = self.process
            if proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await proc.wait()
                    except Exception:
                        pass
            # Close stdin writer to release the pipe (stdout/stderr readers are
            # torn down with the process).
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass


class HTTPTransport:
    """Minimal Streamable-HTTP MCP client.

    POSTs JSON-RPC requests to ``url``; reads the JSON (or SSE-framed) response,
    matches it by id, and negotiates the ``Mcp-Session-Id`` header returned by
    ``initialize``. This is what modern MCP servers ("Streamable HTTP") speak.

    ``client`` may be injected (an ``httpx.AsyncClient``) for testing; otherwise
    one is created in :meth:`start`.
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None,
                 client=None):
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self.default_timeout = _DEFAULT_TIMEOUT
        self._client = client
        self._owns_client = client is None
        self._session_id: str | None = None
        self._request_id = 0

    async def start(self):
        """Initialize the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self.default_timeout, headers=self.headers)

    def _base_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(self.headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    @staticmethod
    def _parse_sse(text: str, want_id) -> dict | None:
        """Extract the JSON-RPC message matching ``want_id`` from an SSE body."""
        data_lines: list[str] = []
        for raw in text.splitlines():
            if raw.startswith("data:"):
                data_lines.append(raw[5:].lstrip())
            elif raw.strip() == "":
                if data_lines:
                    payload = "\n".join(data_lines)
                    data_lines = []
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("id") == want_id:
                        return obj
        if data_lines:
            try:
                obj = json.loads("\n".join(data_lines))
                if isinstance(obj, dict) and obj.get("id") == want_id:
                    return obj
            except json.JSONDecodeError:
                pass
        return None

    def _capture_session(self, response):
        sid = response.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid

    async def send(self, method: str, params: dict | None = None,
                   timeout: float | None = None) -> dict:
        if not self._client:
            raise RuntimeError("Transport not started")

        timeout = self.default_timeout if timeout is None else timeout
        self._request_id += 1
        req_id = self._request_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params:
            payload["params"] = params

        response = await self._client.post(
            self.url, json=payload, headers=self._base_headers(), timeout=timeout)
        response.raise_for_status()
        self._capture_session(response)

        ctype = response.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            data = self._parse_sse(response.text, req_id)
        else:
            data = response.json()
            if isinstance(data, list):
                data = next((m for m in data
                             if isinstance(m, dict) and m.get("id") == req_id), None)

        if data is None:
            raise RuntimeError(
                f"No JSON-RPC response matching id {req_id} for '{method}'")
        if data.get("id") != req_id:
            raise RuntimeError(
                f"MCP response id mismatch: expected {req_id}, got {data.get('id')}")
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    async def notify(self, method: str, params: dict | None = None):
        if not self._client:
            raise RuntimeError("Transport not started")
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        response = await self._client.post(
            self.url, json=payload, headers=self._base_headers())
        # Servers reply 202 Accepted (or 200) to notifications; ignore the body.
        if response.status_code >= 400:
            response.raise_for_status()

    async def stop(self):
        if self._client is None:
            return
        # Best-effort session termination.
        if self._session_id:
            try:
                await self._client.request(
                    "DELETE", self.url,
                    headers={"Mcp-Session-Id": self._session_id})
            except Exception:
                pass
        if self._owns_client:
            try:
                await self._client.aclose()
            except Exception:
                pass
        self._client = None


# Backwards-compatible alias: the old name referred to an (incorrect) SSE client.
# Modern servers speak Streamable HTTP, which HTTPTransport implements.
SSETransport = HTTPTransport
