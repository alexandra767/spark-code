"""MCP client — connects to MCP servers and exposes their tools."""

import asyncio
import base64
import os
import tempfile
from typing import Any

from ..tools.base import Tool
from .registry import expand_mcp_env
from .transport import HTTPTransport, StdioTransport

# Phase 5 Task 5: marks an MCP ``image`` content item as "decoded and saved
# to the temp file at the path that follows (mime type before it)" rather
# than the old inert ``[Image: <mime>]`` text placeholder. An MCP tool
# result's content array can contain base64 image bytes directly (e.g.
# ``browser_take_screenshot``), but tool results are plain strings subject
# to per-tool truncation (agent.py's ``_budget_for``/``_truncate_result``)
# before they ever reach ``Agent._store_tool_result`` — inlining the raw
# base64 payload would risk it being clipped mid-stream and corrupted. So,
# exactly like ``tools/simulator.py``'s ``SCREENSHOT_SENTINEL_PREFIX``, the
# actual bytes never travel through the tool-result string: they're decoded
# and written to a temp file here, and only this short sentinel (well under
# any budget) travels in the result text. ``Agent._store_tool_result``
# detects it — that's also where ``model.supports_vision`` is checked
# (MCPTool has no model reference of its own, unlike
# ``SimulatorScreenshotTool``): a vision-capable model gets the file read
# back + base64-buffered into the same ``_pending_image_injections`` /
# ``_flush_pending_images`` mechanism Phase 4 Task 4 built; a non-vision
# model instead gets back the original ``[Image: <mime>]`` placeholder,
# unchanged from before this task. Shape:
# ``f"{MCP_IMAGE_SENTINEL_PREFIX}{mime_type}:{path}"`` — one per image, each
# on its own line, so a result mixing text with N images (a multi-screenshot
# call) stays a single string with everything else passed through untouched.
MCP_IMAGE_SENTINEL_PREFIX = "__SPARK_MCP_IMAGE__:"


def _save_mcp_image_to_temp(data_b64: str, mime_type: str) -> str:
    """Decode an MCP image content item's base64 payload to a fresh temp
    file and return its path. Raises on malformed base64 / IO failure —
    callers catch broadly and fall back to the plain-text placeholder, the
    same "never crash on an unexpected shape" contract the rest of the MCP
    client follows."""
    ext = mime_type.rsplit("/", 1)[-1].split("+")[0]
    if not ext.isalnum():
        ext = "bin"
    fd, path = tempfile.mkstemp(prefix="spark_mcp_img_", suffix=f".{ext}")
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64decode(data_b64))
    return path


class MCPTool(Tool):
    """A tool provided by an MCP server."""

    def __init__(self, server_name: str, tool_def: dict, transport):
        self._name = f"{server_name}__{tool_def['name']}"
        self._display_name = tool_def["name"]
        self._description = tool_def.get("description", f"MCP tool from {server_name}")
        self._parameters = tool_def.get("inputSchema", {"type": "object", "properties": {}})
        self._server_name = server_name
        self._transport = transport

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"[{self._server_name}] {self._description}"

    @property
    def parameters(self) -> dict:
        return self._parameters

    async def execute(self, **kwargs) -> str:
        """Call the MCP server tool."""
        result = await self._transport.send("tools/call", {
            "name": self._display_name,
            "arguments": kwargs,
        })

        # MCP tools return content array
        content = result.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "image":
                mime_type = item.get("mimeType", "image")
                data = item.get("data")
                if data:
                    try:
                        path = _save_mcp_image_to_temp(data, mime_type)
                        parts.append(f"{MCP_IMAGE_SENTINEL_PREFIX}{mime_type}:{path}")
                    except Exception:
                        # Malformed base64 / IO failure — degrade to the old
                        # inert placeholder rather than raise.
                        parts.append(f"[Image: {mime_type}]")
                else:
                    parts.append(f"[Image: {mime_type}]")
            else:
                parts.append(str(item))

        return "\n".join(parts) if parts else str(result)


class MCPClient:
    """Manages connections to MCP servers."""

    def __init__(self):
        self.servers: dict[str, Any] = {}
        self.tools: list[MCPTool] = []

    async def connect(self, name: str, config: dict):
        """Connect to an MCP server and discover its tools."""
        # Expand ${VAR} references in env before spawning the server.
        config = expand_mcp_env(config)
        transport_type = config.get("transport", "stdio")

        if transport_type == "stdio":
            command = config.get("command", "")
            args = config.get("args", [])
            env = config.get("env", {})
            transport = StdioTransport(command, args, env)
        elif transport_type in ("sse", "http"):
            url = config.get("url", "")
            headers = config.get("headers", {})
            transport = HTTPTransport(url, headers=headers)
        else:
            raise ValueError(f"Unknown transport: {transport_type}")

        try:
            await transport.start()

            # Initialize MCP session
            await transport.send("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "spark-code", "version": "0.1.0"},
            })

            # Spec-required: tell the server initialization is complete. Without
            # this notification, compliant SDK servers reject/queue tools/list.
            await transport.notify("notifications/initialized")

            # Discover tools
            result = await transport.send("tools/list")
            server_tools = result.get("tools", [])

            mcp_tools = []
            for tool_def in server_tools:
                mcp_tool = MCPTool(name, tool_def, transport)
                mcp_tools.append(mcp_tool)

            self.servers[name] = {
                "transport": transport,
                "config": config,
                "tool_count": len(mcp_tools),
            }
            self.tools.extend(mcp_tools)

            return mcp_tools

        except asyncio.CancelledError:
            # A cancellation (e.g. an outer asyncio.wait_for timeout firing
            # while we're awaiting `initialize`) raises CancelledError, which
            # is a BaseException — NOT caught by the `except Exception` below.
            # Without this branch the transport (and its spawned subprocess +
            # reader/stderr tasks) would be orphaned, since it isn't yet in
            # self.servers to be torn down by disconnect_all. stop() is
            # idempotent and safe even when the handshake never completed
            # (it guards a None process), then we re-raise so the cancellation
            # keeps propagating.
            await transport.stop()
            raise
        except Exception as e:
            await transport.stop()
            raise RuntimeError(f"Failed to connect to MCP server '{name}': {e}")

    async def connect_all(self, mcp_configs: dict) -> list[MCPTool]:
        """Connect to all configured MCP servers."""
        all_tools = []
        for name, config in mcp_configs.items():
            try:
                tools = await self.connect(name, config)
                all_tools.extend(tools)
            except Exception as e:
                # Don't fail startup if an MCP server is unavailable
                print(f"  Warning: MCP server '{name}' failed: {e}")
        return all_tools

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for name, info in self.servers.items():
            try:
                await info["transport"].stop()
            except Exception:
                pass
        self.servers.clear()
        self.tools.clear()
