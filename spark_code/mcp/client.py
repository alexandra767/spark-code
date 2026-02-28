"""MCP client — connects to MCP servers and exposes their tools."""

import asyncio
from typing import Any

from ..tools.base import Tool
from .transport import StdioTransport, SSETransport


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
                parts.append(f"[Image: {item.get('mimeType', 'image')}]")
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
        transport_type = config.get("transport", "stdio")

        if transport_type == "stdio":
            command = config.get("command", "")
            args = config.get("args", [])
            env = config.get("env", {})
            transport = StdioTransport(command, args, env)
        elif transport_type in ("sse", "http"):
            url = config.get("url", "")
            transport = SSETransport(url)
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
