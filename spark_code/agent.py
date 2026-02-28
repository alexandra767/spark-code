"""Agent loop — the core of Spark Code.

Sends messages to the model, parses tool calls, executes tools,
feeds results back, and repeats until the model gives a final answer.
"""

import asyncio
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from .model import ModelClient
from .context import Context
from .permissions import PermissionManager
from .tools.base import ToolRegistry
from .ui.output import (
    StreamingRenderer,
    render_tool_call,
    render_tool_result,
    render_tool_denied,
    render_tool_error,
    render_error,
    render_markdown,
    render_warning,
)


class Agent:
    """The agent loop that connects the model to tools."""

    MAX_TOOL_ROUNDS = 25  # Safety limit

    def __init__(self, model: ModelClient, context: Context,
                 tools: ToolRegistry, permissions: PermissionManager,
                 console: Console | None = None,
                 output_prefix: str = ""):
        self.model = model
        self.context = context
        self.tools = tools
        self.permissions = permissions
        self.console = console or Console()
        self.output_prefix = output_prefix

    async def run_without_user_add(self) -> str:
        """Run the agent loop without adding a user message.

        Used when the message (e.g. image) was already added to context.
        """
        return await self._agent_loop()

    async def run(self, user_input: str) -> str:
        """Process user input through the agent loop.

        Returns the final text response from the model.
        """
        self.context.add_user(user_input)
        return await self._agent_loop()

    async def _agent_loop(self) -> str:
        """Core agent loop — chat, handle tool calls, repeat."""

        full_response = ""
        rounds = 0

        while rounds < self.MAX_TOOL_ROUNDS:
            rounds += 1

            # Collect response from model
            text_parts = []
            tool_calls = []
            # Workers skip the Live display to avoid Rich conflicts
            use_live = not self.output_prefix
            renderer = StreamingRenderer(self.console, live_mode=use_live)
            renderer.start()

            async for chunk in self.model.chat(
                messages=self.context.get_messages(),
                tools=self.tools.schemas(),
                stream=True,
            ):
                if chunk["type"] == "text":
                    text_parts.append(chunk["content"])
                    renderer.feed(chunk["content"])

                elif chunk["type"] == "tool_call":
                    tool_calls.append(chunk)

                elif chunk["type"] == "done":
                    pass

            # Finalize — final markdown render and stop live display
            renderer.flush()

            text = "".join(text_parts)

            if text:
                full_response += text

            # No tool calls — model is done
            if not tool_calls:
                self.context.add_assistant(text)
                break

            # Process tool calls (live display already stopped by flush)
            self.context.add_assistant_tool_calls(tool_calls)

            for tc in tool_calls:
                tool = self.tools.get(tc["name"])
                if not tool:
                    result = f"Error: Unknown tool '{tc['name']}'"
                    self.context.add_tool_result(tc["id"], tc["name"], result)
                    render_error(self.console, f"Unknown tool '{tc['name']}'")
                    continue

                # Guard: skip tool calls with empty/missing arguments
                if not tc.get("arguments") or tc["arguments"] == {}:
                    result = f"Error: Tool '{tc['name']}' called with empty arguments. The response may have been truncated due to token limits."
                    self.context.add_tool_result(tc["id"], tc["name"], result)
                    render_tool_call(self.console, tc["name"], tc["arguments"])
                    render_tool_error(self.console, tc["name"], "Empty arguments — response may have been truncated")
                    continue

                # Check permission
                if not self.permissions.check(tc["name"], tool.is_read_only,
                                              tc["arguments"]):
                    result = "Permission denied by user."
                    render_tool_call(self.console, tc["name"], tc["arguments"])
                    render_tool_denied(self.console, tc["name"])
                    self.context.add_tool_result(tc["id"], tc["name"], result)
                    continue

                # Display tool call with styled panel
                render_tool_call(self.console, tc["name"], tc["arguments"])

                # Execute tool
                try:
                    result = await tool.execute(**tc["arguments"])
                except Exception as e:
                    result = f"Error executing {tc['name']}: {e}"

                # Truncate very long results
                if len(result) > 15000:
                    result = result[:15000] + "\n\n... (truncated)"

                self.context.add_tool_result(tc["id"], tc["name"], result)

                # Display result with smart preview
                render_tool_result(self.console, result, tool_name=tc["name"])

            # Continue loop — model will process tool results

        if rounds >= self.MAX_TOOL_ROUNDS:
            render_warning(self.console, "Reached maximum tool rounds")

        return full_response
