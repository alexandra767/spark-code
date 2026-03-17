"""Agent loop — the core of Spark Code.

Sends messages to the model, parses tool calls, executes tools,
feeds results back, and repeats until the model gives a final answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

from .context import Context
from .model import ModelClient
from .permissions import PermissionManager
from .tools.base import ToolRegistry
from .ui.output import (
    StreamingRenderer,
    render_error,
    render_tool_call,
    render_tool_denied,
    render_tool_error,
    render_tool_result,
    render_warning,
)

if TYPE_CHECKING:
    from .stats import SessionStats


class Agent:
    """The agent loop that connects the model to tools."""

    MAX_TOOL_ROUNDS = 25  # Safety limit

    def __init__(self, model: ModelClient, context: Context,
                 tools: ToolRegistry, permissions: PermissionManager,
                 console: Console | None = None,
                 output_prefix: str = "",
                 stats: SessionStats | None = None):
        self.model = model
        self.context = context
        self.tools = tools
        self.permissions = permissions
        self.console = console or Console()
        self.output_prefix = output_prefix
        self.stats = stats

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

                # Guard: skip tool calls with None/missing arguments (not empty dict — that's valid)
                if tc.get("arguments") is None:
                    result = f"Error: Tool '{tc['name']}' called with no arguments. The response may have been truncated due to token limits."
                    self.context.add_tool_result(tc["id"], tc["name"], result)
                    render_tool_call(self.console, tc["name"], tc["arguments"])
                    render_tool_error(self.console, tc["name"], "Missing arguments — response may have been truncated")
                    continue

                # Show inline diff preview for edit_file before permission check
                if tc["name"] == "edit_file" and self.permissions.mode != "trust":
                    try:
                        from .ui.diff import render_inline_diff
                        file_path = tc["arguments"].get("file_path", "")
                        old_str = tc["arguments"].get("old_string", "")
                        new_str = tc["arguments"].get("new_string", "")
                        if file_path and old_str:
                            render_inline_diff(self.console, file_path,
                                               old_str, new_str)
                    except Exception:
                        pass  # Don't break on preview failure

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

                # Execute tool (with streaming for bash)
                is_streamed_bash = (
                    tc["name"] == "bash"
                    and tool.supports_streaming
                    and not self.output_prefix
                )
                try:
                    if is_streamed_bash:
                        # Stream bash output line-by-line
                        connector = "\u23bf"  # ⎿
                        def _print_line(line: str):
                            try:
                                t = Text(f"  {connector} ", style="#7b88a1")
                                t.append(line, style="#8899aa")
                                self.console.print(t)
                            except Exception:
                                pass  # Don't crash on display failure
                        result = await tool.execute_streaming(
                            callback=_print_line, **tc["arguments"]
                        )
                    else:
                        result = await tool.execute(**tc["arguments"])
                except Exception as e:
                    result = f"Error executing {tc['name']}: {e}"

                # Record stats
                if self.stats:
                    self.stats.record_tool_call(tc["name"], tc["arguments"])

                # Truncate very long results
                if len(result) > 15000:
                    result = result[:15000] + "\n\n... (truncated)"
                    if is_streamed_bash:
                        t = Text("  \u23bf ... (truncated)", style="#7b88a1")
                        self.console.print(t)

                self.context.add_tool_result(tc["id"], tc["name"], result)

                # Display result with smart preview (skip for streamed bash)
                if is_streamed_bash:
                    pass  # Already streamed
                else:
                    render_tool_result(self.console, result, tool_name=tc["name"])

            # Continue loop — model will process tool results

        if rounds >= self.MAX_TOOL_ROUNDS:
            render_warning(self.console, "Reached maximum tool rounds")

        return full_response
