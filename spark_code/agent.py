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


class Agent:
    """The agent loop that connects the model to tools."""

    MAX_TOOL_ROUNDS = 25  # Safety limit

    def __init__(self, model: ModelClient, context: Context,
                 tools: ToolRegistry, permissions: PermissionManager,
                 console: Console | None = None):
        self.model = model
        self.context = context
        self.tools = tools
        self.permissions = permissions
        self.console = console or Console()

    async def run(self, user_input: str) -> str:
        """Process user input through the agent loop.

        Returns the final text response from the model.
        """
        self.context.add_user(user_input)

        full_response = ""
        rounds = 0

        while rounds < self.MAX_TOOL_ROUNDS:
            rounds += 1

            # Collect response from model
            text_parts = []
            tool_calls = []

            # Show spinner while waiting for first token
            with self.console.status("[bold cyan]Thinking...", spinner="dots"):
                first_chunk = True
                async for chunk in self.model.chat(
                    messages=self.context.get_messages(),
                    tools=self.tools.schemas(),
                    stream=True,
                ):
                    if first_chunk:
                        first_chunk = False
                        # Clear the spinner by doing nothing — status context exits

                    if chunk["type"] == "text":
                        text_parts.append(chunk["content"])
                        # Stream text to terminal
                        self.console.print(chunk["content"], end="", highlight=False)

                    elif chunk["type"] == "tool_call":
                        tool_calls.append(chunk)

                    elif chunk["type"] == "done":
                        pass

            text = "".join(text_parts)

            # If we got text, print newline
            if text:
                self.console.print()
                full_response += text

            # No tool calls — model is done
            if not tool_calls:
                self.context.add_assistant(text)
                break

            # Process tool calls
            self.context.add_assistant_tool_calls(tool_calls)

            for tc in tool_calls:
                tool = self.tools.get(tc["name"])
                if not tool:
                    result = f"Error: Unknown tool '{tc['name']}'"
                    self.context.add_tool_result(tc["id"], tc["name"], result)
                    continue

                # Format details for permission prompt
                args_str = ", ".join(f"{k}={repr(v)[:80]}" for k, v in tc["arguments"].items())
                details = f"{tc['name']}({args_str})"

                # Check permission
                if not self.permissions.check(tc["name"], tool.is_read_only, details):
                    result = "Permission denied by user."
                    self.console.print(f"  [red]✗ {tc['name']} — denied[/red]")
                    self.context.add_tool_result(tc["id"], tc["name"], result)
                    continue

                # Execute tool
                self.console.print(f"  [cyan]⚡ {details}[/cyan]")
                try:
                    result = await tool.execute(**tc["arguments"])
                except Exception as e:
                    result = f"Error executing {tc['name']}: {e}"

                # Truncate very long results
                if len(result) > 15000:
                    result = result[:15000] + "\n\n... (truncated)"

                self.context.add_tool_result(tc["id"], tc["name"], result)

                # Show brief result
                lines = result.split("\n")
                preview = lines[0][:100] if lines else ""
                self.console.print(f"  [dim]→ {preview}[/dim]")

            # Continue loop — model will process tool results

        if rounds >= self.MAX_TOOL_ROUNDS:
            self.console.print("[yellow]Warning: Reached maximum tool rounds[/yellow]")

        return full_response
