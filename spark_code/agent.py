"""Agent loop — the core of Spark Code.

Sends messages to the model, parses tool calls, executes tools,
feeds results back, and repeats until the model gives a final answer.
"""

from __future__ import annotations

import asyncio
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
    from .hooks import HookManager
    from .stats import SessionStats
    from .tool_cache import ToolCache


class _RepeatDetector:
    """Detects when a model is stuck in a repetition loop.

    Checks for:
    1. Same line/sentence appearing 3+ times in accumulated text
    2. Same chunk repeated 5+ times consecutively
    """

    REPEAT_THRESHOLD = 3      # same line appears this many times → stuck
    CHUNK_REPEAT_THRESHOLD = 5  # same chunk in a row this many times → stuck
    CHECK_INTERVAL = 20        # only check every N chunks (perf)

    def __init__(self):
        self._chunk_count = 0
        self._last_chunk: str = ""
        self._same_chunk_run: int = 0
        self._accumulated: list[str] = []

    def feed(self, chunk: str) -> bool:
        """Feed a chunk. Returns True if repetition detected."""
        self._chunk_count += 1
        self._accumulated.append(chunk)

        # Check consecutive identical chunks
        if chunk == self._last_chunk and chunk.strip():
            self._same_chunk_run += 1
            if self._same_chunk_run >= self.CHUNK_REPEAT_THRESHOLD:
                return True
        else:
            self._same_chunk_run = 1
            self._last_chunk = chunk

        # Periodic check for repeated lines in accumulated text
        if self._chunk_count % self.CHECK_INTERVAL == 0:
            return self._check_repeated_lines()

        return False

    def _check_repeated_lines(self) -> bool:
        """Check if any non-trivial line appears 3+ times."""
        full = "".join(self._accumulated)
        # Split on newlines, filter out short/empty lines
        lines = [ln.strip() for ln in full.split("\n") if len(ln.strip()) > 20]
        if not lines:
            return False
        seen: dict[str, int] = {}
        for line in lines:
            seen[line] = seen.get(line, 0) + 1
            if seen[line] >= self.REPEAT_THRESHOLD:
                return True
        return False


class Agent:
    """The agent loop that connects the model to tools."""

    MAX_TOOL_ROUNDS = 25  # Safety limit

    def __init__(self, model: ModelClient, context: Context,
                 tools: ToolRegistry, permissions: PermissionManager,
                 console: Console | None = None,
                 output_prefix: str = "",
                 stats: SessionStats | None = None,
                 on_tool_start: object | None = None,
                 tool_cache: ToolCache | None = None,
                 hooks: HookManager | None = None):
        self.model = model
        self.context = context
        self.tools = tools
        self.permissions = permissions
        self.console = console or Console()
        self.output_prefix = output_prefix
        self.stats = stats
        self.on_tool_start = on_tool_start  # callback(tool_name, args)
        self.tool_cache = tool_cache
        self.hooks = hooks
        self._cancelled = False

    def cancel(self):
        """Signal the agent to stop generation (called from Ctrl+C handler)."""
        self._cancelled = True

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
        self._cancelled = False

        while rounds < self.MAX_TOOL_ROUNDS:
            rounds += 1

            # Collect response from model
            text_parts = []
            tool_calls = []
            # Workers skip the Live display to avoid Rich conflicts
            use_live = not self.output_prefix
            renderer = StreamingRenderer(self.console, live_mode=use_live)
            renderer.start()

            repeat_detector = _RepeatDetector()
            repeat_detected = False

            try:
                async for chunk in self.model.chat(
                    messages=self.context.get_messages(),
                    tools=self.tools.schemas(),
                    stream=True,
                ):
                    # Yield to event loop so signal handlers can fire
                    await asyncio.sleep(0)

                    # Check cancellation flag (set by Ctrl+C handler)
                    if self._cancelled:
                        break

                    if chunk["type"] == "text":
                        text_parts.append(chunk["content"])
                        renderer.feed(chunk["content"])

                        # Check for model stuck in repetition loop
                        if repeat_detector.feed(chunk["content"]):
                            repeat_detected = True
                            break

                    elif chunk["type"] == "tool_call":
                        tool_calls.append(chunk)

                    elif chunk["type"] == "done":
                        pass
            except asyncio.CancelledError:
                self._cancelled = True

            if self._cancelled:
                # Show whatever was generated before the interrupt
                renderer.flush()
                partial = "".join(text_parts)
                if partial.strip():
                    self.context.add_assistant(partial)
                return full_response

            if repeat_detected:
                renderer.stop()
                render_warning(self.console, "Repetition loop detected — stopped generation. Try rephrasing or breaking your request into smaller parts.")
                break

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

            # Separate tool calls into: need-permission (sequential) and auto-allowed
            # Then execute independent auto-allowed calls in parallel
            sequential_tcs = []
            parallel_tcs = []
            for tc in tool_calls:
                tool = self.tools.get(tc["name"])
                if not tool or tc.get("arguments") is None:
                    sequential_tcs.append(tc)
                elif (self.permissions.mode == "trust"
                      or tc["name"] in self.permissions.always_allow
                      or tc["name"] in self.permissions.session_allow
                      or (self.permissions.mode == "auto" and tool.is_read_only)):
                    parallel_tcs.append(tc)
                else:
                    sequential_tcs.append(tc)

            # Execute parallel tool calls concurrently
            parallel_results: list[str] = []
            if len(parallel_tcs) > 1:
                parallel_results = await self._execute_parallel(parallel_tcs)
                for tc, result in zip(parallel_tcs, parallel_results):
                    self.context.add_tool_result(tc["id"], tc["name"], result)
            elif parallel_tcs:
                # Single auto-allowed call — run normally
                sequential_tcs = parallel_tcs + sequential_tcs
                parallel_tcs = []

            # Execute sequential tool calls one by one
            for tc in sequential_tcs:
                await self._execute_single_tool(tc)

            # Display results for parallel calls
            if parallel_tcs and parallel_results:
                for tc, result in zip(parallel_tcs, parallel_results):
                    render_tool_result(self.console, result, tool_name=tc["name"])

            # Continue loop — model will process tool results

        if rounds >= self.MAX_TOOL_ROUNDS:
            render_warning(self.console, "Reached maximum tool rounds")

        return full_response

    async def _execute_single_tool(self, tc: dict):
        """Execute a single tool call with all the checks and display."""
        tool = self.tools.get(tc["name"])
        if not tool:
            result = f"Error: Unknown tool '{tc['name']}'"
            self.context.add_tool_result(tc["id"], tc["name"], result)
            render_error(self.console, f"Unknown tool '{tc['name']}'")
            return

        # Guard: skip tool calls with None/missing arguments
        if tc.get("arguments") is None:
            result = (f"Error: Tool '{tc['name']}' called with no arguments. "
                      "The response may have been truncated due to token limits.")
            self.context.add_tool_result(tc["id"], tc["name"], result)
            render_tool_call(self.console, tc["name"], tc["arguments"])
            render_tool_error(self.console, tc["name"],
                              "Missing arguments — response may have been truncated")
            return

        # Show inline diff preview for edit_file before permission check
        if tc["name"] == "edit_file" and self.permissions.mode != "trust":
            try:
                from .ui.diff import render_inline_diff
                file_path = tc["arguments"].get("file_path", "")
                old_str = tc["arguments"].get("old_string", "")
                new_str = tc["arguments"].get("new_string", "")
                if file_path and old_str:
                    render_inline_diff(self.console, file_path, old_str, new_str)
            except Exception:
                pass

        # Check permission
        if not self.permissions.check(tc["name"], tool.is_read_only,
                                      tc["arguments"]):
            result = "Permission denied by user."
            render_tool_call(self.console, tc["name"], tc["arguments"])
            render_tool_denied(self.console, tc["name"])
            self.context.add_tool_result(tc["id"], tc["name"], result)
            return

        # Notify progress callback
        if self.on_tool_start:
            try:
                self.on_tool_start(tc["name"], tc["arguments"])
            except Exception:
                pass

        # Run pre-hooks
        if self.hooks and self.hooks.has_hooks(f"before_{tc['name']}"):
            hook_ctx = {"tool": tc["name"], **tc["arguments"]}
            await self.hooks.run_hooks(
                f"before_{tc['name']}", hook_ctx, self.console)

        # Display tool call
        render_tool_call(self.console, tc["name"], tc["arguments"])

        # Check cache for read-only tools
        if (self.tool_cache
                and tc["name"] in self.tool_cache.CACHEABLE_TOOLS):
            cached = self.tool_cache.get(tc["name"], tc["arguments"])
            if cached is not None:
                result = cached
                self.context.add_tool_result(tc["id"], tc["name"], result)
                render_tool_result(self.console, result, tool_name=tc["name"])
                return

        # Execute tool (with streaming for bash)
        is_streamed_bash = (
            tc["name"] == "bash"
            and tool.supports_streaming
            and not self.output_prefix
        )
        try:
            if is_streamed_bash:
                connector = "\u23bf"
                def _print_line(line: str):
                    try:
                        t = Text(f"  {connector} ", style="#7b88a1")
                        t.append(line, style="#8899aa")
                        self.console.print(t)
                    except Exception:
                        pass
                result = await tool.execute_streaming(
                    callback=_print_line, **tc["arguments"])
            else:
                result = await tool.execute(**tc["arguments"])
        except Exception as e:
            result = f"Error executing {tc['name']}: {e}"
            # Rich error context — gather extra info on failure
            result += self._gather_error_context(tc["name"], tc["arguments"])

        # Record stats
        if self.stats:
            self.stats.record_tool_call(tc["name"], tc["arguments"])

        # Cache read-only results
        if (self.tool_cache
                and tc["name"] in self.tool_cache.CACHEABLE_TOOLS
                and not result.startswith("Error")):
            self.tool_cache.put(tc["name"], tc["arguments"], result)

        # Invalidate cache on writes
        if (self.tool_cache
                and tc["name"] in self.tool_cache.INVALIDATING_TOOLS):
            path = tc["arguments"].get("file_path", "")
            if path:
                self.tool_cache.invalidate_path(path)

        # Truncate very long results
        if len(result) > 15000:
            result = result[:15000] + "\n\n... (truncated)"
            if is_streamed_bash:
                t = Text("  \u23bf ... (truncated)", style="#7b88a1")
                self.console.print(t)

        self.context.add_tool_result(tc["id"], tc["name"], result)

        # Display result
        if is_streamed_bash:
            pass
        else:
            render_tool_result(self.console, result, tool_name=tc["name"])

        # Run post-hooks
        if self.hooks and self.hooks.has_hooks(f"after_{tc['name']}"):
            hook_ctx = {
                "tool": tc["name"],
                "path": tc["arguments"].get("file_path", ""),
                **tc["arguments"],
            }
            await self.hooks.run_hooks(
                f"after_{tc['name']}", hook_ctx, self.console)

    async def _execute_parallel(self, tool_calls: list[dict]) -> list[str]:
        """Execute multiple independent tool calls concurrently."""

        async def _run_one(tc):
            tool = self.tools.get(tc["name"])
            if not tool:
                return f"Error: Unknown tool '{tc['name']}'"

            # Check cache
            if (self.tool_cache
                    and tc["name"] in self.tool_cache.CACHEABLE_TOOLS):
                cached = self.tool_cache.get(tc["name"], tc["arguments"])
                if cached is not None:
                    return cached

            # Notify progress
            if self.on_tool_start:
                try:
                    self.on_tool_start(tc["name"], tc["arguments"])
                except Exception:
                    pass

            render_tool_call(self.console, tc["name"], tc["arguments"])

            try:
                result = await tool.execute(**tc["arguments"])
            except Exception as e:
                result = f"Error executing {tc['name']}: {e}"

            if self.stats:
                self.stats.record_tool_call(tc["name"], tc["arguments"])

            # Cache
            if (self.tool_cache
                    and tc["name"] in self.tool_cache.CACHEABLE_TOOLS
                    and not result.startswith("Error")):
                self.tool_cache.put(tc["name"], tc["arguments"], result)

            if len(result) > 15000:
                result = result[:15000] + "\n\n... (truncated)"

            return result

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
        return list(results)

    def _gather_error_context(self, tool_name: str, args: dict) -> str:
        """Gather additional context when a tool fails."""
        extra = ""
        try:
            if tool_name in ("edit_file", "write_file"):
                path = args.get("file_path", "")
                if path:
                    import os
                    if os.path.exists(path):
                        extra += f"\nFile exists: {path} ({os.path.getsize(path)} bytes)"
                    else:
                        extra += f"\nFile does not exist: {path}"
                        parent = os.path.dirname(path)
                        if not os.path.isdir(parent):
                            extra += f"\nParent directory does not exist: {parent}"
            elif tool_name == "bash":
                cmd = args.get("command", "")
                if cmd:
                    import shutil
                    binary = cmd.split()[0] if cmd.split() else ""
                    if binary and not shutil.which(binary):
                        extra += f"\nBinary not found in PATH: {binary}"
        except Exception:
            pass
        return extra
