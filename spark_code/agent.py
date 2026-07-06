"""Agent loop — the core of Spark Code.

Sends messages to the model, parses tool calls, executes tools,
feeds results back, and repeats until the model gives a final answer.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
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


CHECKPOINT_DIR = Path.home() / ".spark" / "checkpoints"

# Cap tool results fed back into the context window.
MAX_TOOL_RESULT_CHARS = 15000

# Per-tool result budgets (chars). Tools that gush volume (file reads, greps,
# sub-agent transcripts) get a tighter cap than the 15K default so a single
# noisy result can't dominate the window. Head+tail truncation semantics are
# unchanged; only the limit differs per tool. Config-overridable via
# `tools.result_budgets`; any tool not listed falls back to MAX_TOOL_RESULT_CHARS.
TOOL_RESULT_BUDGETS = {
    "read_file": 10000,
    "bash": 15000,
    "grep": 8000,
    "dispatch_agent": 8000,
}

# The summary request the agent sends the model when auto-compacting. Kept as a
# module constant so /compact and the auto-trigger share one wording.
_COMPACT_SUMMARY_ASK = (
    "Summarize this conversation so far for your own future reference. "
    "Preserve: task goal, key decisions, file paths touched, unresolved items. "
    "Under 300 words."
)


async def generate_compaction_summary(model, context,
                                      instructions: str | None = None) -> str | None:
    """Ask the model for a recap of the conversation, for auto-compaction.

    Sends ONE non-streamed request (the existing model client, ``stream=False``)
    over the current history plus a summary instruction. ``instructions`` (from
    ``/compact <instructions>``) is folded in so the recap can focus where the
    user asked. Returns the summary text, or ``None`` on ANY failure (error
    chunk, exception, empty output) so the caller falls back to the mechanical
    digest and the session never wedges. Deliberately does NOT record usage —
    this ephemeral request must not move the context meter.
    """
    ask = _COMPACT_SUMMARY_ASK
    if instructions:
        ask += f"\n\nFocus especially on: {instructions}"
    request_messages = context.get_messages() + [{"role": "user", "content": ask}]
    parts: list[str] = []
    try:
        stream = model.chat(messages=request_messages, tools=None, stream=False)
        try:
            async for chunk in stream:
                ctype = chunk.get("type")
                if ctype == "text":
                    parts.append(chunk.get("content", ""))
                elif ctype == "error":
                    # A model/transport error — discard any partial and force the
                    # mechanical fallback rather than compacting on half a recap.
                    return None
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
    except Exception:
        return None
    text = "".join(parts).strip()
    return text or None


def _truncate_result(result: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate an over-long tool result, keeping the head AND the tail.

    Bash appends its exit code at the very end, so a head-only cut would hide
    whether a long-output command actually failed. Keeping the tail preserves
    exit codes and trailing error summaries.
    """
    if len(result) <= limit:
        return result
    head = int(limit * 0.8)
    tail = limit - head
    omitted = len(result) - limit
    return (f"{result[:head]}\n\n... ({omitted:,} chars truncated) ...\n\n"
            f"{result[-tail:]}")


def save_checkpoint(path: str, messages: list, cwd: str, provider: str,
                    model: str, round_count: int, files_created: list):
    """Save a session checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "messages": messages,
        "cwd": cwd,
        "provider": provider,
        "model": model,
        "round_count": round_count,
        "timestamp": datetime.now().isoformat(),
        "files_created": files_created,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path: str) -> dict | None:
    """Load a session checkpoint. Returns None if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class _RepeatDetector:
    """Detects when a model is stuck in a repetition loop.

    Checks for:
    1. Same line/sentence appearing 3+ times in accumulated text
    2. Same chunk repeated 5+ times consecutively
    """

    REPEAT_THRESHOLD = 6      # same long line appears this many times → stuck
    MIN_LINE_LEN = 40         # ignore short lines (code boilerplate repeats)
    CHUNK_REPEAT_THRESHOLD = 8  # same chunk in a row this many times → stuck
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
        lines = [ln.strip() for ln in full.split("\n")
                 if len(ln.strip()) > self.MIN_LINE_LEN]
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

    MAX_TOOL_ROUNDS = 75  # Safety limit

    def __init__(self, model: ModelClient, context: Context,
                 tools: ToolRegistry, permissions: PermissionManager,
                 console: Console | None = None,
                 output_prefix: str = "",
                 stats: SessionStats | None = None,
                 on_tool_start: object | None = None,
                 tool_cache: ToolCache | None = None,
                 hooks: HookManager | None = None,
                 on_iteration: object | None = None,
                 result_budgets: dict[str, int] | None = None):
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
        # Optional callback run at the TOP of each loop iteration (a safe
        # message boundary). Workers use this to drain their inbox so
        # inter-agent messages are injected without corrupting a tool exchange.
        self.on_iteration = on_iteration
        self._cancelled = False
        # Guards against the auto-compaction summary request re-triggering
        # compaction (which would recurse). Set only while _maybe_compact runs.
        self._compacting = False
        # Per-tool result char budgets: module defaults merged with any config
        # override (tools.result_budgets). Unlisted tools use the 15K default.
        self._result_budgets = {**TOOL_RESULT_BUDGETS, **(result_budgets or {})}
        # Seed the context's schema reserve from the tool schemas we send on
        # every request (2-4K tokens of JSON that never live in `messages`), so
        # estimate_tokens/auto-compaction budget for that fixed overhead.
        try:
            self.context.schema_reserve_tokens = (
                len(json.dumps(self.tools.schemas())) // 4)
        except Exception:
            pass

    def cancel(self):
        """Signal the agent to stop generation (called from Ctrl+C handler)."""
        self._cancelled = True

    def _budget_for(self, tool_name: str) -> int:
        """Char budget for a tool's result, from the merged per-tool map."""
        return self._result_budgets.get(tool_name, MAX_TOOL_RESULT_CHARS)

    async def _maybe_compact(self) -> str | None:
        """Auto-compact when the real-usage meter says we're low on room.

        Fires only when the context window is a KNOWN positive size AND
        context_left(window) < 0.25 (>75% used). The sharp edge from T1:
        context_left() returns 0.0 for a falsy/unknown window — that must NOT be
        read as "compact now", so an unset/zero window never compacts.

        Generates the summary by asking the model (one non-streamed request);
        on ANY failure falls back to the mechanical digest so the session never
        wedges and no message is lost. The ``_compacting`` guard ensures the
        summary request itself can't re-trigger compaction (recursion).
        """
        if self._compacting:
            return None
        window = self.context.max_tokens
        if not isinstance(window, int) or window <= 0:
            return None
        if self.context.context_left(window) >= 0.25:
            return None

        self._compacting = True
        try:
            summary = await generate_compaction_summary(self.model, self.context)
        finally:
            self._compacting = False
        # summary is None on model failure → compact() builds the mechanical
        # digest instead (never wedges).
        return self.context.compact(summary=summary)

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

            # Safe boundary: drain any queued inter-agent messages into context
            # before building the next request (workers set this hook).
            if self.on_iteration:
                try:
                    self.on_iteration()
                except Exception:
                    pass

            # Keep the context within budget mid-turn (tool results pile up
            # fastest inside a single turn's many rounds). Uses the real-usage
            # meter, a model-generated summary, and the safe, tool-boundary-aware
            # compact.
            await self._maybe_compact()

            # Round-limit nudges are TRANSIENT — injected into this request only,
            # never persisted (else "Finish immediately." poisons every later turn).
            remaining = self.MAX_TOOL_ROUNDS - rounds
            round_note = None
            if remaining == 15:
                round_note = ("You have 15 tool rounds remaining. Begin wrapping "
                              "up — summarize progress and finish current work.")
            elif 0 <= remaining <= 5:
                round_note = f"{remaining} tool rounds remaining. Finish immediately."

            request_messages = self.context.get_messages()
            if round_note:
                request_messages.append({"role": "system", "content": round_note})

            # Collect response from model
            text_parts = []
            tool_calls = []
            # Workers skip the Live display to avoid Rich conflicts
            use_live = not self.output_prefix
            renderer = StreamingRenderer(self.console, live_mode=use_live)
            renderer.start()

            repeat_detector = _RepeatDetector()
            repeat_detected = False
            stream_error = None

            # Hold the async generator so we can close it (aborting the HTTP
            # stream, stopping server-side generation) on cancel.
            chat_stream = self.model.chat(
                messages=request_messages,
                tools=self.tools.schemas(),
                stream=True,
            )
            try:
                async for chunk in chat_stream:
                    # Yield to event loop so signal handlers can fire
                    await asyncio.sleep(0)

                    # Check cancellation flag (set by Ctrl+C handler)
                    if self._cancelled:
                        break

                    if chunk["type"] == "thinking_start":
                        renderer.feed_status("  Thinking...")

                    elif chunk["type"] == "thinking_end":
                        renderer.clear_status()

                    elif chunk["type"] == "error":
                        # Model/transport error surfaced by the client. Record it
                        # to show once the stream ends; don't persist to history.
                        stream_error = chunk.get("content", "Model error")

                    elif chunk["type"] == "text":
                        text_parts.append(chunk["content"])
                        renderer.feed(chunk["content"])

                        # Check for model stuck in repetition loop
                        if repeat_detector.feed(chunk["content"]):
                            repeat_detected = True
                            break

                    elif chunk["type"] == "tool_call":
                        tool_calls.append(chunk)

                    elif chunk["type"] == "done":
                        usage = chunk.get("usage", {})
                        if self.stats and usage:
                            self.stats.record_token_usage(
                                input_tokens=usage.get("prompt_tokens", 0),
                                output_tokens=usage.get("completion_tokens", 0),
                            )
                        prompt_tokens = usage.get("prompt_tokens")
                        if prompt_tokens:
                            # Server-reported (real tokenizer) count — feeds the
                            # toolbar's context-left meter. Some servers (e.g.
                            # certain Ollama configs) omit usage entirely; leave
                            # the estimate-based fallback alone in that case
                            # rather than recording a wrong 0.
                            self.context.record_usage(prompt_tokens)
                        if self.stats and "_speed" in chunk:
                            speed = chunk["_speed"]
                            self.stats.record_generation_speed(
                                speed["tokens"], speed["elapsed"])
            except asyncio.CancelledError:
                self._cancelled = True
            finally:
                # Close the generator so the underlying HTTP stream is aborted
                # and the server stops generating (frees the GPU on a local box).
                await chat_stream.aclose()

            if self._cancelled:
                # Show whatever was generated before the interrupt
                renderer.flush()
                partial = "".join(text_parts)
                if partial.strip():
                    self.context.add_assistant(partial)
                return full_response

            # Finalize — final markdown render and stop live display
            renderer.flush()

            text = "".join(text_parts)

            if text:
                full_response += text

            # A transport/model error ends the turn. Show it, but don't persist
            # the error string into conversation history.
            if stream_error:
                if text.strip():
                    self.context.add_assistant(text)
                render_error(self.console, stream_error)
                break

            if repeat_detected:
                render_warning(self.console, "Repetition loop detected — stopped generation. Try rephrasing or breaking your request into smaller parts.")
                # Preserve whatever the model produced before the loop tripped
                # (don't silently vaporize it from screen and history).
                if text.strip():
                    self.context.add_assistant(text)
                break

            # No tool calls — model is done
            if not tool_calls:
                self.context.add_assistant(text)
                break

            # Process tool calls — keep any narration the model produced
            # alongside the tool calls (don't drop it to content:None).
            self.context.add_assistant_tool_calls(tool_calls, content=text)

            # Partition tool calls. Only READ-ONLY, already-authorized tools run
            # in parallel; anything that mutates state (write/edit/bash) or needs
            # a permission prompt runs sequentially through the full pipeline.
            sequential_tcs = []
            parallel_tcs = []
            for tc in tool_calls:
                tool = self.tools.get(tc["name"])
                authorized = tool is not None and self.permissions.allows_without_prompt(
                    tc["name"], tool.is_read_only)
                if (tool and tc.get("arguments") is not None
                        and tool.is_read_only and authorized):
                    parallel_tcs.append(tc)
                else:
                    sequential_tcs.append(tc)

            # Execute parallel tool calls concurrently
            parallel_results: list[str] = []
            if len(parallel_tcs) > 1:
                parallel_results = await self._execute_parallel(parallel_tcs)
                for tc, result in zip(parallel_tcs, parallel_results):
                    self.context.add_tool_result(
                        tc["id"], tc["name"],
                        _truncate_result(result, self._budget_for(tc["name"])))
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
            try:
                checkpoint_path = str(CHECKPOINT_DIR / "latest.json")
                files = list(self.stats.files_created) if self.stats else []
                save_checkpoint(
                    checkpoint_path,
                    self.context.messages,
                    os.getcwd(),
                    getattr(self.model, "provider", ""),
                    getattr(self.model, "model", ""),
                    rounds,
                    files,
                )
                from spark_code.ui.output import render_info
                render_info(self.console, "Checkpoint saved. Use /continue to resume.")
            except Exception:
                pass

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

        # Detect bash side-effects — override auto-allow
        if tc["name"] == "bash" and tc["arguments"]:
            from .tools.bash import detect_side_effects
            side_effect_warnings = detect_side_effects(tc["arguments"].get("command", ""))
            if side_effect_warnings:
                # Force permission prompt with warnings
                warning_text = "\n".join(f"  ⚠ {w}" for w in side_effect_warnings)
                self.console.print(f"[#ebcb8b]{warning_text}[/#ebcb8b]")

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
            result = self.permissions.last_denial_reason or "Permission denied by user."
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

        # Truncate BEFORE caching so a cache hit returns exactly what a fresh
        # execution would (otherwise a re-read re-injects the full untruncated
        # result and blows the context window).
        truncated = _truncate_result(result, self._budget_for(tc["name"]))
        if is_streamed_bash and len(truncated) != len(result):
            t = Text("  \u23bf ... (truncated)", style="#7b88a1")
            self.console.print(t)
        result = truncated

        # Record stats
        if self.stats:
            self.stats.record_tool_call(tc["name"], tc["arguments"])

        # Track file creation for /clean
        if self.stats and tc["name"] == "write_file" and "Error" not in result:
            file_path = tc["arguments"].get("file_path", "")
            if file_path:
                self.stats.record_file_created(file_path)

        # Cache read-only results
        if (self.tool_cache
                and tc["name"] in self.tool_cache.CACHEABLE_TOOLS
                and not result.startswith("Error")):
            self.tool_cache.put(tc["name"], tc["arguments"], result)

        # Invalidate cache on anything that can mutate the filesystem, including
        # bash (sed/git/formatters), not just write_file/edit_file.
        self._invalidate_cache_for(tc)

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
        """Execute multiple independent, read-only, already-authorized tool
        calls concurrently. Mutating tools never reach this path (they are
        routed sequentially), so no cache-invalidation or side-effect handling
        is needed here — but read hooks still run."""

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

            # Pre-hooks
            if self.hooks and self.hooks.has_hooks(f"before_{tc['name']}"):
                await self.hooks.run_hooks(
                    f"before_{tc['name']}", {"tool": tc["name"], **tc["arguments"]},
                    self.console)

            render_tool_call(self.console, tc["name"], tc["arguments"])

            try:
                result = await tool.execute(**tc["arguments"])
            except Exception as e:
                result = f"Error executing {tc['name']}: {e}"

            if self.stats:
                self.stats.record_tool_call(tc["name"], tc["arguments"])

            # Truncate before caching so hits match fresh executions.
            result = _truncate_result(result, self._budget_for(tc["name"]))
            if (self.tool_cache
                    and tc["name"] in self.tool_cache.CACHEABLE_TOOLS
                    and not result.startswith("Error")):
                self.tool_cache.put(tc["name"], tc["arguments"], result)

            # Post-hooks
            if self.hooks and self.hooks.has_hooks(f"after_{tc['name']}"):
                await self.hooks.run_hooks(
                    f"after_{tc['name']}",
                    {"tool": tc["name"], "path": tc["arguments"].get("file_path", ""),
                     **tc["arguments"]},
                    self.console)

            return result

        results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])
        return list(results)

    # Bash verbs that can mutate the filesystem (and thus stale the read caches).
    _BASH_MUTATORS = (">", ">>", "sed -i", " mv ", " cp ", " rm ", "rm -",
                      "mkdir", "touch ", "tee ", "git checkout", "git restore",
                      "git reset", "git apply", "git stash", "git pull",
                      "git merge", "npm install", "pip install", "black ",
                      "ruff ", "prettier", "gofmt", "dd ", "truncate ")

    def _invalidate_cache_for(self, tc: dict):
        """Invalidate stale read caches after a tool that may mutate files."""
        if not self.tool_cache:
            return
        name = tc["name"]
        args = tc.get("arguments") or {}
        if name in self.tool_cache.INVALIDATING_TOOLS:
            path = args.get("file_path", "")
            if path:
                self.tool_cache.invalidate_path(path)
        elif name == "bash":
            cmd = args.get("command", "")
            if any(tok in cmd for tok in self._BASH_MUTATORS):
                # bash can touch anything — clear the whole read cache.
                self.tool_cache.invalidate_all()

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
