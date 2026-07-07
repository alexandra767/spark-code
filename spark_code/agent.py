"""Agent loop — the core of Spark Code.

Sends messages to the model, parses tool calls, executes tools,
feeds results back, and repeats until the model gives a final answer.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

from .context import Context
from .mcp.client import MCP_IMAGE_SENTINEL_PREFIX, is_spark_mcp_temp_file
from .model import ModelClient
from .permissions import PermissionManager
from .plan_mode import PLAN_DENIAL, PLAN_NUDGE, PlanState
from .routing import call_with_fallback
from .skills.base import SkillRegistry
from .tools.base import ToolRegistry
from .tools.simulator import SCREENSHOT_SENTINEL_PREFIX
from .ui.input import RESERVED_COMMAND_NAMES
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
    # Phase 4 Task 3: multiple path:line — snippet blocks per call; capped
    # well under the 15K default so a broad query can't dominate the window.
    "code_search": 6000,
}

# Phase 4 Task 5: browser-automation MCP tools (Playwright MCP names them all
# `browser_*`) return very large accessibility-tree / DOM dumps — a single
# `browser_snapshot` of a content page measured ~300 KB in the first real-server
# trial. Cap them together, well under the 15K default, so one snapshot can't
# swamp the context window. Applied by `_budget_for` to any MCP tool whose bare
# name (after the `server__` prefix) starts with `browser_`, regardless of the
# server name the user picks in `mcp_servers`. A user can still raise/lower a
# specific tool via `tools.result_budgets` (an exact `server__tool` key wins).
# See docs/browser-mcp.md.
BROWSER_MCP_RESULT_BUDGET = 8000

# Verification habit (Task 6): tool names whose successful execution "dirties"
# the turn (per the plan's wording — bash-invoked writes/formatters don't
# count, only the two dedicated file-mutation tools).
_WRITE_TOOL_NAMES = ("write_file", "edit_file")

# Transient nudge injected (like the round-limit notes above) when the model
# produced a final answer after changing files this turn without running the
# project's tests. Never persisted to context history — see _agent_loop.
VERIFY_NUDGE_TEMPLATE = (
    "You changed files but have not run the project's tests (`{cmd}`). "
    "Run them before concluding, or state explicitly why not."
)

# 80B-friendly skills (Task 8): matches a FINAL answer that consists SOLELY
# of a slash-skill invocation — "/<skill-name> [args]" — once surrounding
# whitespace is stripped. Deliberately requires the caller to first confirm
# there's no internal newline (i.e. no other content lines) before matching;
# see Agent._maybe_expand_skill_reply. Skill names may contain the
# plugin-namespacing ":" (e.g. "superpowers:test-driven-development").
_SLASH_LINE_RE = re.compile(r"^/([A-Za-z0-9_:.\-]+)(?:[ \t]+(.*))?$")


def _bash_ran_test_command(command: str, test_command: str) -> bool:
    """True iff ``command`` invokes ``test_command``, matched at word
    boundaries — a plain substring check would let e.g. test_command
    "pytest" false-positive on an unrelated command like "ls mypytest_dir/".
    """
    return re.search(r"\b" + re.escape(test_command) + r"\b", command) is not None


# How many recent messages compaction keeps verbatim (the auto-trigger and
# Context.compact's default must agree).
_COMPACT_KEEP_RECENT = 6

# Anti-thrash guards (review round 1): only compact when it can plausibly help.
# Skip when the removable slice estimates under this fraction of the window (a
# giant STATIC prompt can park usage in the 75-90% band while compaction frees
# ~nothing — without this, a full-context summary request fired EVERY round);
# and never auto-compact twice within this many rounds.
_COMPACT_MIN_GAIN_FRACTION = 0.05
_COMPACT_COOLDOWN_ROUNDS = 5

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
        # NOTE: this request's tokens DO count toward the ModelClient's lifetime
        # /stats totals — intentional: those tokens were really spent.
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
                 result_budgets: dict[str, int] | None = None,
                 plan_state: PlanState | None = None,
                 test_command: str | None = None,
                 skills: SkillRegistry | None = None,
                 editor: str | None = None,
                 diff_in_editor: bool = False,
                 utility_model: ModelClient | None = None,
                 agent_defs: dict | None = None):
        self.model = model
        # Phase 4 Task 2: an optional cheaper model for the auto-compaction
        # summary request. Resolved and use_for-gated ONCE by the caller
        # (get_utility_client + routing.enabled_for, see cli.py) — the Agent
        # itself just holds whatever it was handed and falls back to
        # ``self.model`` per call via routing.call_with_fallback if this is
        # None or the utility call fails. Never rebuilt/closed by the Agent —
        # its lifecycle is the caller's (a primary /model switch reassigns
        # ``self.model`` only, this is untouched).
        self.utility_model = utility_model
        self.context = context
        self.tools = tools
        self.permissions = permissions
        # Shared plan-mode flag (same object the CLI holds). None in contexts
        # that never use plan mode (one-shot, sub-agents) → enforcement off.
        self.plan_state = plan_state
        # Verification habit (Task 6): the project's test command, detected
        # once at construction (detect_test_command — instructions override,
        # else project-type markers). None → feature silently off (nothing
        # detectable to nudge about). Per-turn tracking flags are reset at the
        # top of _agent_loop (see there for why that's the right boundary).
        self._test_command = test_command
        self._verify_dirty = False
        self._verify_tests_run = False
        self._verify_nudged = False
        self._verify_nudge_pending = False
        # Paths successfully written/edited THIS TURN (auto-review scoping,
        # Task 7) — reset per turn alongside the flags above.
        self._verify_written_paths: list[str] = []
        # Headless/machine-contract exposure (Phase 4 Task 1): how the turn
        # ended, read by spark_code.headless (via cli.py's _one_shot) AFTER
        # run() returns so a script can build an accurate JSON result without
        # the Agent knowing anything about JSON. Reset per turn alongside the
        # verification flags above; pure bookkeeping — never gates behavior.
        self._last_rounds: int = 0
        self._hit_max_rounds: bool = False
        self._last_stream_error: str | None = None
        # True once the model ends the turn with a real final answer (the
        # no-tool-calls break). Distinguishes answered-on-the-last-round from
        # loop-exhausted: rounds increments at the top of the loop, so the
        # post-loop `rounds >= MAX_TOOL_ROUNDS` check alone would misreport a
        # valid final answer given exactly on the cap round as a cap-out.
        self._final_answer_given: bool = False
        # 80B-friendly skills (Task 8): the registry used to expand a model
        # FINAL answer that is solely a "/<skill-name> [args]" line. None
        # (one-shot mode, sub-agents) → feature silently off, same pattern as
        # test_command above. Guard resets per turn in _agent_loop.
        self.skills = skills
        self._skill_auto_expanded = False
        # Phase 5 Task 4: custom subagent definitions (.spark/agents/*.md +
        # ~/.spark/agents/*.md), keyed by name — read-only lookup used by
        # ``_plan_denies`` to extend the explore/reviewer plan-mode allowance
        # to a custom def whose own base_type is read-only. {} (the default)
        # means no custom types are known, matching pre-Task-4 behavior.
        self._agent_defs = agent_defs or {}
        # Phase 4 Task 4: screenshot image turns buffered during a round and
        # flushed AFTER the round's full tool-result run is recorded (see
        # _store_tool_result / _flush_pending_images). Injecting them inline
        # would splice a user message into the middle of a multi-tool round's
        # tool-result run and corrupt the transcript. Each entry is
        # (text, image_b64, mime).
        self._pending_image_injections: list[tuple[str, str, str]] = []
        self.console = console or Console()
        # Resolved host editor (Phase 3 Task 4 — "cursor"/"code"/"xed", or
        # None). Threaded into render_tool_call so rendered file paths get
        # an OSC 8 hyperlink; None (the default) keeps every existing
        # caller's output byte-identical to before this feature existed.
        self.editor = editor
        # Diff-in-editor (Phase 3 Task 5): config read ONCE here at
        # construction (same pattern as ``editor`` above), not re-read per
        # tool call. Only takes effect when ``self.editor`` is "cursor" or
        # "code" — see the edit_file preview in _execute_single_tool.
        self.diff_in_editor = diff_in_editor
        # One-time-per-session note for xed users when diff_in_editor is
        # enabled but xed has no `--diff` flag to hand off to. Threaded
        # through maybe_preview_diff_in_editor, which returns the flag's
        # next value so this stays the single source of truth.
        self._diff_editor_xed_noted = False
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
        # Persistent "this session was interrupted at least once" flag
        # (Phase 4 Task 6). Unlike self._cancelled — which is reset at the
        # top of every turn — this is set in the cancel branch and NEVER
        # reset within a session, so an earlier interrupted turn can't be
        # laundered into a "clean" corpus export by a later clean turn.
        self._session_interrupted = False
        # Guards against the auto-compaction summary request re-triggering
        # compaction (which would recurse). Set only while _maybe_compact runs.
        self._compacting = False
        # Anti-thrash cooldown state: _maybe_compact calls are the round clock
        # (one call per round boundary); a successful compact stamps the round
        # so another can't fire within _COMPACT_COOLDOWN_ROUNDS.
        self._compact_round_index = 0
        self._last_compact_round: int | None = None
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
        """Char budget for a tool's result, from the merged per-tool map.

        Exact tool name wins (this is what a `tools.result_budgets` config
        override targets). MCP tools are named ``server__tool`` where the
        ``server`` prefix is user-chosen, so an exact code default isn't
        practical; for those, fall back to the bare tool name after the ``__``
        separator, then to a shared cap for browser-automation tools (Playwright
        MCP's ``browser_*`` a11y/DOM dumps run into the hundreds of KB — see
        docs/browser-mcp.md).
        """
        if tool_name in self._result_budgets:
            return self._result_budgets[tool_name]
        if "__" in tool_name:
            bare = tool_name.split("__", 1)[1]
            if bare in self._result_budgets:
                return self._result_budgets[bare]
            if bare.startswith("browser_"):
                return BROWSER_MCP_RESULT_BUDGET
        return MAX_TOOL_RESULT_CHARS

    def _plan_denies(self, tc: dict, tool) -> bool:
        """True iff plan mode is active AND this tool call is a write action.

        The single enforcement predicate, called from BOTH the sequential path
        (``_execute_single_tool``) and the parallel path (``_execute_parallel``)
        so the two can never drift apart — the same failure mode the Phase-1
        ``allows_without_prompt`` extraction was created to prevent.

        Rules (in order):
          * plan mode off / no state → never denies;
          * ``exit_plan_mode`` → always allowed (it IS the plan gate);
          * any ``is_read_only`` tool → allowed (research runs free);
          * ``dispatch_agent`` → allowed ONLY for the read-only sub-agent types
            ``explore``/``reviewer`` (built-in) or a CUSTOM agent type
            (``.spark/agents/*.md``, Phase 5 Task 4) whose own ``base_type``
            is read-only, and blocked for ``implementer`` or a custom
            implementer-class def. The tool's ``is_read_only`` is
            conservatively False (a single instance can't vary it per call),
            so the generic rule would block all three — this special-case
            inspects the CALL's ``agent_type`` (Task-4 carry-forward);
          * everything else that isn't read-only → denied.
        """
        if self.plan_state is None or not self.plan_state.active:
            return False
        name = tc.get("name")
        if name == "exit_plan_mode":
            return False
        if tool is not None and tool.is_read_only:
            return False
        if name == "dispatch_agent":
            # _parse_tool_arguments doesn't guarantee a dict (a bare-scalar
            # payload like the json string '"explore"' parses fine) — anything
            # non-dict can't prove a read-only agent_type, so it stays blocked
            # (fail closed) instead of raising on .get().
            args = tc.get("arguments")
            if isinstance(args, dict):
                agent_type = args.get("agent_type")
                if agent_type in ("explore", "reviewer"):
                    return False
                custom = self._agent_defs.get(agent_type)
                if custom is not None and custom.is_read_only:
                    return False
        return True

    def _should_nudge_verification(self) -> bool:
        """True iff the model just gave a final answer this round having
        changed files this turn without running the project's tests.

        Guards (all must hold):
          * a test command was detectable at all (``None`` → feature off);
          * a write/edit tool succeeded THIS TURN (``_verify_dirty``);
          * no bash command containing that test command ran THIS TURN
            (``_verify_tests_run``);
          * the nudge hasn't already fired this turn (``_verify_nudged`` —
            once means once);
          * plan mode isn't active. Redundant in practice — writes are denied
            before they can succeed while plan mode is on, so ``_verify_dirty``
            can't become True there — but kept explicit (defense in depth,
            same reasoning as ``_plan_denies`` being checked on both tool
            paths) in case that invariant ever changes.
        """
        if self._verify_nudged:
            return False
        if not self._test_command:
            return False
        if not self._verify_dirty or self._verify_tests_run:
            return False
        if self.plan_state is not None and self.plan_state.active:
            return False
        return True

    def _maybe_expand_skill_reply(self, text: str) -> str | None:
        """If the model's FINAL answer (no tool calls) consists SOLELY of a
        ``/<skill-name> [args]`` line, return that skill's expanded prompt —
        the caller feeds it back in as the next user message and continues
        the turn, exactly like the CLI's handle_slash_command fallback but
        triggered by the model instead of the user typing it.

        Returns None (leaving ``text`` to stand as the real final answer) for
        every case that ISN'T that exact shape:
          * no skill registry configured (one-shot mode, sub-agents);
          * already expanded once this turn (``_skill_auto_expanded`` —
            one hop max, so a model that keeps re-emitting a slash line can't
            loop forever);
          * the answer has more than one line of content once surrounding
            whitespace is stripped — a reply that merely CONTAINS a slash
            line among other prose must not trigger this;
          * the single line isn't shaped like a slash command at all;
          * the name collides with a real CLI command
            (``RESERVED_COMMAND_NAMES``) — at the CLI that name resolves to
            the COMMAND, not the skill (e.g. /review is the multi-agent
            swarm, not the lesser builtin `review` skill), and the model
            surface must not resolve the same name to something else;
          * the name doesn't match a registered skill — treated as normal
            text, same as an unknown command at the CLI.
        """
        if self.skills is None or self._skill_auto_expanded:
            return None
        stripped = text.strip()
        if not stripped or "\n" in stripped:
            return None
        match = _SLASH_LINE_RE.match(stripped)
        if not match:
            return None
        name = match.group(1).lower()
        if name in RESERVED_COMMAND_NAMES:
            return None
        args = (match.group(2) or "").strip()
        skill = self.skills.get(name)
        if skill is None:
            return None
        return skill.get_prompt(args)

    async def _maybe_compact(self) -> str | None:
        """Auto-compact when the real-usage meter says we're low on room AND
        compaction can plausibly help.

        Fires only when the context window is a KNOWN positive size AND
        context_left(window) < 0.25 (>75% used). The sharp edge from T1:
        context_left() returns 0.0 for a falsy/unknown window — that must NOT be
        read as "compact now", so an unset/zero window never compacts.

        Anti-thrash guards (review round 1): (a) the removable slice must
        estimate at least 5% of the window — a giant STATIC prompt (big pinned
        files) can park usage in the 75-90% band while compaction frees
        ~nothing, which previously fired a full-context summary request EVERY
        round; (b) a 5-round cooldown after any successful compact.

        Generates the summary by asking the model (one non-streamed request);
        on ANY failure falls back to the mechanical digest so the session never
        wedges and no message is lost. The ``_compacting`` guard ensures the
        summary request itself can't re-trigger compaction (recursion).
        """
        if self._compacting:
            return None
        # One _maybe_compact call per round boundary — this IS the round clock.
        self._compact_round_index += 1
        window = self.context.max_tokens
        if not isinstance(window, int) or window <= 0:
            return None
        if self.context.context_left(window) >= 0.25:
            return None
        # Cooldown: never auto-compact twice within _COMPACT_COOLDOWN_ROUNDS.
        if (self._last_compact_round is not None
                and (self._compact_round_index - self._last_compact_round)
                < _COMPACT_COOLDOWN_ROUNDS):
            return None
        # Gain guard: skip entirely (no summary request, no compact) unless the
        # removable slice is worth at least 5% of the window. Also covers the
        # nothing-to-compact case (slice estimate 0 when len <= keep_recent).
        if (self.context.estimate_compactable_tokens(_COMPACT_KEEP_RECENT)
                < window * _COMPACT_MIN_GAIN_FRACTION):
            return None

        self._compacting = True
        try:
            # Phase 4 Task 2: prefer the utility model (if configured and
            # use_for-gated in) for the summary request; a raise/timeout/None
            # result there silently retries against the primary model — this
            # ONLY changes which model gets asked, not the existing "None on
            # any failure → mechanical digest" contract callers rely on.
            summary = await call_with_fallback(
                self.utility_model, self.model,
                lambda m: generate_compaction_summary(m, self.context))
        finally:
            self._compacting = False
        # summary is None on model failure → compact() builds the mechanical
        # digest instead (never wedges).
        result = self.context.compact(keep_recent=_COMPACT_KEEP_RECENT,
                                      summary=summary)
        if result is not None:
            self._last_compact_round = self._compact_round_index
        return result

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
        # Verification-habit flags reset PER USER TURN — _agent_loop runs once
        # per run()/run_without_user_add() call, i.e. once per turn, so this is
        # the natural reset boundary (mirrors self._cancelled just above).
        self._verify_dirty = False
        self._verify_tests_run = False
        self._verify_nudged = False
        self._verify_nudge_pending = False
        self._verify_written_paths = []
        # Skill auto-expand guard (Task 8) — one hop max per turn, same reset
        # boundary as the verification flags above.
        self._skill_auto_expanded = False
        # Headless exposure (Phase 4 Task 1) — same per-turn reset boundary.
        self._last_rounds = 0
        self._hit_max_rounds = False
        self._last_stream_error = None
        self._final_answer_given = False
        # Defensive: images buffer per-round and always flush in-round, but
        # reset here too so the "always flushed" invariant is guaranteed.
        self._pending_image_injections = []

        while rounds < self.MAX_TOOL_ROUNDS:
            rounds += 1
            self._last_rounds = rounds

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
            # Plan-mode nudge is TRANSIENT too — injected into this request only
            # while plan mode is active, never persisted (so it vanishes the
            # moment the plan is approved / mode changes, same as round notes).
            if self.plan_state is not None and self.plan_state.active:
                request_messages.append({"role": "system", "content": PLAN_NUDGE})
            # Verification nudge is TRANSIENT too — injected into exactly ONE
            # request (the round right after it fires, set by the "no tool
            # calls" branch below) and consumed immediately so it can never
            # persist into history or repeat beyond that single extra chance.
            if self._verify_nudge_pending:
                request_messages.append({"role": "system", "content":
                    VERIFY_NUDGE_TEMPLATE.format(cmd=self._test_command)})
                self._verify_nudge_pending = False

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
                # Persistent session-level mark (Phase 4 Task 6): this turn
                # was interrupted, so the session must never export to the
                # training corpus as a clean completion — even if a later
                # turn finishes cleanly. Set here at the single choke point
                # every cancellation path (Ctrl+C flag + CancelledError)
                # converges on before returning.
                self._session_interrupted = True
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
                self._last_stream_error = stream_error
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
                skill_prompt = self._maybe_expand_skill_reply(text)
                if skill_prompt is not None:
                    # The model asked for a skill by name instead of doing the
                    # work directly — expand it into the skill's real prompt
                    # and feed that back in as the next user message, exactly
                    # like a user typing the slash command themselves.
                    # _skill_auto_expanded guards against firing more than
                    # once per turn (a model that keeps re-emitting a slash
                    # line can't loop forever on this).
                    self._skill_auto_expanded = True
                    self.context.add_user(skill_prompt)
                    continue
                if self._should_nudge_verification():
                    # Give the model ONE more round to run tests (or explain
                    # why not) before letting the turn end. _verify_nudged
                    # guards against firing more than once per turn.
                    self._verify_nudged = True
                    self._verify_nudge_pending = True
                    continue
                self._final_answer_given = True
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
                # exit_plan_mode is read-only but runs a BLOCKING interactive
                # approval prompt — it must never run under _execute_parallel's
                # asyncio.gather (concurrent Prompt.ask), so force it sequential.
                if (tool and tc.get("arguments") is not None
                        and tool.is_read_only and authorized
                        and tc["name"] != "exit_plan_mode"):
                    parallel_tcs.append(tc)
                else:
                    sequential_tcs.append(tc)

            # Execute parallel tool calls concurrently
            parallel_results: list[str] = []
            if len(parallel_tcs) > 1:
                parallel_results = await self._execute_parallel(parallel_tcs)
                for i, tc in enumerate(parallel_tcs):
                    truncated = _truncate_result(
                        parallel_results[i], self._budget_for(tc["name"]))
                    # Reassign so the display loop below (which reuses this
                    # same list) renders the friendly text, not a screenshot
                    # sentinel.
                    parallel_results[i] = self._store_tool_result(tc, truncated)
            elif parallel_tcs:
                # Single auto-allowed call — run normally
                sequential_tcs = parallel_tcs + sequential_tcs
                parallel_tcs = []

            # Execute sequential tool calls one by one
            for tc in sequential_tcs:
                await self._execute_single_tool(tc)

            # Flush any buffered screenshot image turns NOW — after every tool
            # result for this round (parallel + sequential) is recorded — so
            # the image lands after a complete, valid tool-result run rather
            # than spliced into the middle of it. See _flush_pending_images.
            self._flush_pending_images()

            # Display results for parallel calls
            if parallel_tcs and parallel_results:
                for tc, result in zip(parallel_tcs, parallel_results):
                    render_tool_result(self.console, result, tool_name=tc["name"],
                                      **self._todo_render_kwargs(tc["name"]))

            # Continue loop — model will process tool results

        # Cap-out is only a cap-out when the loop was EXHAUSTED — a final
        # answer given exactly on the last allowed round is a normal
        # completion (no warning, no checkpoint, exit 0 in headless).
        if rounds >= self.MAX_TOOL_ROUNDS and not self._final_answer_given:
            self._hit_max_rounds = True
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

    def _todo_render_kwargs(self, tool_name: str) -> dict:
        """Extra kwargs for render_tool_result when ``tool_name`` is
        ``todo_write`` — hands the CURRENT (post-update) items to the
        renderer so it draws the live checklist panel instead of a generic
        confirmation line. Empty for every other tool."""
        if tool_name != "todo_write":
            return {}
        tool = self.tools.get(tool_name)
        items = getattr(tool, "items", None)
        return {"todo_items": items} if items is not None else {}

    def _store_tool_result(self, tc: dict, result: str) -> str:
        """Store a tool's result into context; returns the text used for
        storage/display (normally just ``result`` unchanged).

        Special-cases ``simulator_screenshot``'s image sentinel (Phase 4
        Task 4, see tools/simulator.py's module docstring): tool results are
        plain strings subject to per-tool truncation, so the actual PNG
        bytes never travel through ``result`` — the tool instead returns
        ``SCREENSHOT_SENTINEL_PREFIX`` + the saved file's path. Detected
        here, this reads the file itself and BUFFERS an
        ``add_user_with_image``-shaped multimodal turn (context.py:207) —
        the same mechanism dragged-and-dropped images use (cli.py's
        ``_is_image_drop``/``/image`` handling). The tool_call id is answered
        immediately with a short plain-text confirmation (role:"tool"
        messages must stay text-only for OpenAI-compatible APIs), but the
        image turn is NOT emitted here — it is flushed by
        ``_flush_pending_images`` only after every tool result for the round
        is recorded, so a screenshot called alongside other tools can't
        splice a user message into the middle of the tool-result run.

        Also handles MCP image content (Phase 5 Task 5, see mcp/client.py's
        ``MCP_IMAGE_SENTINEL_PREFIX`` docstring) via ``_resolve_mcp_images``
        — same buffer-now/flush-after-round mechanism, generalized to
        possibly-multiple images embedded anywhere in an MCP tool's result
        text (unlike simulator_screenshot's single whole-string sentinel).
        """
        if tc["name"] == "simulator_screenshot" and result.startswith(
                SCREENSHOT_SENTINEL_PREFIX):
            path = result[len(SCREENSHOT_SENTINEL_PREFIX):]
            try:
                with open(path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
            except OSError as e:
                text = (f"Error: screenshot saved to {path} but could not be "
                        f"read back for the model ({e})")
                self.context.add_tool_result(tc["id"], tc["name"], text)
                return text
            text = f"Screenshot captured — see image below. (saved to {path})"
            self.context.add_tool_result(tc["id"], tc["name"], text)
            self._pending_image_injections.append(
                ("Simulator screenshot:", image_b64, "image/png"))
            return text
        if MCP_IMAGE_SENTINEL_PREFIX in result:
            result = self._resolve_mcp_images(tc["name"], result)
        self.context.add_tool_result(tc["id"], tc["name"], result)
        return result

    def _resolve_mcp_images(self, tool_name: str, result: str) -> str:
        """Resolve MCP image sentinels (Phase 5 Task 5) into either buffered
        image turns or the original inert placeholder text, gated on
        ``model.supports_vision`` exactly like the simulator_screenshot path
        above — the only difference is WHERE the gate is checked: MCPTool
        has no model reference of its own (unlike SimulatorScreenshotTool,
        which is constructed with one), so it always emits a sentinel for
        every image and this method — which has ``self.model`` for free —
        makes the vision/no-vision call instead.

        An MCP tool result can mix plain text with one sentinel line per
        image (e.g. a multi-screenshot call): only sentinel-prefixed lines
        are rewritten here, everything else passes through unchanged. A
        vision-capable model gets each image read back + base64-encoded and
        appended to ``_pending_image_injections`` (flushed by
        ``_flush_pending_images`` after the round, same as
        simulator_screenshot); a non-vision model gets back the original
        ``[Image: <mime>]`` placeholder — byte-identical to this tool's
        result before this task existed.

        SECURITY (Phase 5 Task 5 review — the Critical gate): the sentinel
        prefix is a public constant and MCP text content is attacker-
        controlled (a hostile page's accessibility tree via
        ``browser_snapshot``, a malicious MCP server, or a forged line in
        ANY tool's output — this scan is intentionally not tool-scoped, so
        tool-name scoping alone would not be enough). Every candidate path is
        therefore run through ``is_spark_mcp_temp_file`` FIRST: a path that
        isn't a Spark-created temp file is refused outright — never opened,
        never unlinked — and degrades to the ``[Image: <mime>]`` placeholder.
        That is what stops a forged ``__SPARK_MCP_IMAGE__:image/png:/etc/passwd``
        line from exfiltrating that file's bytes to the model.

        A confined (Spark-created) temp file is unlinked as soon as it has
        been consumed — read + buffered for a vision model, or skipped for a
        non-vision model — so MCP image temp files don't accumulate on disk
        (Fix 2). A refused path is NEVER unlinked (it isn't ours to touch).
        """
        vision = bool(getattr(self.model, "supports_vision", False))
        out_lines = []
        for line in result.split("\n"):
            if not line.startswith(MCP_IMAGE_SENTINEL_PREFIX):
                out_lines.append(line)
                continue
            mime, _, path = line[len(MCP_IMAGE_SENTINEL_PREFIX):].partition(":")
            if not is_spark_mcp_temp_file(path):
                # Forged / attacker-controlled path — refuse to read it. Emit
                # the same inert placeholder a non-vision model would get; do
                # NOT unlink (the file, if any, is not one Spark created).
                out_lines.append(f"[Image: {mime}]")
                continue
            if not vision:
                out_lines.append(f"[Image: {mime}]")
                self._unlink_mcp_image(path)
                continue
            try:
                with open(path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
            except OSError as e:
                out_lines.append(
                    f"Error: image could not be read back for the model ({e})")
                self._unlink_mcp_image(path)
                continue
            out_lines.append("Image captured — see image below.")
            self._pending_image_injections.append(
                (f"Image from {tool_name}:", image_b64, mime))
            self._unlink_mcp_image(path)
        return "\n".join(out_lines)

    @staticmethod
    def _unlink_mcp_image(path: str) -> None:
        """Best-effort removal of a consumed MCP image temp file. Only ever
        called on paths already confirmed by ``is_spark_mcp_temp_file`` — a
        forged path is never passed here — so this cannot delete an
        attacker-named file. Silent on failure (already gone, races, etc.)."""
        try:
            os.remove(path)
        except OSError:
            pass

    def _flush_pending_images(self):
        """Emit any buffered screenshot image turns, AFTER the round's full
        tool-result run is recorded.

        Image content can't live in a role:"tool" message (it must be a user
        multimodal message), so a screenshot is followed by an injected user
        turn. Emitting that turn the instant the screenshot's own result is
        stored would splice a user message INTO the middle of a multi-tool
        round's tool-result run — the next request's
        ``sanitize_orphaned_tool_calls`` would then see a broken run, backfill
        a synthetic "[interrupted]" result, and strand the round's remaining
        real tool results in an invalid position (a 400 on strict servers).
        Buffering during the round and flushing once here — after every tool
        result for the round is in — keeps the tool-result run contiguous and
        valid, with the image(s) landing immediately after it. Idempotent: a
        no-op when nothing is buffered, and clears the buffer so it never
        leaks into a later round."""
        if not self._pending_image_injections:
            return
        for text, image_b64, mime in self._pending_image_injections:
            self.context.add_user_with_image(text, image_b64, mime)
        self._pending_image_injections = []

    async def _run_hooks_safe(self, event: str, context: dict) -> None:
        """Run hooks for an event without ever letting a hook — a failing
        exit code, a timeout, or an outright bug in the hook plumbing —
        block or crash the tool call / agent loop (Task 6 hardening).

        HookManager.run_hooks already reports a nonzero exit or a timeout as
        a normal (False, message) result rather than raising, but this is
        defense in depth for anything unexpected (e.g. a hostile/odd context
        value tripping something in hook matching or console rendering):
        any exception is swallowed and surfaced only as a dim console note,
        never propagated.
        """
        try:
            await self.hooks.run_hooks(event, context, self.console)
        except Exception as e:
            try:
                self.console.print(Text(f"  hook error: {e}", style="dim"))
            except Exception:
                pass

    async def _execute_single_tool(self, tc: dict):
        """Execute a single tool call with all the checks and display."""
        tool = self.tools.get(tc["name"])
        if not tool:
            result = f"Error: Unknown tool '{tc['name']}'"
            self.context.add_tool_result(tc["id"], tc["name"], result)
            render_error(self.console, f"Unknown tool '{tc['name']}'")
            return

        # Plan mode: block write actions before any side effect (bash detect,
        # edit diff, permission prompt, execution). Shared with the parallel
        # path via _plan_denies so the two can't drift.
        if self._plan_denies(tc, tool):
            render_tool_call(self.console, tc["name"], tc.get("arguments") or {},
                            editor=self.editor)
            render_tool_denied(self.console, tc["name"])
            self.context.add_tool_result(tc["id"], tc["name"], PLAN_DENIAL)
            return

        # Guard: skip tool calls with None/missing arguments
        if tc.get("arguments") is None:
            result = (f"Error: Tool '{tc['name']}' called with no arguments. "
                      "The response may have been truncated due to token limits.")
            self.context.add_tool_result(tc["id"], tc["name"], result)
            render_tool_call(self.console, tc["name"], tc["arguments"], editor=self.editor)
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
                    # Editor-side diff view (Phase 3 Task 5, opt-in via
                    # ui.diff_in_editor). ADDITIVE to the inline diff above,
                    # never a replacement — it still rendered unconditionally.
                    from .editor import maybe_preview_diff_in_editor
                    self._diff_editor_xed_noted = maybe_preview_diff_in_editor(
                        self.console, self.editor, self.diff_in_editor,
                        file_path, old_str, new_str, self._diff_editor_xed_noted,
                    )
            except Exception:
                pass

        # Check permission
        if not self.permissions.check(tc["name"], tool.is_read_only,
                                      tc["arguments"]):
            result = self.permissions.last_denial_reason or "Permission denied by user."
            render_tool_call(self.console, tc["name"], tc["arguments"], editor=self.editor)
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
            await self._run_hooks_safe(f"before_{tc['name']}", hook_ctx)

        # Display tool call
        render_tool_call(self.console, tc["name"], tc["arguments"], editor=self.editor)

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

        # Verification habit (Task 6): a successful write/edit "dirties" the
        # turn; a bash command that contains the detected test command marks
        # tests as run — checked AFTER execute() so a permission denial or a
        # missing-arguments guard (both `return` earlier) never counts as
        # either. A failing test run still counts as "run" — the nudge is
        # about running tests, not passing them.
        if tc["name"] in _WRITE_TOOL_NAMES and "Error" not in result:
            self._verify_dirty = True
            # Collect the written path for auto-review's turn scoping (Task 7).
            written = tc["arguments"].get("file_path", "")
            if written and written not in self._verify_written_paths:
                self._verify_written_paths.append(written)
        elif tc["name"] == "bash" and self._test_command:
            command = tc["arguments"].get("command", "")
            if _bash_ran_test_command(command, self._test_command):
                self._verify_tests_run = True

        # Cache read-only results
        if (self.tool_cache
                and tc["name"] in self.tool_cache.CACHEABLE_TOOLS
                and not result.startswith("Error")):
            self.tool_cache.put(tc["name"], tc["arguments"], result)

        # Invalidate cache on anything that can mutate the filesystem, including
        # bash (sed/git/formatters), not just write_file/edit_file.
        self._invalidate_cache_for(tc)

        result = self._store_tool_result(tc, result)

        # Display result
        if is_streamed_bash:
            pass
        else:
            render_tool_result(self.console, result, tool_name=tc["name"],
                               **self._todo_render_kwargs(tc["name"]))

        # Run post-hooks
        if self.hooks and self.hooks.has_hooks(f"after_{tc['name']}"):
            hook_ctx = {
                "tool": tc["name"],
                "path": tc["arguments"].get("file_path", ""),
                **tc["arguments"],
            }
            await self._run_hooks_safe(f"after_{tc['name']}", hook_ctx)

    async def _execute_parallel(self, tool_calls: list[dict]) -> list[str]:
        """Execute multiple independent, read-only, already-authorized tool
        calls concurrently. Mutating tools never reach this path (they are
        routed sequentially), so no cache-invalidation or side-effect handling
        is needed here — but read hooks still run."""

        async def _run_one(tc):
            tool = self.tools.get(tc["name"])
            if not tool:
                return f"Error: Unknown tool '{tc['name']}'"

            # Plan-mode enforcement lives on BOTH tool paths (defense in depth):
            # today only read-only tools reach here, but a future partition
            # change must not silently open a write hole in plan mode.
            if self._plan_denies(tc, tool):
                return PLAN_DENIAL

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
                await self._run_hooks_safe(
                    f"before_{tc['name']}", {"tool": tc["name"], **tc["arguments"]})

            render_tool_call(self.console, tc["name"], tc["arguments"], editor=self.editor)

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
                await self._run_hooks_safe(
                    f"after_{tc['name']}",
                    {"tool": tc["name"], "path": tc["arguments"].get("file_path", ""),
                     **tc["arguments"]})

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
