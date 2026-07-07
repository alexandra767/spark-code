"""dispatch_agent — subagents as context isolation (Phase 2 Task 4).

A ``dispatch_agent`` tool lets the lead agent hand an open-ended sub-task
(codebase search, review, focused implementation) to a fresh sub-agent that
runs in its OWN context window. Only the sub-agent's final message comes back
to the lead — its exploration, tool spew, and reasoning never touch the lead's
context. That is the whole point: keep the lead's context small while still
doing deep work.

This module composes EXISTING pieces — it owns no new agent machinery:
  * ``Agent`` (the same loop the lead runs) with its own round cap;
  * a fresh ``Context`` seeded with a compact sub-agent system prompt;
  * a tool registry filtered by agent type (read-only for explore/reviewer,
    everything-minus-dispatch_agent for implementer — the depth guard);
  * a NEW non-interactive ``PermissionManager`` inheriting the lead's mode
    (the Phase-0 fail-safe path: it never prompts, and denies anything that
    would have needed a prompt);
  * the lead's SHARED ``ModelClient`` (one engine handles the concurrency);
  * a module-level ``asyncio.Semaphore`` capping concurrent sub-agents.

``run_subagent(...)`` is the standalone core so other callers (Task 7's
``/review`` swarm) can reuse it directly; ``DispatchAgentTool`` is a thin
wrapper around it.
"""

from __future__ import annotations

import asyncio

from rich.console import Console

from .agent import Agent, _truncate_result
from .agents_registry import AgentDef
from .config import get
from .context import Context
from .permissions import PermissionManager
from .tools.base import Tool, ToolRegistry

# Sub-agent transcripts get the tight 8K head+tail budget (also declared in
# agent.TOOL_RESULT_BUDGETS so the LEAD truncates the returned summary the same
# way if the tool ever returns more).
_SUBAGENT_RESULT_BUDGET = 8000

_DEFAULT_MAX_CONCURRENT = 3
_DEFAULT_MAX_ROUNDS = 15

AGENT_TYPES = ("explore", "reviewer", "implementer")
_READ_ONLY_TYPES = ("explore", "reviewer")

# Compact sub-agent system prompt. The load-bearing sentence (per plan): the
# final message is returned to the caller as DATA — report findings, not
# conversation. Kept short: the sub-agent's context is precious.
_SUBAGENT_BASE_PROMPT = (
    "You are a Spark Code sub-agent, launched by the lead agent to handle ONE "
    "focused task in an isolated context. Your FINAL message is returned to the "
    "caller verbatim as data — so finish with a complete, self-contained report "
    "of your findings (concrete file paths, line numbers, values, conclusions), "
    "not a conversational reply. Use your tools to do the work, then report. Be "
    "concise and specific; do not ask the caller questions — you cannot receive "
    "an answer."
)

_SUBAGENT_TYPE_PROMPTS = {
    "explore": (
        "\n\nTask type: EXPLORE. Search the codebase to answer the question. "
        "You have read-only tools. Report exactly what you found, with file "
        "paths and the relevant values or snippets."
    ),
    "reviewer": (
        "\n\nTask type: REVIEW. Read the code under review and report issues as "
        "concrete findings in the form `file:line — problem`. You have read-only "
        "tools; do not attempt to change anything."
    ),
    "implementer": (
        "\n\nTask type: IMPLEMENT. Make the requested change with your tools, "
        "verify it if you can, then report what you changed (files and a short "
        "summary of each edit)."
    ),
}


# --- shared concurrency gate ------------------------------------------------
# One semaphore for the whole session caps how many sub-agents run at once
# (agents.max_concurrent, default 3). It is created lazily on first use so it
# binds to the running event loop, and rebound if the loop changes (tests run
# each case in a fresh loop) — within a single session there is exactly one
# loop, so it is created once and shared across every dispatch.
_dispatch_semaphore: asyncio.Semaphore | None = None
_dispatch_semaphore_loop: object | None = None


def _get_semaphore(config: dict) -> asyncio.Semaphore:
    global _dispatch_semaphore, _dispatch_semaphore_loop
    loop = asyncio.get_running_loop()
    if _dispatch_semaphore is None or _dispatch_semaphore_loop is not loop:
        limit = int(get(config, "agents", "max_concurrent",
                        default=_DEFAULT_MAX_CONCURRENT) or _DEFAULT_MAX_CONCURRENT)
        _dispatch_semaphore = asyncio.Semaphore(max(1, limit))
        _dispatch_semaphore_loop = loop
    return _dispatch_semaphore


def _reset_semaphore(limit: int | None = None) -> None:
    """Test hook: drop the shared semaphore (or force a fixed limit)."""
    global _dispatch_semaphore, _dispatch_semaphore_loop
    if limit is None:
        _dispatch_semaphore = None
        _dispatch_semaphore_loop = None
    else:
        _dispatch_semaphore = asyncio.Semaphore(max(1, limit))
        try:
            _dispatch_semaphore_loop = asyncio.get_running_loop()
        except RuntimeError:
            _dispatch_semaphore_loop = None


def _build_subagent_registry(agent_type: str,
                             base_registry: ToolRegistry | None = None) -> ToolRegistry:
    """Build the filtered tool registry for a sub-agent of ``agent_type``.

    Starts from a fresh full tool set (``build_tools()`` — which gives the
    sub-agent its OWN tool instances, e.g. a fresh todo list, so it can never
    leak state into the lead) and filters:

      * ``explore`` / ``reviewer`` → read-only tools only (``is_read_only``);
      * ``implementer`` → everything EXCEPT ``dispatch_agent`` (depth guard:
        a sub-agent can never dispatch another sub-agent).

    ``dispatch_agent`` is stripped by name regardless of type — belt and
    suspenders on top of the fact that a fresh ``build_tools()`` (called with
    no model/config) never registers it in the first place.
    """
    if base_registry is not None:
        base = base_registry
    else:
        # Local import to avoid an import cycle (cli imports this module).
        from .cli import build_tools
        base = build_tools()

    read_only_only = agent_type in _READ_ONLY_TYPES
    sub = ToolRegistry()
    for tool in base.all():
        if tool.name == "dispatch_agent":
            continue  # depth guard — never nest sub-agents
        if read_only_only and not tool.is_read_only:
            continue
        sub.register(tool)
    return sub


def _build_custom_registry(agent_def: AgentDef,
                           base_registry: ToolRegistry | None = None) -> ToolRegistry:
    """Build the filtered tool registry for a CUSTOM agent def.

    SECURITY: this is an INTERSECTION, never a replacement. Start from the
    def's ``base_type``'s own filtered registry (read-only for
    explore/reviewer-class defs, everything-minus-dispatch_agent for
    implementer-class) via ``_build_subagent_registry`` — exactly the same
    call a built-in dispatch would make. THEN, if the def declares its own
    ``tools`` allowlist, narrow further to that named subset.

    A read-only-class def (``base_type in ("explore", "reviewer")``, the
    default) can therefore never end up with a write tool no matter what its
    ``tools:`` frontmatter claims — ``write_file`` was never in the base
    registry to begin with, so it has nothing to intersect with. ``tools``
    can only SHRINK the base type's permissions, never grow them.
    """
    base = _build_subagent_registry(agent_def.base_type, base_registry=base_registry)
    if agent_def.tools is None:
        return base
    allow = set(agent_def.tools)
    sub = ToolRegistry()
    for tool in base.all():
        if tool.name in allow:
            sub.register(tool)
    return sub


async def run_subagent(model, prompt: str, agent_type: str, config: dict,
                       lead_mode: str,
                       agent_defs: dict[str, AgentDef] | None = None,
                       utility_model=None) -> str:
    """Run a sub-agent to completion in a fresh context and return its report.

    Shares the lead's ``model`` (same engine handles concurrent requests) but
    builds a brand-new ``Context``, a type-filtered tool registry, and a
    NON-interactive ``PermissionManager`` in the lead's ``lead_mode``. Bounded
    by ``agents.max_rounds`` rounds and the session-wide concurrency semaphore.

    ``agent_defs`` (Phase 5 Task 4) additionally maps custom agent-type names
    (from ``.spark/agents/*.md``) to :class:`~spark_code.agents_registry.AgentDef`.
    When ``agent_type`` matches a key there instead of a built-in
    ``AGENT_TYPES`` entry, the sub-agent runs with that def's
    ``system_prompt`` and an INTERSECTED tool registry — see
    ``_build_custom_registry`` for why a def can never exceed its
    ``base_type``'s permissions. ``utility_model`` is used instead of the
    lead's ``model`` only when the matched def's ``model_hint == "utility"``
    (Phase 4 Task 2 dual-model routing); every other path is unaffected and
    behaves exactly as before this feature existed.

    Any failure inside the sub-agent is caught and returned as a
    ``"[dispatch_agent error] ..."`` string — it NEVER raises into the lead's
    loop. The returned text is truncated head+tail to 8,000 chars.
    """
    agent_defs = agent_defs or {}
    custom = agent_defs.get(agent_type)
    if agent_type not in AGENT_TYPES and custom is None:
        known = list(AGENT_TYPES) + sorted(agent_defs)
        return (f"[dispatch_agent error] unknown agent_type '{agent_type}' "
                f"(expected one of: {', '.join(known)})")
    if not prompt or not prompt.strip():
        return "[dispatch_agent error] empty prompt — nothing to dispatch"

    try:
        if custom is not None:
            registry = _build_custom_registry(custom)
            system_prompt = _SUBAGENT_BASE_PROMPT + "\n\n" + custom.system_prompt
            chosen_model = (utility_model if custom.model_hint == "utility"
                            and utility_model is not None else model)
        else:
            registry = _build_subagent_registry(agent_type)
            system_prompt = _SUBAGENT_BASE_PROMPT + _SUBAGENT_TYPE_PROMPTS[agent_type]
            chosen_model = model
        sub_context = Context(
            system_prompt=system_prompt,
            max_tokens=int(get(config, "model", "context_window", default=32768)),
            provider_prompt=get(config, "model", "system_prompt", default="") or "",
        )
        # Inherit the lead's mode but NEVER prompt: a sub-agent runs on the
        # shared event loop, where a blocking prompt would freeze the session.
        # The non-interactive manager fails safe — it allows only what the mode
        # would allow without a prompt and denies the rest.
        permissions = PermissionManager(mode=lead_mode, interactive=False)

        sub_agent = Agent(
            chosen_model, sub_context, registry, permissions,
            console=Console(quiet=True),
            # Non-empty prefix so the sub-agent skips the Rich Live display
            # (its output must not fight the lead's) in addition to the quiet
            # console.
            output_prefix="sub",
            result_budgets=get(config, "tools", "result_budgets", default=None),
        )
        # Own round cap: agents.max_rounds (default 15), independent of the
        # lead's MAX_TOOL_ROUNDS. Instance attribute shadows the class attr.
        sub_agent.MAX_TOOL_ROUNDS = int(
            get(config, "agents", "max_rounds", default=_DEFAULT_MAX_ROUNDS)
            or _DEFAULT_MAX_ROUNDS)

        semaphore = _get_semaphore(config)
        async with semaphore:
            result = await sub_agent.run(prompt)
    except Exception as e:  # noqa: BLE001 — must never escape into the lead loop
        return f"[dispatch_agent error] {type(e).__name__}: {e}"

    result = (result or "").strip() or "(sub-agent produced no output)"
    return _truncate_result(result, _SUBAGENT_RESULT_BUDGET)


class DispatchAgentTool(Tool):
    """Launch a sub-agent in a fresh context and return only its final report.

    Thin wrapper over :func:`run_subagent`. Holds the lead's shared model,
    the config, and a way to read the lead's CURRENT permission mode at call
    time (so a mid-session mode change via Shift+Tab is respected).
    """

    name = "dispatch_agent"
    _BASE_DESCRIPTION = (
        "Launch a sub-agent that runs in its OWN fresh context and returns only "
        "a final summary — ideal for open-ended codebase searches, code reviews, "
        "or focused research where you don't want the raw exploration to fill "
        "your own context. agent_type: 'explore' (read-only search), 'reviewer' "
        "(read-only review), or 'implementer' (can edit files). The sub-agent "
        "cannot itself dispatch further sub-agents."
    )

    def __init__(self, model=None, config: dict | None = None,
                 permissions: PermissionManager | None = None,
                 lead_mode: str | None = None,
                 get_lead_mode=None,
                 agent_defs: dict[str, AgentDef] | None = None,
                 utility_model=None):
        self._model = model
        self._config = config or {}
        # Lead-mode resolution, most-specific first: an explicit getter, then a
        # live PermissionManager (reads .mode at call time — respects Shift+Tab),
        # then a static string, defaulting to "ask" (the safest mode).
        self._permissions = permissions
        self._lead_mode = lead_mode
        self._get_lead_mode = get_lead_mode
        # Phase 5 Task 4: custom agent types loaded from .spark/agents/*.md +
        # ~/.spark/agents/*.md (agents_registry.load_agent_defs). {} when the
        # feature is unused — the enum/description/dispatch behavior are then
        # byte-for-byte what they were before this feature existed.
        self._agent_defs = agent_defs or {}
        # Phase 4 Task 2 dual-model routing: only consulted for a custom def
        # whose model_hint == "utility" (see run_subagent) — every built-in
        # dispatch and every def without that hint ignores this entirely.
        self._utility_model = utility_model

    @property
    def description(self) -> str:
        # Custom agent types (Phase 5 Task 4) get a LEAN index line each —
        # name + one-line description, same shape as the skill index
        # (SkillRegistry.build_index) — appended to the static base
        # description. Never the def's full system_prompt: that only enters
        # context on an actual dispatch (see run_subagent).
        if not self._agent_defs:
            return self._BASE_DESCRIPTION
        from .agents_registry import build_agent_index
        index = build_agent_index(self._agent_defs)
        return (self._BASE_DESCRIPTION
                + "\n\nCustom agent types (.spark/agents/*.md):\n" + index)

    @property
    def is_read_only(self) -> bool:
        # A single Tool instance carries ONE is_read_only, but the answer really
        # depends on agent_type (explore/reviewer are read-only; implementer is
        # not). The agent's parallel/permission machinery checks is_read_only
        # per-tool, not per-call, so there is no clean way to vary it per
        # invocation here. We choose the CONSERVATIVE value — False — so a
        # dispatch is always routed through the sequential/permission path and
        # never treated as an auto-allowed read. (See task-4-report.md.)
        return False

    @property
    def requires_permission(self) -> bool:
        return True

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The self-contained task for the sub-agent. It starts "
                        "with NO knowledge of this conversation, so include all "
                        "context it needs and say exactly what to report back."
                    ),
                },
                "agent_type": {
                    "type": "string",
                    # Built-ins first (stable order — existing tests assert
                    # this exact list), then custom names sorted for a
                    # deterministic, cache-friendly schema.
                    "enum": list(AGENT_TYPES) + sorted(self._agent_defs),
                    "description": (
                        "explore = read-only codebase search/research; "
                        "reviewer = read-only code review; "
                        "implementer = may edit files. See the description "
                        "above for any custom agent types."
                    ),
                },
            },
            "required": ["prompt", "agent_type"],
        }

    def _resolve_lead_mode(self) -> str:
        if self._get_lead_mode is not None:
            try:
                return self._get_lead_mode() or "ask"
            except Exception:
                return "ask"
        if self._permissions is not None:
            return getattr(self._permissions, "mode", "ask") or "ask"
        return self._lead_mode or "ask"

    async def execute(self, prompt: str = "", agent_type: str = "explore",
                      **kwargs) -> str:
        return await run_subagent(
            self._model, prompt, agent_type, self._config,
            self._resolve_lead_mode(),
            agent_defs=self._agent_defs, utility_model=self._utility_model)
