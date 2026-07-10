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
import logging
import os
import secrets
import subprocess

from rich.console import Console

from .agent import Agent, _truncate_result
from .agents_registry import AgentDef
from .config import get, load_keys, resolve_provider_key, resolve_write_roots
from .context import Context
from .model import ModelClient
from .permissions import PermissionManager
from .tools.base import Tool, ToolRegistry

logger = logging.getLogger(__name__)

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


# --- worktree isolation (Phase 5 Task 8) ------------------------------------
# An `isolated` implementer dispatch runs in a fresh `git worktree` under
# `.spark/worktrees/<short-id>/` instead of the real project tree, so its
# writes can be reviewed (and merged, or discarded) before they ever touch
# the user's working tree. Opt-in (default False) and implementer-only —
# explore/reviewer are read-only, so there's nothing to isolate.
#
# Every git invocation goes through `_run_git`, the ONE subprocess entry
# point, as an argv LIST (never a shell string — the Task 5/6 injection
# lesson: a value that embeds shell metacharacters can't be reinterpreted as
# shell syntax because there is no shell). Tests monkeypatch this single
# function to simulate any git outcome without touching a real repo.

_WORKTREE_PROMPT = (
    "\n\nISOLATION: you are running in a git worktree, NOT the main project "
    "tree. Your working directory for this task is:\n  {path}\n"
    "Use ABSOLUTE paths under this directory for every file you read, write, "
    "edit, list, glob, or grep, and prefix bash commands with `cd {path} && "
    "...`. Writes outside this directory will be refused. Do not attempt to "
    "modify anything outside {path}."
)


def _run_git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Run a git command as an argv list against *cwd*. Never a shell string.

    The sole subprocess entry point for the whole worktree lifecycle (repo
    detection, create, diffstat, remove) — one patch point for tests.
    """
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        errors="replace", timeout=30,
    )


def _repo_root(cwd: str) -> str | None:
    """The repo's top-level directory, or None if *cwd* isn't inside a git
    work tree (or git isn't runnable at all) — the non-git-repo fallback
    trigger. Never raises."""
    try:
        result = _run_git(["rev-parse", "--show-toplevel"], cwd)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    top = (result.stdout or "").strip()
    return top or None


async def _create_worktree(repo_root: str) -> tuple[str | None, str]:
    """Create a worktree under '<repo_root>/.spark/worktrees/<short-id>' from
    the repo's current HEAD (detached). Returns (path, "") on success, or
    (None, reason) on any failure — NEVER raises, so a git quirk (a locked
    index, a weird detached-HEAD edge case) degrades to the same graceful
    fallback as "not a git repo" rather than crashing the dispatch.

    The short id is generated HERE with `secrets.token_hex` — never taken
    from the model/prompt. The worktree path is therefore always exactly
    '<repo_root>/.spark/worktrees/<8 random hex chars>', strictly confined
    under `.spark/worktrees/` regardless of what the sub-agent's task text
    says.
    """
    short_id = secrets.token_hex(4)
    worktree_path = os.path.join(repo_root, ".spark", "worktrees", short_id)
    try:
        result = await asyncio.to_thread(
            _run_git, ["worktree", "add", "--detach", worktree_path, "HEAD"],
            repo_root)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"git worktree add failed to run: {e}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        return None, f"git worktree add failed: {detail}"
    return worktree_path, ""


async def _worktree_diffstat(worktree_path: str) -> str:
    """`git diff --stat` inside the worktree, for the lead to review.

    Runs `git add -A -N .` (intent-to-add) first so brand-new UNTRACKED
    files show up too — plain `git diff` never reports untracked files, so
    without this a new file created by the sub-agent would look like "no
    change" and the worktree would be auto-removed out from under it. `-N`
    only marks paths as intent-to-add (empty placeholder in the index); it
    never actually stages their content, so this has no side effect beyond
    making the diff complete.
    """
    try:
        await asyncio.to_thread(_run_git, ["add", "-A", "-N", "."], worktree_path)
        result = await asyncio.to_thread(_run_git, ["diff", "--stat"], worktree_path)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


async def _remove_worktree(repo_root: str, worktree_path: str,
                           force: bool = False) -> bool:
    """`git worktree remove` — argv, never shell. Returns whether it
    succeeded. A failure (e.g. git refuses because it still sees changes)
    just leaves the worktree in place rather than raising or deleting it by
    hand — the caller reports its path either way so the lead can inspect
    or clean it up itself."""
    args = ["worktree", "remove", worktree_path]
    if force:
        args.append("--force")
    try:
        result = await asyncio.to_thread(_run_git, args, repo_root)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _scope_registry_for_worktree(base_registry: ToolRegistry, worktree_path: str,
                                 config: dict) -> ToolRegistry:
    """A COPY of *base_registry* with the three MUTATING tools —
    write_file, edit_file, and bash — replaced by fresh instances rooted at
    *worktree_path* instead of the ambient process cwd. This is the actual
    isolation boundary:

      * write_file/edit_file get their write validation scoped to the
        worktree (a write outside it is refused). Task 1's
        `permissions.write_roots` composes in as an ADDITIONAL allowance on
        top of the worktree root (e.g. a configured sibling repo), exactly as
        it would for a non-isolated dispatch.
      * bash gets its child-process cwd scoped to the worktree, so a
        relative-path command (`echo > f`, `rm -rf *`, `sed -i`, a build/
        test that writes files) executes IN the worktree — NOT the user's
        real tracked tree. Without this, the feature's headline guarantee
        ("runs in a worktree, won't touch your files") is a lie for bash:
        one un-cd'd command is one `rm` away from real data loss.

    Deliberately NOT a global `os.chdir()`: this process can be running other
    concurrent agents/workers (team.py spawns worker Agents via
    `asyncio.create_task`, sharing this same process) that must keep seeing
    the real cwd throughout. An instance-level override on just these tool
    objects — bash uses the per-child subprocess `cwd=` kwarg, write/edit use
    an instance field — has no such cross-task blast radius.

    Read-only tools (read_file, list_dir, glob, grep, ...) are reused
    UNCHANGED from base_registry — they still resolve relative paths against
    the real process cwd, which is why the sub-agent is told (see
    `_WORKTREE_PROMPT`) to use ABSOLUTE worktree paths for reads. Scoping
    their relative-path resolution to the worktree too is a documented
    read-only follow-up (no data-loss risk, so out of scope for this fix).
    """
    from .tools.bash import BashTool
    from .tools.edit_file import EditFileTool
    from .tools.write_file import WriteFileTool

    write_roots = resolve_write_roots(config)
    scoped = ToolRegistry()
    for tool in base_registry.all():
        if tool.name == "write_file":
            scoped.register(WriteFileTool(write_roots=write_roots, cwd=worktree_path))
        elif tool.name == "edit_file":
            scoped.register(EditFileTool(write_roots=write_roots, cwd=worktree_path))
        elif tool.name == "bash":
            scoped.register(BashTool(cwd=worktree_path))
        else:
            scoped.register(tool)
    return scoped


async def _setup_worktree_isolation(
    base_registry: ToolRegistry, config: dict
) -> tuple[ToolRegistry, str | None, str | None, str]:
    """Attempt to stand up worktree isolation for an implementer dispatch.

    Returns (registry, repo_root, worktree_path, note):
      * success: (scoped registry, repo_root, worktree_path, "")
      * fallback (not a git repo, or `git worktree add` itself failed):
        (base_registry UNCHANGED, None, None, note) — note is a short,
        human-readable reason appended to the sub-agent's final report so
        the lead knows isolation was requested but didn't happen.
    """
    cwd = os.getcwd()
    repo_root = await asyncio.to_thread(_repo_root, cwd)
    if repo_root is None:
        return (base_registry, None, None,
                "not a git repository — ran without worktree isolation")
    worktree_path, err = await _create_worktree(repo_root)
    if worktree_path is None:
        return base_registry, None, None, f"{err} — ran without worktree isolation"
    # Close the leak window: scoping runs AFTER the worktree exists on disk,
    # so if it raises (e.g. a malformed config trips resolve_write_roots) the
    # just-created worktree would otherwise leak — the caller's worktree_path
    # is still None at this point, so its own cleanup wouldn't catch it.
    # Remove it here, then re-raise for the outer handler to report.
    try:
        scoped = _scope_registry_for_worktree(base_registry, worktree_path, config)
    except Exception:
        await _remove_worktree(repo_root, worktree_path, force=True)
        raise
    return scoped, repo_root, worktree_path, ""


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


def _resolve_subagent_model(custom, model, utility_model, config):
    """Pick the ModelClient a custom sub-agent runs on.

    Returns ``(chosen_model, owns_client)``. ``owns_client=True`` means the
    CALLER must close ``chosen_model`` after the run — it was built here,
    unlike the caller-managed lead ``model`` / ``utility_model``.

    ``custom.provider`` (Phase 6 Task 1) takes priority when set and names a
    provider present in ``config["providers"]``: a brand-new ``ModelClient``
    is built for it (mirrors ``cli.py``'s ``_model_factory``). An unknown or
    missing provider name falls back to the lead's primary ``model`` — this
    must never raise, so a bad/typo'd provider name in a ``.spark/agents/*.md``
    def degrades gracefully instead of crashing dispatch. With no provider,
    behavior is unchanged from before this feature: ``model_hint == "utility"``
    routes to ``utility_model``, everything else uses the lead ``model``.
    """
    if custom is not None and getattr(custom, "provider", None):
        name = custom.provider
        pconf = (config.get("providers", {}) or {}).get(name)
        if not pconf:
            logger.warning("agent '%s': unknown provider '%s' — using primary model",
                           custom.name, name)
            return model, False
        client = ModelClient(
            endpoint=pconf.get("endpoint", "http://localhost:11434"),
            model=pconf.get("model", "unknown"),
            temperature=pconf.get("temperature", 0.7),
            max_tokens=pconf.get("max_tokens", 8192),
            api_key=resolve_provider_key(name, pconf, load_keys()),
            provider=name,
            timeout=float(pconf.get("timeout", 300)),
            supports_vision=bool(pconf.get("vision", False)),
        )
        return client, True
    if custom is not None and custom.model_hint == "utility" and utility_model is not None:
        return utility_model, False
    return model, False


async def run_subagent(model, prompt: str, agent_type: str, config: dict,
                       lead_mode: str,
                       agent_defs: dict[str, AgentDef] | None = None,
                       utility_model=None,
                       isolated: bool = False,
                       mcp_tools: list[Tool] | None = None) -> str:
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

    ``mcp_tools`` (browser/MCP bugfix) are the LEAD's already-connected MCP
    tool instances (e.g. Playwright browser control) — there is one browser,
    so unlike local tools these are SHARED with the sub-agent rather than
    rebuilt fresh. When non-empty, a base registry is assembled from a fresh
    ``build_tools()`` (still fresh local-tool instances — no state leak) plus
    these shared MCP tools, and that becomes the base BOTH
    ``_build_subagent_registry`` and ``_build_custom_registry`` filter from —
    the only way an MCP tool (or a custom def's ``tools:`` allowlist naming
    one) can ever reach a sub-agent's registry. Empty/``None`` (the default)
    keeps behavior byte-identical to before this parameter existed.

    ``isolated`` (Phase 5 Task 8) is opt-in and applies ONLY to the literal
    built-in ``agent_type == "implementer"`` (never a custom def, even one
    whose ``base_type`` is implementer, and never explore/reviewer — they're
    read-only, nothing to isolate). When it applies, a git worktree is
    created fresh under ``.spark/worktrees/<short-id>/`` from the repo's
    current HEAD, and the sub-agent's write_file/edit_file tools are scoped
    to it instead of the real project tree (see
    ``_scope_registry_for_worktree``). Worktree creation happens INSIDE the
    concurrency semaphore, right alongside the sub-agent run — it's the
    expensive part, so it's capped exactly like everything else this
    semaphore gates. Not a git repo (or ``git worktree add`` itself fails)
    → falls back to a plain, non-isolated dispatch with a note appended to
    the report. On success, the report gains the worktree path and a
    ``git diff --stat`` of what changed; an unchanged worktree is removed
    automatically, a changed one is left for the lead to review/merge.

    Any failure inside the sub-agent is caught and returned as a
    ``"[dispatch_agent error] ..."`` string — it NEVER raises into the lead's
    loop; the worktree is force-removed first so a crash can't leak one. The
    returned text is truncated head+tail to 8,000 chars.
    """
    agent_defs = agent_defs or {}
    custom = agent_defs.get(agent_type)
    if agent_type not in AGENT_TYPES and custom is None:
        known = list(AGENT_TYPES) + sorted(agent_defs)
        return (f"[dispatch_agent error] unknown agent_type '{agent_type}' "
                f"(expected one of: {', '.join(known)})")
    if not prompt or not prompt.strip():
        return "[dispatch_agent error] empty prompt — nothing to dispatch"

    repo_root = worktree_path = None
    isolation_note = ""
    try:
        # Only construct a base registry (and pay a fresh build_tools() cost)
        # when there are MCP tools to thread through — the common no-MCP
        # case falls straight back to each helper's own build_tools()
        # default, unchanged from before this parameter existed.
        base_reg = None
        if mcp_tools:
            from .cli import build_tools
            base_reg = build_tools()
            for t in mcp_tools:
                base_reg.register(t)

        if custom is not None:
            registry = _build_custom_registry(custom, base_registry=base_reg)
            system_prompt = _SUBAGENT_BASE_PROMPT + "\n\n" + custom.system_prompt
            chosen_model, _owns_subagent_model = _resolve_subagent_model(
                custom, model, utility_model, config)
        else:
            registry = _build_subagent_registry(agent_type, base_registry=base_reg)
            system_prompt = _SUBAGENT_BASE_PROMPT + _SUBAGENT_TYPE_PROMPTS[agent_type]
            chosen_model = model
            _owns_subagent_model = False

        # Everything from here through the semaphore-gated run is wrapped in
        # a try/finally so a provider-owned client (built just above by
        # _resolve_subagent_model) is closed on EVERY exit path — including
        # asyncio.CancelledError raised while still parked on
        # ``semaphore.acquire()``. That is a real path: the lead can fan out
        # several dispatch_agent calls concurrently via asyncio.gather
        # (default agents.max_concurrent=3), and a Ctrl+C or task-group
        # cancellation can cancel a sub-agent's task while it is still
        # waiting for a semaphore slot — before it ever reaches the inner
        # try/except below. Previously the close lived INSIDE the inner
        # try's finally, nested inside ``async with semaphore:``, so that
        # cancellation path (and any failure from ``_get_semaphore`` itself)
        # skipped it and leaked the owned client's eagerly-opened
        # httpx.AsyncClient. The lead's shared ``model``/``utility_model``
        # are still never closed here — this finally only acts when
        # ``_owns_subagent_model`` is True, which is exclusively the
        # provider-built-client path from ``_resolve_subagent_model``.
        try:
            semaphore = _get_semaphore(config)
            async with semaphore:
                # Worktree creation is the expensive part of isolation, so it
                # happens HERE — inside the same semaphore acquisition as the
                # sub-agent run — rather than before it. Concurrency stays capped
                # at agents.max_concurrent exactly like a non-isolated dispatch.
                if isolated and agent_type == "implementer" and custom is None:
                    registry, repo_root, worktree_path, isolation_note = (
                        await _setup_worktree_isolation(registry, config))
                    if worktree_path is not None:
                        system_prompt += _WORKTREE_PROMPT.format(path=worktree_path)

                # Everything from here on (Context/Agent construction AND the
                # actual run) is wrapped so ANY failure — not just one inside
                # sub_agent.run() itself — force-removes a worktree that was
                # already created rather than leaking it. Re-raised either way,
                # into the outer handler, which formats the
                # "[dispatch_agent error] ..." string the lead sees.
                try:
                    sub_context = Context(
                        system_prompt=system_prompt,
                        max_tokens=int(get(config, "model", "context_window", default=32768)),
                        provider_prompt=get(config, "model", "system_prompt", default="") or "",
                    )
                    # Inherit the lead's mode but NEVER prompt: a sub-agent runs on
                    # the shared event loop, where a blocking prompt would freeze
                    # the session. The non-interactive manager fails safe — it
                    # allows only what the mode would allow without a prompt and
                    # denies the rest.
                    permissions = PermissionManager(mode=lead_mode, interactive=False)

                    sub_agent = Agent(
                        chosen_model, sub_context, registry, permissions,
                        console=Console(quiet=True),
                        # Non-empty prefix so the sub-agent skips the Rich Live
                        # display (its output must not fight the lead's) in
                        # addition to the quiet console.
                        output_prefix="sub",
                        result_budgets=get(config, "tools", "result_budgets", default=None),
                    )
                    # Own round cap: agents.max_rounds (default 15), independent
                    # of the lead's MAX_TOOL_ROUNDS. Instance attribute shadows
                    # the class attr.
                    sub_agent.MAX_TOOL_ROUNDS = int(
                        get(config, "agents", "max_rounds", default=_DEFAULT_MAX_ROUNDS)
                        or _DEFAULT_MAX_ROUNDS)

                    result = await sub_agent.run(prompt)
                except BaseException:
                    # Never leak a worktree on a sub-agent crash OR cancellation.
                    # asyncio.CancelledError is a BaseException (NOT Exception),
                    # so a plain `except Exception` skipped it — and cancellation
                    # is a real path here (a /loop round's 15-min asyncio.wait_for
                    # timeout cancels agent.run, and a lead Ctrl+C cancels the
                    # task group), which then leaked `.spark/worktrees/<id>/` on
                    # disk and in `git worktree list`. Force-remove, then re-raise
                    # so the cancellation/exception still propagates unchanged.
                    if worktree_path is not None:
                        try:
                            await _remove_worktree(
                                repo_root, worktree_path, force=True)
                        except Exception:
                            pass
                    raise
        finally:
            # Only a provider-built client is ours to close (owns_client is
            # True only from _resolve_subagent_model's provider path) — the
            # lead's shared model and utility_model are caller-managed and
            # must NEVER be closed here. This finally wraps
            # _get_semaphore(...) AND ``async with semaphore:`` themselves
            # (not just the run inside it), so the close also fires if the
            # task is cancelled while still waiting for a slot, or if
            # _get_semaphore raises. Guarded by its own try/except so a
            # failure while closing can't mask the original exception/
            # cancellation propagating out of this finally.
            if _owns_subagent_model:
                try:
                    await chosen_model.close()
                except Exception:
                    pass
    except Exception as e:  # noqa: BLE001 — must never escape into the lead loop
        return f"[dispatch_agent error] {type(e).__name__}: {e}"

    result = (result or "").strip() or "(sub-agent produced no output)"

    if worktree_path is not None:
        diffstat = await _worktree_diffstat(worktree_path)
        if diffstat:
            result += (f"\n\n[isolated worktree kept for review: {worktree_path}]\n"
                      f"git diff --stat:\n{diffstat}")
        else:
            removed = await _remove_worktree(repo_root, worktree_path)
            if removed:
                result += f"\n\n[isolated worktree removed — no changes made: {worktree_path}]"
            else:
                # Belt-and-suspenders: git itself declined to remove it (still
                # sees something dirty) — leave it rather than force-delete
                # potential work, and say so.
                result += (f"\n\n[isolated worktree: {worktree_path}]\n"
                          "(no diff detected, but it could not be auto-removed — "
                          "left in place for review)")
    elif isolation_note:
        result += f"\n\n[isolation note: {isolation_note}]"

    return _truncate_result(result, _SUBAGENT_RESULT_BUDGET)


def _resolve_lead_mode(get_lead_mode, permissions, lead_mode) -> str:
    """Resolve the lead's CURRENT permission mode, most-specific source first:
    an explicit getter (respects a mid-session Shift+Tab change), then a live
    PermissionManager's ``.mode``, then a static string, defaulting to the
    safest ``"ask"``. Shared by every dispatch-style tool so a sub-agent it
    launches always inherits the mode the lead is in RIGHT NOW."""
    if get_lead_mode is not None:
        try:
            return get_lead_mode() or "ask"
        except Exception:
            return "ask"
    if permissions is not None:
        return getattr(permissions, "mode", "ask") or "ask"
    return lead_mode or "ask"


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
        "cannot itself dispatch further sub-agents. isolated=true (implementer "
        "only) runs it in a fresh git worktree instead of the real project "
        "tree, so its diff can be reviewed before it touches your files."
    )

    def __init__(self, model=None, config: dict | None = None,
                 permissions: PermissionManager | None = None,
                 lead_mode: str | None = None,
                 get_lead_mode=None,
                 agent_defs: dict[str, AgentDef] | None = None,
                 utility_model=None,
                 mcp_tools=None):
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
        # Browser/MCP bugfix: the lead's already-connected MCP tool instances
        # (e.g. Playwright browser control) — threaded into run_subagent so a
        # sub-agent's registry can actually contain them (see run_subagent's
        # docstring). Empty/None when the lead has no MCP servers connected —
        # every existing dispatch is byte-for-byte unaffected.
        self._mcp_tools = list(mcp_tools or [])

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
                "isolated": {
                    "type": "boolean",
                    "description": (
                        "Only affects agent_type='implementer' (ignored for "
                        "explore/reviewer/custom types). When true, run in a "
                        "fresh git worktree under .spark/worktrees/ instead of "
                        "the real project tree — the report includes the "
                        "worktree path and a diffstat for you to review/merge. "
                        "Falls back to a normal (non-isolated) dispatch, with a "
                        "note, if this isn't a git repo. Default: false."
                    ),
                },
            },
            "required": ["prompt", "agent_type"],
        }

    def _resolve_lead_mode(self) -> str:
        return _resolve_lead_mode(
            self._get_lead_mode, self._permissions, self._lead_mode)

    async def execute(self, prompt: str = "", agent_type: str = "explore",
                      isolated: bool = False, **kwargs) -> str:
        return await run_subagent(
            self._model, prompt, agent_type, self._config,
            self._resolve_lead_mode(),
            agent_defs=self._agent_defs, utility_model=self._utility_model,
            isolated=bool(isolated), mcp_tools=self._mcp_tools)


class BrowseWebTool(Tool):
    """Drive a REAL web browser via the ``web-driver`` sub-agent (Gemini vision).

    A thin wrapper over :func:`run_subagent` that always routes to
    ``agent_type="web-driver"`` — the ``.spark/agents/web-driver.md`` custom
    def that actually controls a browser (navigate, click, type, fill, submit)
    using Gemini vision. It exists because the model kept hallucinating a
    nonexistent ``web_driver`` tool instead of ``dispatch_agent(agent_type=
    "web-driver")``; this gives that capability its own first-class name.

    Constructor mirrors :class:`DispatchAgentTool` exactly (same model, config,
    lead-mode resolution, ``agent_defs``, ``utility_model`` and shared
    ``mcp_tools`` — the browser MCP tools the web-driver def's ``tools:``
    allowlist actually names). Register it in ``cli.build_tools`` right beside
    ``DispatchAgentTool`` with the same arguments; it is intentionally NOT
    self-registering.
    """

    name = "browse_web"
    _BASE_DESCRIPTION = (
        "Drive a REAL web browser via the web-driver sub-agent (Gemini vision). "
        "USE THIS for interactive web tasks — clicking, typing, filling forms, "
        "logging in, submitting, or navigating a multi-step flow on a live "
        "page. Do NOT use it for read-only page reads: to just grab a page's "
        "text or check a URL, use web_fetch instead. The web-driver runs in "
        "its own isolated context and returns only a final report of what it "
        "did and the ending page state."
    )

    def __init__(self, model=None, config: dict | None = None,
                 permissions: PermissionManager | None = None,
                 lead_mode: str | None = None,
                 get_lead_mode=None,
                 agent_defs: dict[str, AgentDef] | None = None,
                 utility_model=None,
                 mcp_tools=None):
        self._model = model
        self._config = config or {}
        # Lead-mode resolution, identical to DispatchAgentTool: getter, then a
        # live PermissionManager, then a static string, defaulting to "ask".
        self._permissions = permissions
        self._lead_mode = lead_mode
        self._get_lead_mode = get_lead_mode
        self._agent_defs = agent_defs or {}
        self._utility_model = utility_model
        # The lead's already-connected MCP tool instances (the playwright
        # browser control the web-driver def's tools: allowlist names) — SHARED
        # with the sub-agent, since there is one browser.
        self._mcp_tools = list(mcp_tools or [])

    @property
    def description(self) -> str:
        return self._BASE_DESCRIPTION

    @property
    def is_read_only(self) -> bool:
        return False

    @property
    def requires_permission(self) -> bool:
        return True

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "What to do in the browser, in plain language — e.g. "
                        "'log into example.com with these credentials and "
                        "download the latest invoice'. Be specific and "
                        "self-contained: the web-driver starts with NO "
                        "knowledge of this conversation, so include every "
                        "detail it needs and say exactly what to report back."
                    ),
                },
                "url": {
                    "type": "string",
                    "description": (
                        "Optional starting URL to open before doing the task."
                    ),
                },
            },
            "required": ["task"],
        }

    def _resolve_lead_mode(self) -> str:
        return _resolve_lead_mode(
            self._get_lead_mode, self._permissions, self._lead_mode)

    async def execute(self, task: str = "", url: str = "", **kwargs) -> str:
        prompt = (task or "").strip()
        if url:
            prompt = f"Starting URL: {url}\n\n{prompt}"
        return await run_subagent(
            self._model, prompt, "web-driver", self._config,
            self._resolve_lead_mode(),
            agent_defs=self._agent_defs, utility_model=self._utility_model,
            mcp_tools=self._mcp_tools)
