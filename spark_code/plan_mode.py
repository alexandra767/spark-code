"""True plan mode — a shared state object, the mode cycle, and the exit gate.

Plan mode graduated from a loose ``config["_plan_mode"]`` dict flag (which only
prompt-wrapped the user's message) to real enforcement at tool-execution time.
The moving parts live here so the CLI and the Agent hold the SAME objects:

  * :class:`PlanState` — a one-field object (``.active``) shared by the CLI's
    Shift+Tab / slash handlers and the Agent's per-tool enforcement, so the two
    can never disagree about whether plan mode is on.
  * :func:`cycle_mode` — the Shift+Tab mode cycle (ask → auto → plan → ask);
    entering plan flips ``PlanState.active`` on and drops permissions to
    ``auto`` (reads run free, writes are blocked by the plan gate).
  * :class:`ExitPlanModeTool` — the ``exit_plan_mode`` tool the model calls to
    present its finished plan; renders a Rich panel and runs a Ctrl+C-safe,
    watcher-paused approval prompt. Approve → leave plan mode + permissions to
    ``auto``; reject → stay, carry the feedback back to the model.

Enforcement itself lives in :meth:`spark_code.agent.Agent._plan_denies` (called
from both the sequential and parallel tool paths); this module owns the shared
denial text (:data:`PLAN_DENIAL`) and the on-entry system nudge
(:data:`PLAN_NUDGE`).
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .tools.base import Tool

# Returned as the tool result when plan mode blocks a write action. Kept here so
# the Agent's enforcement and any test assert against ONE string.
PLAN_DENIAL = (
    "Plan mode: write actions are blocked. Present your plan with exit_plan_mode."
)

# Transient system nudge injected on every request WHILE plan mode is active
# (mirrors the agent's round-limit nudges — never persisted to history, and
# automatically gone the moment plan mode exits). See Agent._agent_loop.
PLAN_NUDGE = (
    "You are in plan mode: use read-only tools to research, then present a "
    "concrete plan by calling exit_plan_mode. Do NOT attempt writes, edits, or "
    "shell side effects — they are blocked until your plan is approved."
)

# Shift+Tab mode cycle. Trust is deliberately absent (reachable only via
# /trust, --trust, or /mode trust) — matching Claude Code, where
# bypassPermissions is never cycled into. Kept in lockstep with
# spark_code.ui.input._MODE_CYCLE (the toolbar's copy).
MODE_CYCLE = ("ask", "auto", "plan")


class PlanState:
    """Shared plan-mode flag held by BOTH the CLI and the Agent.

    Replaces the old ``config["_plan_mode"]`` dict entry. Exactly one instance
    is created per interactive session; the CLI (Shift+Tab, ``/mode``,
    ``/ask``/``/auto``/``/trust``) and the Agent's tool-execution enforcement
    read and write the SAME object, so they always agree.
    """

    __slots__ = ("active",)

    def __init__(self, active: bool = False):
        self.active = bool(active)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"PlanState(active={self.active})"


def cycle_mode(current: str, plan_state: PlanState, permissions) -> str:
    """Advance the Shift+Tab mode cycle one step and apply the result in place.

    ``current`` is the mode name currently displayed ("plan" when plan mode is
    active, else ``permissions.mode``). Returns the new mode name. Entering
    plan sets ``plan_state.active`` True and drops ``permissions.mode`` to
    ``auto`` (reads free, writes blocked by the plan gate); any non-plan target
    clears plan mode and sets ``permissions.mode`` to that target. A ``current``
    not in the cycle (e.g. ``trust``) restarts the cycle at index 0 → ``auto``.
    """
    idx = MODE_CYCLE.index(current) if current in MODE_CYCLE else 0
    next_mode = MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]
    if next_mode == "plan":
        plan_state.active = True
        permissions.mode = "auto"
    else:
        plan_state.active = False
        permissions.mode = next_mode
    return next_mode


# Nord palette (mirrors permissions.py) for the approval panel.
_C_GREEN = "#a3be8c"
_C_YELLOW = "#ebcb8b"
_C_DIM = "#7b88a1"


class ExitPlanModeTool(Tool):
    """``exit_plan_mode`` — present the finished plan for approval.

    ``is_read_only = True`` so plan-mode enforcement never blocks the gate
    itself. It runs its OWN interactive approval (not the generic permission
    prompt), so ``requires_permission`` is False. The approval ``Prompt.ask``
    happens mid-turn, so it is wrapped in ``esc_watcher.pause_all()`` (else the
    Esc watcher thread swallows the user's y/n from the shared stdin fd) and is
    Ctrl+C-safe (an interrupt is treated as "revise", never crashing the turn).
    """

    name = "exit_plan_mode"
    description = (
        "Present your completed implementation plan to the user for approval. "
        "Call this ONLY in plan mode, once you have finished read-only research "
        "and are ready to make changes — pass the full plan (markdown, with the "
        "concrete steps and files) as `plan`. The user approves it (you may then "
        "edit files) or asks for revisions. This is the ONLY way to leave plan "
        "mode; you cannot write, edit, or run side effects until it is approved."
    )

    def __init__(self, plan_state: PlanState,
                 permissions=None, console: Console | None = None):
        self._plan_state = plan_state
        self._permissions = permissions
        self._console = console or Console()

    @property
    def is_read_only(self) -> bool:
        # The gate must never be blocked by plan-mode enforcement, and it makes
        # no filesystem changes of its own.
        return True

    @property
    def requires_permission(self) -> bool:
        # It runs its own approval flow; the generic permission prompt would be
        # redundant (and wrong — approving the TOOL is not approving the PLAN).
        return False

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": (
                        "The full plan to present for approval — the concrete "
                        "steps you intend to take and the files you will change, "
                        "as markdown."
                    ),
                },
            },
            "required": ["plan"],
        }

    async def execute(self, plan: str = "", **kwargs) -> str:
        if self._plan_state is None or not self._plan_state.active:
            return "Not in plan mode."

        console = self._console
        body = plan.strip() or "(no plan text provided)"
        try:
            console.print(Panel(
                Markdown(body),
                title=f"[{_C_GREEN}]Ready to code?[/{_C_GREEN}]",
                border_style=_C_GREEN,
            ))
        except Exception:
            # Never let a rendering hiccup swallow the approval flow.
            console.print(body)

        # The prompt reads the SAME stdin the Esc watcher holds during
        # generation; pause_all() restores cooked/echo termios and stops the
        # watcher reading first, so the user's keystroke isn't eaten. No-op when
        # no watcher is active (tests / non-tty). Ctrl+C / EOF → treat as
        # "revise" rather than crashing the turn.
        from .ui.esc_watcher import pause_all
        try:
            with pause_all():
                choice = Prompt.ask(
                    f"[{_C_YELLOW}]Approve this plan?[/{_C_YELLOW}] "
                    "[y]es / [n]o (revise)",
                    choices=["y", "n"],
                    default="y",
                )
                feedback = ""
                if choice == "n":
                    feedback = Prompt.ask(
                        f"[{_C_DIM}]What should change? (optional)[/{_C_DIM}]",
                        default="",
                    )
        except (KeyboardInterrupt, EOFError):
            return ("Plan rejected: user interrupted before approving. Revise "
                    "and present again.")

        if choice == "y":
            self._plan_state.active = False
            if self._permissions is not None:
                self._permissions.mode = "auto"
            return "Plan approved. Proceed."

        feedback = feedback.strip() or "no feedback given"
        return f"Plan rejected: {feedback}. Revise and present again."
