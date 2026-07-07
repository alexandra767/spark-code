"""Input handling — slash-command dropdown, persistent bottom toolbar.

Uses prompt_toolkit's built-in bottom_toolbar for a reliable
always-visible footer below the input, matching Claude Code's layout.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, merge_completers
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

# ---------------------------------------------------------------------------
# Builtin commands
# ---------------------------------------------------------------------------

_BUILTIN_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/clear": "Clear conversation history",
    "/compact": "Summarize conversation to save context",
    "/config": "Show current configuration",
    "/model": "Show model info or switch provider (/model <provider>)",
    "/model list": "List available providers",
    "/providers": "Show API providers with signup URLs",
    "/setkey": "Guided cloud API key setup (masked prompt) — /setkey <provider>",
    "/tokens": "Show token usage",
    "/stats": "Show session statistics",
    "/status": "Show session statistics (alias for /stats)",
    "/diff": "Show git diff with syntax highlighting",
    "/review": "Multi-agent review swarm on the diff (find + skeptic-verify)",
    "/memory": "View or add to memory",
    "/image": "Send an image file with a prompt",
    "/mode": "Switch permission mode (ask/auto/trust/plan)",
    "/yolo": "Toggle agent mode (autonomous + trust all)",
    "/trust": "Switch to trust mode (allow all)",
    "/auto": "Switch to auto mode (allow reads)",
    "/ask": "Switch to ask mode (confirm all)",
    "/plan": "Create a plan before executing",
    "/plan show": "Show the current plan",
    "/plan copy": "Copy plan to clipboard",
    "/plan go": "Execute the approved plan",
    "/projectplan": "Create a RAG-researched project plan",
    "/projectplan show": "Show the current project plan",
    "/projectplan copy": "Copy project plan to clipboard",
    "/projectplan go": "Execute the approved project plan",
    "/publish": "Create a GitHub repo and push (auto-detects name)",
    "/new": "Scaffold a new project with git",
    "/init": "Generate a CLAUDE.md for this project",
    "/run": "Run the project (auto-detect or specify command)",
    "/team": "Spawn a background worker agent",
    "/tasks": "Show the shared task list",
    "/messages": "Check messages from workers",
    "/history": "List and resume past sessions",
    "/resume": "Pick a past session to resume",
    "/config set": "Set a config value (e.g. /config set model.temperature 0.5)",
    "/retry": "Re-send the last message",
    "/undo": "Undo last file write/edit",
    "/pin": "Pin a file to always stay in context",
    "/unpin": "Remove a pinned file",
    "/open": "Open a file (optionally file:line) in your editor",
    "/git": "Smart git: /git sync, /git pr, /git log",
    "/fork": "Branch the conversation (save + start fresh)",
    "/snippet": "Save/run reusable prompts",
    "/export": "Export session as markdown",
    "/cost": "Show session costs and budget",
    "/watch": "Auto-run a command on file changes",
    "/watch off": "Stop the file watcher",
    "/checkpoint": "Create a git stash checkpoint",
    "/rollback": "Restore from a checkpoint",
    "/rewind": "Restore files from undo stack or checkpoint",
    "/continue": "Resume from last checkpoint",
    "/clean": "Delete files created this session",
    "/profile": "Benchmark model performance (TTFT, tokens/sec)",
    "/benchmark": "Measure model speed (time-to-first-token, tok/s)",
    "/apply": "Apply code from a URL (gist, PR, diff)",
    "/teach": "Define a custom tool (/teach name desc -- command)",
    "/branch": "Create a conversation branch",
    "/switch": "Switch to a different branch",
    "/branches": "List all conversation branches",
    "/share": "Export session as shareable HTML",
    "/share gist": "Export session as markdown for gist",
    "/search": "Search past sessions by keyword",
    "/analytics": "Show detailed session analytics dashboard",
    "/quit": "Exit Spark Code",
    "/exit": "Exit Spark Code",
    "/q": "Exit Spark Code (alias for /quit)",
    "/docs": "Search indexed docs (Swift, SwiftUI, HIG, App Store Guidelines, CNN)",
    "/index": "Index this project into the RAG service for code_search",
    "/doctor": "Health check: engine, RAG, MCP, editor, keys (read-only)",
}

# Bare command names (first token, no "/") reserved by the real commands
# above — subcommand entries ("/plan show", "/config set") collapse into
# their command. Skills whose names collide with these are excluded from the
# model-facing surfaces (the system-prompt skill index built in cli.py and
# agent.py's auto-expand) and from clobbering command descriptions in the
# autocomplete merge below: handle_slash_command resolves these names to the
# COMMAND (checked first), so advertising a same-named skill to the model
# would teach it a name that expands to something other than what the user
# gets at the CLI (e.g. the /review swarm vs. the lesser builtin `review`
# skill).
RESERVED_COMMAND_NAMES = frozenset(
    cmd.split()[0].lstrip("/") for cmd in _BUILTIN_COMMANDS
)


# ---------------------------------------------------------------------------
# Slash-command completer
# ---------------------------------------------------------------------------

class SlashCommandCompleter(Completer):
    """Completions triggered only when input starts with '/'."""

    def __init__(
        self,
        commands: dict[str, str] | None = None,
        value_providers: dict[str, Callable[[], list[str]]] | None = None,
    ):
        super().__init__()
        self._commands: dict[str, str] = dict(_BUILTIN_COMMANDS)
        if commands:
            self._commands.update(commands)
        # Zero-arg callables, keyed by command name (e.g. "/model"), returning
        # candidate argument strings — called lazily per keystroke (Phase 3
        # Task 1) so a slow or failing provider (e.g. session listing, a
        # config read) never blocks typing.
        self._value_providers: dict[str, Callable[[], list[str]]] = dict(
            value_providers or {}
        )

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return

        # Argument completion: "<command> <partial-arg>" where <command> has
        # a registered value provider. This must be checked before the
        # whole-phrase command matching below and must compute
        # start_position from the PARTIAL ARG's length only — never the
        # full text — or accepting a completion corrupts the command portion
        # the same way the subcommand bug below once did (e.g. "/model co"
        # must become "/model coder", not "/mod/model coder").
        if " " in text:
            cmd, _, rest = text.partition(" ")
            provider = self._value_providers.get(cmd)
            if provider is not None:
                partial = rest.rsplit(" ", 1)[-1]
                try:
                    candidates = provider()
                except Exception:
                    candidates = []
                for value in candidates:
                    if value.startswith(partial) and value != partial:
                        yield Completion(
                            text=value,
                            start_position=-len(partial),
                            display=FormattedText([
                                ("class:completion.command", f" {value} "),
                            ]),
                        )
                return

        # Match against the ENTIRE slash phrase typed so far, not just the first
        # token — otherwise completing "/plan sh" → "/plan show" replaces only
        # the first token's worth of characters and produces "/pl/plan show".
        # For multi-word command entries ("/plan show", "/config set") this
        # lets subcommands complete correctly.
        word = text
        for cmd, desc in sorted(self._commands.items()):
            if cmd.startswith(word) and cmd != word:
                yield Completion(
                    text=cmd,
                    start_position=-len(word),
                    display=FormattedText([
                        ("class:completion.command", f" {cmd} "),
                        ("class:completion.description", f" {desc} "),
                    ]),
                    display_meta=desc,
                )


# ---------------------------------------------------------------------------
# File path completer
# ---------------------------------------------------------------------------

class FilePathCompleter(Completer):
    """Completes file paths, including absolute paths.

    Defers to the slash-command completer ONLY when the user is typing a lone
    slash-command token (e.g. "/plan"). An absolute path ("/Users/..") or a
    path argument after other text still completes.

    Also activates on Claude Code-style `@path` mentions (e.g. "@src/ap"):
    the path portion after the leading "@" is completed against the
    filesystem, and the "@" is re-prepended to the completion text so the
    buffer becomes "@src/app.py" rather than losing the marker.
    """

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()
        if not words:
            return
        last_word = words[-1]

        # A single token that looks like a slash command ("/word", exactly one
        # "/") belongs to the slash-command completer. An absolute path picks up
        # a second "/" quickly ("/Users/"), at which point we complete it.
        if (text.lstrip().startswith("/") and len(words) == 1
                and last_word.count("/") == 1):
            return

        at_mention = last_word.startswith("@")
        path_word = last_word[1:] if at_mention else last_word

        # Only trigger on things that look like paths (an "@" mention always
        # qualifies, even before any "/" or "." has been typed).
        if not at_mention and not (
            "/" in last_word or "." in last_word or last_word.startswith("~")
        ):
            return

        # Expand ~ and resolve the directory part
        expanded = os.path.expanduser(path_word)
        if os.path.isdir(expanded):
            directory = expanded
            prefix = ""
        else:
            directory = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)

        if not os.path.isdir(directory):
            return

        try:
            entries = os.listdir(directory)
        except OSError:
            return

        # For a bare "@name" (no "/" typed yet after the "@"), the "@" itself
        # must be replaced along with the prefix since the completion text
        # re-inserts it. Once a "/" has been typed ("@src/ap"), the "@" sits
        # before the directory portion already in the buffer and is left
        # alone — only the basename ("ap") is replaced.
        at_prefix = "@" if (at_mention and "/" not in path_word) else ""
        replace_len = len(prefix) + len(at_prefix)

        for entry in sorted(entries):
            if entry.startswith(".") and not prefix.startswith("."):
                continue
            if prefix and not entry.lower().startswith(prefix.lower()):
                continue
            full = os.path.join(directory, entry)
            is_dir = os.path.isdir(full)
            display_suffix = "/" if is_dir else ""
            completion = at_prefix + entry + display_suffix
            yield Completion(
                text=completion,
                start_position=-replace_len,
                display=FormattedText([
                    ("class:completion.command",
                     f" {completion} "),
                ]),
            )


# ---------------------------------------------------------------------------
# Style — Nord palette matching Claude Code
# ---------------------------------------------------------------------------

INPUT_STYLE = Style.from_dict({
    # Prompt
    "prompt": "fg:#5e81ac bold",

    # Bottom toolbar (the persistent footer)
    "bottom-toolbar": "bg:#2e3440 fg:#7b88a1",
    "bottom-toolbar.text": "fg:#7b88a1",
    "bottom-toolbar.info": "fg:#8899aa",
    "bottom-toolbar.mode": "fg:#a3be8c bold",
    "bottom-toolbar.mode-text": "fg:#d8dee9",
    "bottom-toolbar.context": "fg:#8899aa",
    # Context-left meter, colored by how much room remains (green/yellow/red
    # mirror the worker-status colors below for a consistent palette).
    "bottom-toolbar.context-green": "fg:#a3be8c",
    "bottom-toolbar.context-yellow": "fg:#ebcb8b",
    "bottom-toolbar.context-red": "fg:#bf616a bold",

    # Team status in toolbar
    "bottom-toolbar.team": "fg:#88c0d0 bold",
    "bottom-toolbar.team-text": "fg:#88c0d0",
    "bottom-toolbar.worker-running": "fg:#ebcb8b",
    "bottom-toolbar.worker-done": "fg:#a3be8c",
    "bottom-toolbar.worker-failed": "fg:#bf616a",
    "bottom-toolbar.worker-name": "fg:#d8dee9",
    "bottom-toolbar.worker-task": "fg:#8899aa",

    # Completion menu
    "completion-menu": "bg:#2e3440 fg:#d8dee9",
    "completion-menu.completion": "bg:#2e3440 fg:#d8dee9",
    "completion-menu.completion.current": "bg:#434c5e fg:#eceff4",
    "completion.command": "fg:#88c0d0 bold",
    "completion.description": "fg:#8899aa",
})


# ---------------------------------------------------------------------------
# Key bindings
# ---------------------------------------------------------------------------

def _create_bindings(
    team_display_callback: Callable | None = None,
    mode_switch_callback: Callable | None = None,
) -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add(Keys.Escape, Keys.Enter)
    def _newline(event):
        """Escape+Enter for newline."""
        event.current_buffer.insert_text("\n")

    @bindings.add("c-t")
    def _show_team(event):
        """Ctrl+T to show team status."""
        if team_display_callback:
            team_display_callback()

    @bindings.add(Keys.BackTab)
    def _cycle_mode(event):
        """Shift+Tab to cycle permission mode: ask → auto → plan → ask.

        Trust (bypassPermissions) is deliberately not in this cycle — same
        as Claude Code, it's only entered explicitly (/trust, --trust,
        /mode trust), never landed on by repeated Shift+Tab presses.
        """
        if mode_switch_callback:
            mode_switch_callback()

    return bindings


# ---------------------------------------------------------------------------
# Non-CPR terminal fallback (Phase 3 Task 3)
# ---------------------------------------------------------------------------
#
# Investigation findings (spike scripts against the installed prompt_toolkit
# 3.0.52, see the task report for the full pexpect transcripts):
#
# 1. The warning ("WARNING: your terminal doesn't support cursor position
#    requests (CPR).") is printed by `Application.cpr_not_supported_callback`
#    (prompt_toolkit/application/application.py), invoked by the `Renderer`
#    (renderer.py) exactly once: when a CPR request was sent, `Output.
#    responds_to_cpr` was True at the time, and no CPR response arrived
#    within `Renderer.CPR_TIMEOUT` (2 seconds). `Renderer` captures this
#    callback (a bound method of `Application`) as a plain, mutable instance
#    attribute (`self.cpr_not_supported_callback = ...`) at construction time
#    — there is no `Application`/`PromptSession` constructor kwarg to inject
#    a replacement, but the attribute on the already-built `Renderer` (reachable
#    as ``session.app.renderer`` immediately after `PromptSession.__init__`,
#    confirmed empirically) can be overwritten post-construction. Doing so
#    neuters *only* the terminal print — the underlying `renderer.cpr_support`
#    state machine (UNKNOWN → SUPPORTED/NOT_SUPPORTED) is untouched, so this
#    is safe for CPR-capable terminals: there, CPR resolves to SUPPORTED well
#    within the 2s window and the callback is never invoked at all.
#
# 2. The bottom toolbar's invisibility in a non-CPR pty is a SEPARATE,
#    deeper prompt_toolkit behavior, not just the warning: prompt_toolkit's
#    `bottom_toolbar` container is wrapped in
#    ``Condition(lambda: ...) & renderer_height_is_known`` (shortcuts/prompt.py),
#    and `Renderer.height_is_known()` on a vt100 `Output` (i.e. every real
#    Unix pty, not just pexpect's) has **no path to True except a successful
#    CPR response** (`get_rows_below_cursor_position()` is unimplemented for
#    vt100 and always raises `NotImplementedError`). This was confirmed with
#    a bare `PromptSession(bottom_toolbar=...)` driven by pexpect: the toolbar
#    text never appears in the raw pty byte stream, with or without the
#    warning-callback neutered above. In other words: on any vt100 terminal,
#    prompt_toolkit's bottom toolbar is *itself* gated on CPR actually being
#    answered — real terminals answer it in well under 2s (so the toolbar
#    appears almost immediately), a pty like pexpect's never does (so it
#    never appears), and this is not something Spark can force from the
#    outside without reimplementing prompt_toolkit's renderer.
#
# 3. Consequence for `terminal_supports_cpr()`: there is no pre-prompt query
#    that can distinguish "a real terminal that will answer CPR" from "a pty
#    that merely *presents* as a tty (`isatty()` True, a normal $TERM) but
#    will never answer" — both look identical to every signal prompt_toolkit
#    itself exposes (`Output.responds_to_cpr` is fooled the same way; this is
#    exactly the documented Phase 2 acceptance gap). The heuristic below can
#    only catch the DEFINITE non-CPR cases cheaply (no tty at all, `TERM=dumb`,
#    the escape hatches). The AMBIGUOUS case (pexpect and similar) is instead
#    resolved at *runtime*: `create_session()` wires the neutered callback in
#    (1) above to also flip a caller-supplied `cpr_state` dict, so callers can
#    react to "CPR turned out not to work" a couple of seconds into the first
#    prompt — long before a real engine turn (seconds of generation) finishes
#    — and print the inline status fallback from then on, self-healing
#    "auto" mode without a false a-priori guess.
def terminal_supports_cpr(
    env: "dict[str, str] | None" = None,
    isatty: Callable[[], bool] | None = None,
) -> bool:
    """Best-effort PRE-FLIGHT heuristic for whether this terminal is likely
    to answer prompt_toolkit's CPR probe. Mirrors prompt_toolkit's own
    ``Output.responds_to_cpr`` checks (same env vars, same dumb-terminal and
    tty gates) plus Spark's own ``SPARK_NO_CPR=1`` escape hatch.

    Deliberately optimistic on ambiguous input (returns True) — see the
    module-level notes above for why a tty-presenting-but-non-answering pty
    (pexpect) cannot be caught here and is instead handled via runtime
    confirmation in ``create_session``.
    """
    env = os.environ if env is None else env
    if env.get("SPARK_NO_CPR") == "1":
        return False
    if env.get("PROMPT_TOOLKIT_NO_CPR") == "1":
        return False
    if env.get("TERM", "") == "dumb":
        return False
    _isatty = isatty if isatty is not None else (lambda: sys.stdout.isatty())
    try:
        return bool(_isatty())
    except Exception:
        return False


def resolve_statusline_mode(configured: str, cpr_supported: bool) -> tuple[bool, bool]:
    """Route the ``ui.statusline`` config value (+ a CPR-support signal) to
    ``(use_toolbar, use_inline)``. Pure — no session/renderer access — so it's
    the single tested source of truth for both call sites that need it:
    ``create_session`` (deciding whether to attach the toolbar widget at all,
    fed the pre-flight heuristic) and the CLI's per-turn loop (deciding
    whether to print the inline fallback, fed the live, possibly-since-updated
    ``cpr_state``).

    - "off": neither surface.
    - "toolbar": always the toolbar, never inline (explicit user override —
      if CPR genuinely doesn't work the status surface is lost, same as
      before this task; that's the user's own explicit choice).
    - "inline": always the one-line fallback, never the toolbar.
    - "auto" (default, and the fallback for any unrecognized value): toolbar
      when CPR is expected to work, inline otherwise.
    """
    if configured == "off":
        return False, False
    if configured == "toolbar":
        return True, False
    if configured == "inline":
        return False, True
    return cpr_supported, not cpr_supported


def build_status_segments(
    mode: str | None,
    *,
    model_name: str = "",
    provider_name: str = "",
    turns: int = 0,
    context_pct: float | None = None,
    context_style: str = "class:bottom-toolbar.context",
    context_tokens: int | None = None,
    model_prefix: str = "",
    mode_prefix: str = "⏵⏵ ",
    mode_suffix: str = " mode",
    turns_prefix: str = "  ·  ",
    emit_empty_turns: bool = False,
    context_prefix: str = "  ·  ",
    extra_segments: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Pure function: THE single source of status-segment content, shared by
    the inline non-CPR fallback (knob defaults) and the bottom toolbar's two
    lines (via the ``toolbar_status_segments``/``toolbar_mode_segments``
    wrappers below — review fix: previously the cli.py toolbar closures
    duplicated this logic with slightly different formatting). Every input is
    a plain value the caller already computed (``context.estimate_tokens()``,
    ``permissions.mode``, the pct-to-style mapping) — no Context/session
    objects, no I/O — so this is directly unit-testable without constructing
    any prompt_toolkit or Spark session state.

    Segment order is fixed: model → mode → turns → extra_segments → context.
    Presentation knobs (all default to the inline fallback's rendering):

    - ``mode=None`` omits the mode segments entirely (toolbar line 1).
    - ``model_prefix``/``mode_prefix``/``mode_suffix`` — leading pad and the
      " mode" vs " mode on" suffix difference.
    - ``turns_prefix`` + ``emit_empty_turns`` — toolbar line 1 emits an
      EMPTY info segment when turns == 0 (and no separator before the
      count); the inline flavor omits the segment and separates with "  ·  ".
    - ``context_prefix`` — "  ·  " inline vs the toolbar's 10-space gutter.
    - ``context_tokens`` — raw-count fallback segment when no percentage is
      computable (toolbar line 1 only).
    - ``extra_segments`` — pre-formatted (style, text) pairs appended between
      turns and context: speed/cost on line 1, the shift+tab hint and team
      status on line 2.
    """
    parts: list[tuple[str, str]] = []
    if model_name:
        label = f"{model_prefix}{model_name}"
        if provider_name:
            label += f" ({provider_name})"
        parts.append(("class:bottom-toolbar.team", label))
        parts.append(("class:bottom-toolbar.info", "  "))
    if mode is not None:
        parts.append(("class:bottom-toolbar.mode", mode_prefix))
        parts.append(("class:bottom-toolbar.mode-text", f"{mode}{mode_suffix}"))
    if turns > 0:
        parts.append(("class:bottom-toolbar.info", f"{turns_prefix}{turns} turns"))
    elif emit_empty_turns:
        parts.append(("class:bottom-toolbar.info", ""))
    if extra_segments:
        parts.extend(extra_segments)
    if context_pct is not None:
        parts.append((context_style, f"{context_prefix}ctx {int(context_pct)}%"))
    elif context_tokens:
        parts.append(
            ("class:bottom-toolbar.context", f"    {context_tokens:,} tokens")
        )
    return parts


def toolbar_status_segments(
    *,
    model_name: str = "",
    provider_name: str = "",
    turns: int = 0,
    speed_str: str = "",
    cost_str: str = "",
    context_pct: float | None = None,
    context_style: str = "class:bottom-toolbar.context",
    context_tokens: int | None = None,
) -> list[tuple[str, str]]:
    """Toolbar line 1 (model · turns · speed · cost · ctx%) — the exact
    rendering cli.py's ``status_callback`` closure produced before the
    one-source refactor, golden-tested byte-identical in
    tests/test_terminal_compat.py::TestToolbarGoldens.
    """
    extra: list[tuple[str, str]] = []
    if speed_str:
        extra.append(("class:bottom-toolbar.info", f"  {speed_str}"))
    if cost_str:
        extra.append(("class:bottom-toolbar.info", f"  {cost_str}"))
    return build_status_segments(
        None,
        model_name=model_name,
        provider_name=provider_name,
        turns=turns,
        context_pct=context_pct,
        context_style=context_style,
        context_tokens=context_tokens,
        model_prefix="  ",
        turns_prefix="",
        emit_empty_turns=True,
        context_prefix=" " * 10,
        extra_segments=extra,
    )


def toolbar_mode_segments(
    display_mode: str,
    *,
    team_active: int = 0,
    team_total: int = 0,
) -> list[tuple[str, str]]:
    """Toolbar line 2 (⏵⏵ mode on · shift+tab to switch [· ctrl+t team]) —
    the exact rendering cli.py's ``mode_callback`` closure produced before
    the one-source refactor, golden-tested byte-identical in
    tests/test_terminal_compat.py::TestToolbarGoldens.
    """
    extra: list[tuple[str, str]] = [
        ("class:bottom-toolbar.info", "  ·  "),
        ("class:bottom-toolbar.info", "shift+tab to switch"),
    ]
    if team_total:
        extra.append(("class:bottom-toolbar.info", "  ·  "))
        extra.append(("class:bottom-toolbar.team", "ctrl+t "))
        extra.append(
            ("class:bottom-toolbar.team-text", f"team ({team_active}/{team_total})")
        )
    return build_status_segments(
        display_mode,
        mode_prefix="  ⏵⏵ ",
        mode_suffix=" mode on",
        extra_segments=extra,
    )


def renderer_cpr_support(session) -> bool | None:
    """Live read of prompt_toolkit's runtime CPR verdict for this session's
    renderer (review fix for the fast-turn race): ``True`` once a CPR
    response has actually been observed (``CPR_Support.SUPPORTED``),
    ``False`` once prompt_toolkit concluded no answer is coming
    (``NOT_SUPPORTED`` — its 2s timer fired, or ``responds_to_cpr`` was
    already False at construction), ``None`` while still ``UNKNOWN``.

    UNKNOWN is not a transient startup blip to ignore: the 2s CPR timer is a
    background task of the prompt Application, so a prompt submitted in
    under 2 seconds cancels it — in a fast scripted non-CPR pty the verdict
    can stay UNKNOWN forever and the neutered-callback latch never fires.
    The inline-status decision therefore treats UNKNOWN pessimistically
    (see ``cpr_confirmed_supported``). Any read failure also maps to None.
    """
    try:
        from prompt_toolkit.renderer import CPR_Support

        support = session.app.renderer.cpr_support
        if support == CPR_Support.SUPPORTED:
            return True
        if support == CPR_Support.NOT_SUPPORTED:
            return False
        return None
    except Exception:
        return None


def cpr_confirmed_supported(live: bool | None, latched_unsupported: bool) -> bool:
    """Pessimistic merge for the inline-status decision (review fix): CPR
    counts as supported ONLY once positively observed (``live is True``) and
    the neutered-callback latch never fired. UNKNOWN (None) is treated as
    unsupported — a brief cosmetic status duplication in good terminals
    during the first fast turns beats an invisible status surface in bad
    ones. The latch stays authoritative as a secondary signal even against a
    live SUPPORTED reading.
    """
    if latched_unsupported:
        return False
    return live is True


def format_status_line(segments: list[tuple[str, str]]) -> str:
    """Flatten styled ``(style_class, text)`` segments into the plain-text
    line printed above the prompt in inline-fallback mode. Style classes only
    matter for prompt_toolkit's toolbar rendering; the inline fallback is a
    normal terminal line with no styling machinery of its own."""
    return "".join(text for _, text in segments).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_session(
    history_file: str = "~/.spark/input_history",
    skill_names: list[str] | None = None,
    status_callback: Callable[[], AnyFormattedText] | None = None,
    mode_callback: Callable[[], AnyFormattedText] | None = None,
    team_callback: Callable[[], AnyFormattedText] | None = None,
    team_display_callback: Callable | None = None,
    mode_switch_callback: Callable | None = None,
    command_descriptions: dict[str, str] | None = None,
    value_providers: dict[str, Callable[[], list[str]]] | None = None,
    statusline: str = "auto",
    cpr_state: dict | None = None,
) -> PromptSession:
    """Create a prompt session with slash-command autocomplete and
    a persistent bottom toolbar.

    The toolbar shows up to three lines:
      Line 1: turns + context %
      Line 2: ⏵⏵ mode on · ctrl+t team
      Line 3: team workers (if any active)

    ``statusline`` (Phase 3 Task 3 — see the module notes above
    ``terminal_supports_cpr`` for the full investigation) routes via
    ``resolve_statusline_mode``: "auto" (default) attaches the toolbar only
    when CPR looks likely to work (``terminal_supports_cpr()``); "toolbar"
    always attaches it; "inline"/"off" never do (the caller is expected to
    print the one-line fallback itself — this function only owns the
    toolbar widget, not the inline print, since the latter happens between
    prompts, outside this session object).

    Regardless of ``statusline``, prompt_toolkit's own "doesn't support
    cursor position requests (CPR)" warning print is always neutered here
    (see the investigation notes above ``terminal_supports_cpr``): it is a
    strict improvement with no effect on CPR-capable terminals (there, CPR
    resolves before prompt_toolkit's internal 2s timeout ever fires the
    callback). When ``cpr_state`` is provided, the neutered callback also
    flips ``cpr_state["supported"] = False`` — the runtime confirmation
    signal a caller needs to react to the non-CPR case (e.g. pexpect) that
    the pre-flight heuristic cannot detect in advance.
    """
    history_path = os.path.expanduser(history_file)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # Legacy migration: the input history used to live at ~/.spark/history,
    # which collides with the sessions DIRECTORY of the same name. If that old
    # file is still present, move it to the new path (once) so both features work.
    legacy = os.path.expanduser("~/.spark/history")
    if os.path.isfile(legacy) and not os.path.exists(history_path):
        try:
            os.rename(legacy, history_path)
        except OSError:
            pass

    # Build completer
    extra_commands: dict[str, str] = {}
    if command_descriptions:
        extra_commands.update(command_descriptions)
    for name in (skill_names or []):
        # A skill can share its name with a real command (e.g. the builtin
        # `review` SKILL vs. the `/review` COMMAND — handle_slash_command
        # always resolves that in the command's favor). Adding it here with
        # an empty description would let SlashCommandCompleter.__init__'s
        # ``self._commands.update(commands)`` clobber the real command's rich
        # description from _BUILTIN_COMMANDS with "" — same "real commands
        # win" precedence, just for autocomplete instead of dispatch.
        if name not in extra_commands and name not in _BUILTIN_COMMANDS:
            extra_commands[name] = ""
    slash_completer = SlashCommandCompleter(
        commands=extra_commands, value_providers=value_providers
    )
    file_completer = FilePathCompleter()
    completer = merge_completers([slash_completer, file_completer])

    bindings = _create_bindings(
        team_display_callback=team_display_callback,
        mode_switch_callback=mode_switch_callback,
    )

    # Bottom toolbar — combines callbacks into a multi-line footer
    def bottom_toolbar():
        parts: list[tuple[str, str]] = []

        # Line 1: status info
        if status_callback:
            try:
                result = status_callback()
                if isinstance(result, list):
                    parts.extend(result)
                else:
                    parts.append(("class:bottom-toolbar.info", str(result)))
            except Exception:
                pass

        parts.append(("", "\n"))

        # Line 2: mode + team hint
        if mode_callback:
            try:
                result = mode_callback()
                if isinstance(result, list):
                    parts.extend(result)
                else:
                    parts.append(("class:bottom-toolbar.mode-text", str(result)))
            except Exception:
                pass

        # Line 3: team workers (only if there are any)
        if team_callback:
            try:
                result = team_callback()
                if result:
                    parts.append(("", "\n"))
                    if isinstance(result, list):
                        parts.extend(result)
                    else:
                        parts.append(("class:bottom-toolbar.team-text", str(result)))
            except Exception:
                pass

        return parts

    # SPARK_NO_CPR=1 is Spark's own escape hatch; also set prompt_toolkit's
    # native equivalent (if not already set) so the underlying library never
    # even attempts the CPR round-trip — belt (this) and suspenders (the
    # cpr_not_supported_callback neutering below, which covers the case this
    # can't: a pty that presents as CPR-capable but never answers).
    if os.environ.get("SPARK_NO_CPR") == "1":
        os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

    # Initial toolbar-attachment decision: consume the caller's cpr_state
    # (which run_interactive seeds with the pre-flight heuristic) rather
    # than redundantly re-running the heuristic here (review minor); fall
    # back to running it ourselves only when no state was passed.
    if cpr_state is not None and "supported" in cpr_state:
        cpr_likely = bool(cpr_state["supported"])
    else:
        cpr_likely = terminal_supports_cpr()
    use_toolbar, _ = resolve_statusline_mode(statusline, cpr_likely)

    session = PromptSession(
        message=[("class:prompt", "> ")],
        history=FileHistory(history_path),
        completer=completer,
        key_bindings=bindings,
        multiline=False,
        complete_while_typing=True,
        style=INPUT_STYLE,
        bottom_toolbar=bottom_toolbar if use_toolbar else None,
    )

    # Neuter prompt_toolkit's own CPR-not-supported warning print (see the
    # module docstring above terminal_supports_cpr for the full mechanism).
    # `session.app` (and its `.renderer`) already exist at this point —
    # `PromptSession.__init__` builds the `Application` synchronously.
    def _on_cpr_not_supported() -> None:
        if cpr_state is not None:
            cpr_state["supported"] = False

    session.app.renderer.cpr_not_supported_callback = _on_cpr_not_supported

    return session
