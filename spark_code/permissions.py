"""Permission system for tool execution."""

import os
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# Nord palette
_C_TOOL = "bold #88c0d0"
_C_DIM = "#7b88a1"
_C_PATH = "#d8dee9"
_C_CMD = "#eceff4"
_C_BRIGHT = "#eceff4"
_C_GREEN = "#a3be8c"
_C_RED = "#bf616a"
_C_BLUE = "#5e81ac"
_C_YELLOW = "#ebcb8b"


def _abbreviate_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def _format_permission_detail(tool_name: str, args: dict[str, Any]) -> Text:
    """Format tool arguments for the permission dialog — clean and readable."""
    text = Text()

    if tool_name == "read_file":
        path = args.get("file_path", "")
        text.append("Read ", style=_C_DIM)
        text.append(_abbreviate_path(path), style=_C_PATH)

    elif tool_name == "write_file":
        path = args.get("file_path", "")
        content = args.get("content", "")
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        text.append("Write ", style=_C_DIM)
        text.append(_abbreviate_path(path), style=_C_PATH)
        text.append(f"  ({line_count} lines)", style=_C_DIM)

    elif tool_name == "edit_file":
        path = args.get("file_path", "")
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        text.append("Edit ", style=_C_DIM)
        text.append(_abbreviate_path(path), style=_C_PATH)
        text.append("\n\n")
        if old:
            for line in old.split("\n")[:5]:
                text.append(f"  - {line}\n", style=_C_RED)
            if old.count("\n") > 5:
                text.append(f"  ... ({old.count(chr(10)) - 5} more lines)\n", style=_C_DIM)
        if new:
            for line in new.split("\n")[:5]:
                text.append(f"  + {line}\n", style=_C_GREEN)
            if new.count("\n") > 5:
                text.append(f"  ... ({new.count(chr(10)) - 5} more lines)\n", style=_C_DIM)

    elif tool_name == "bash":
        command = args.get("command", "")
        text.append("Run ", style=_C_DIM)
        # Show the FULL command — never truncate. Truncating hid any payload
        # after the cutoff at approval time. Rich wraps long lines in the panel.
        text.append(command, style=_C_CMD)

    elif tool_name == "glob":
        pattern = args.get("pattern", "")
        text.append("Search ", style=_C_DIM)
        text.append(pattern, style=f"bold {_C_BRIGHT}")
        path = args.get("path", "")
        if path:
            text.append(" in ", style=_C_DIM)
            text.append(_abbreviate_path(path), style=_C_PATH)

    elif tool_name == "grep":
        pattern = args.get("pattern", "")
        text.append("Search for ", style=_C_DIM)
        text.append(pattern, style=f"bold {_C_BRIGHT}")
        path = args.get("path", "")
        if path:
            text.append(" in ", style=_C_DIM)
            text.append(_abbreviate_path(path), style=_C_PATH)

    elif tool_name == "web_search":
        query = args.get("query", "")
        text.append("Search: ", style=_C_DIM)
        text.append(query, style=_C_BRIGHT)

    elif tool_name == "web_fetch":
        url = args.get("url", "")
        text.append("Fetch: ", style=_C_DIM)
        text.append(url, style=_C_BLUE)

    else:
        # Generic fallback — show key: value pairs. Use a generous cap so a
        # payload isn't hidden at approval time (the old 80-char cut could hide
        # an injection); only extreme blobs are trimmed, with a clear note.
        _MAX_VAL = 4000
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > _MAX_VAL:
                v_str = v_str[:_MAX_VAL] + f"... [+{len(v_str) - _MAX_VAL} more chars]"
            text.append(f"  {k}: ", style=_C_DIM)
            text.append(v_str, style=_C_PATH)
            text.append("\n")

    return text


# Native Spark Code mode names. Shift+Tab only cycles through ask/auto/plan
# (see spark_code.ui.input._MODE_CYCLE) — trust is reachable but not cycled,
# matching Claude Code where bypassPermissions is never in the cycle.
NATIVE_MODES = {"ask", "auto", "trust", "plan"}

# Claude Code's mode names, accepted as aliases (case-insensitive) at every
# mode-parsing site: /mode, /config set permissions.mode, config file load.
MODE_ALIASES = {
    "default": "ask",
    "acceptedits": "auto",
    "bypasspermissions": "trust",
    "plan": "plan",
}


def resolve_mode_name(name: str) -> str | None:
    """Resolve a user/config-supplied mode name to a native mode name.

    Accepts Spark Code's native names (ask/auto/trust/plan) and Claude
    Code's names (default/acceptEdits/bypassPermissions/plan) — matching is
    case-insensitive and ignores surrounding whitespace. Returns None for
    anything unrecognized so callers can fall back to their existing
    "unknown mode" handling instead of silently accepting typos.
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    if not key:
        return None
    if key in NATIVE_MODES:
        return key
    return MODE_ALIASES.get(key)


class PermissionManager:
    """Manages tool execution permissions."""

    def __init__(self, mode: str = "ask", always_allow: list[str] | None = None,
                 interactive: bool = True):
        """
        Modes:
          - ask: prompt for every tool call (except always_allow)
          - auto: allow read-only tools, ask for writes
          - trust: allow everything

        ``interactive`` controls what happens when a decision would otherwise
        require a user prompt. Background workers run on the SHARED event loop,
        where a blocking ``input()`` prompt would freeze the entire session, so
        they build a NON-interactive manager. It must never call
        ``_prompt_user``/``Prompt.ask`` — but it also must not auto-approve
        everything, since that would make a worker equivalent to trust mode
        regardless of its actual mode. Instead it fails safe: it allows exactly
        what the mode would already allow WITHOUT a prompt (trust allows all;
        always_allow/session_allow tools; auto's read-only tools) and DENIES
        anything that would otherwise have required a prompt.
        """
        self.mode = mode
        self.always_allow = set(always_allow or [])
        self.session_allow = set()  # Tools approved this session
        self.interactive = interactive
        self.last_denial_reason: str | None = None
        self.console = Console()

    def allows_without_prompt(self, tool_name: str, is_read_only: bool = False) -> bool:
        """Return True if this tool call is allowed WITHOUT ever prompting.

        Single source of truth for "no prompt needed" — used by BOTH
        check()'s pre-prompt short-circuits and agent.py's parallel-path
        predicate (the tool-call partitioner that decides which calls can
        run concurrently vs. must go through the full sequential/prompt
        pipeline). Before this extraction, that policy lived in two places
        that could silently drift apart — which is exactly how Phase 0's
        worker auto-approve bug happened. False does NOT mean "denied": it
        means "the caller needs to prompt (if interactive) or fail safe by
        denying (if not)".
        """
        # Trust mode: always allow
        if self.mode == "trust":
            return True

        # Always-allowed tools
        if tool_name in self.always_allow or tool_name in self.session_allow:
            return True

        # Auto mode: allow read-only
        if self.mode == "auto" and is_read_only:
            return True

        return False

    def check(self, tool_name: str, is_read_only: bool, details: Any = "") -> bool:
        """Check if tool execution is allowed. Returns True if allowed."""
        self.last_denial_reason = None

        if self.allows_without_prompt(tool_name, is_read_only):
            return True

        # Non-interactive (background worker on the shared loop): never prompt.
        # Fail safe instead of auto-approving: everything reaching this point
        # would have required a prompt, so deny it rather than reach
        # _prompt_user/Prompt.ask (which would block the shared event loop) or
        # silently approve it (which would make the worker equivalent to trust
        # mode regardless of the mode it actually inherited).
        if not self.interactive:
            self.last_denial_reason = (
                f"Permission denied (non-interactive worker, mode={self.mode}): "
                f"{tool_name} requires interactive approval. Report this back "
                "instead of retrying."
            )
            return False

        # Ask the user
        allowed = self._prompt_user(tool_name, details)
        if not allowed:
            self.last_denial_reason = "Permission denied by user."
        return allowed

    def _prompt_user(self, tool_name: str, details: Any) -> bool:
        """Show permission prompt with styled tool details."""
        # Format the detail content
        if isinstance(details, dict):
            detail_text = _format_permission_detail(tool_name, details)
        elif details:
            detail_text = Text(str(details))
        else:
            detail_text = Text()

        # Build panel content
        header = Text()
        header.append(f"  {tool_name}", style=_C_TOOL)
        if detail_text.plain:
            header.append("\n  ")
            header.append_text(detail_text)

        self.console.print(Panel(
            header,
            title=f"[{_C_YELLOW}]Allow tool?[/{_C_YELLOW}]",
            border_style=_C_YELLOW,
        ))

        # Suspend any active Esc watcher for the duration of the prompt. Without
        # this, the watcher's background thread holds stdin in cbreak (echo off)
        # and drains every byte from the same fd every 50 ms — so the user's "y"
        # is swallowed, nothing echoes, and Prompt.ask hangs (or an Esc cancels
        # the whole turn). pause_all() restores cooked/echo termios and stops
        # the watcher reading before Prompt.ask reads, then re-arms on exit.
        from spark_code.ui.esc_watcher import pause_all
        with pause_all():
            choice = Prompt.ask(
                "[y]es / [N]o / [a]lways allow this tool",
                choices=["y", "n", "a"],
                # Default to No so a stray Enter never approves a tool call.
                default="n",
            )

        if choice == "a":
            self.session_allow.add(tool_name)
            return True
        return choice == "y"
