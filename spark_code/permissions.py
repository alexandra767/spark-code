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
        if len(command) > 120:
            text.append(command[:117] + "...", style=_C_CMD)
        else:
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
        # Generic fallback — show key: value pairs
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "..."
            text.append(f"  {k}: ", style=_C_DIM)
            text.append(v_str, style=_C_PATH)
            text.append("\n")

    return text


class PermissionManager:
    """Manages tool execution permissions."""

    def __init__(self, mode: str = "ask", always_allow: list[str] | None = None):
        """
        Modes:
          - ask: prompt for every tool call (except always_allow)
          - auto: allow read-only tools, ask for writes
          - trust: allow everything
        """
        self.mode = mode
        self.always_allow = set(always_allow or [])
        self.session_allow = set()  # Tools approved this session
        self.console = Console()

    def check(self, tool_name: str, is_read_only: bool, details: Any = "") -> bool:
        """Check if tool execution is allowed. Returns True if allowed."""
        # Trust mode: always allow
        if self.mode == "trust":
            return True

        # Always-allowed tools
        if tool_name in self.always_allow or tool_name in self.session_allow:
            return True

        # Auto mode: allow read-only
        if self.mode == "auto" and is_read_only:
            return True

        # Ask the user
        return self._prompt_user(tool_name, details)

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

        choice = Prompt.ask(
            "[Y]es / [N]o / [A]lways allow this tool",
            choices=["y", "n", "a"],
            default="y",
        )

        if choice == "a":
            self.session_allow.add(tool_name)
            return True
        return choice == "y"
