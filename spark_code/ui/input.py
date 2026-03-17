"""Input handling — slash-command dropdown, persistent bottom toolbar.

Uses prompt_toolkit's built-in bottom_toolbar for a reliable
always-visible footer below the input, matching Claude Code's layout.
"""

from __future__ import annotations

import os
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
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
    "/tokens": "Show token usage",
    "/stats": "Show session statistics",
    "/diff": "Show git diff with syntax highlighting",
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
    "/publish": "Create a GitHub repo and push (auto-detects name)",
    "/new": "Scaffold a new project with git",
    "/run": "Run the project (auto-detect or specify command)",
    "/team": "Spawn a background worker agent",
    "/tasks": "Show the shared task list",
    "/messages": "Check messages from workers",
    "/history": "List and resume past sessions",
    "/config set": "Set a config value (e.g. /config set model.temperature 0.5)",
    "/undo": "Undo last file write/edit",
    "/pin": "Pin a file to always stay in context",
    "/unpin": "Remove a pinned file",
    "/git": "Smart git: /git sync, /git pr, /git log",
    "/fork": "Branch the conversation (save + start fresh)",
    "/snippet": "Save/run reusable prompts",
    "/export": "Export session as markdown",
    "/quit": "Exit Spark Code",
    "/exit": "Exit Spark Code",
}

# Mode cycle order for Shift+Tab
_MODE_CYCLE = ["ask", "auto", "trust", "plan"]


# ---------------------------------------------------------------------------
# Slash-command completer
# ---------------------------------------------------------------------------

class SlashCommandCompleter(Completer):
    """Completions triggered only when input starts with '/'."""

    def __init__(self, commands: dict[str, str] | None = None):
        super().__init__()
        self._commands: dict[str, str] = dict(_BUILTIN_COMMANDS)
        if commands:
            self._commands.update(commands)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return

        word = text.split()[0] if text.split() else text
        for cmd, desc in sorted(self._commands.items()):
            if cmd.startswith(word):
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
        """Shift+Tab to cycle permission mode: ask → auto → trust → plan."""
        if mode_switch_callback:
            mode_switch_callback()

    return bindings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_session(
    history_file: str = "~/.spark/history",
    skill_names: list[str] | None = None,
    status_callback: Callable[[], AnyFormattedText] | None = None,
    mode_callback: Callable[[], AnyFormattedText] | None = None,
    team_callback: Callable[[], AnyFormattedText] | None = None,
    team_display_callback: Callable | None = None,
    mode_switch_callback: Callable | None = None,
    command_descriptions: dict[str, str] | None = None,
) -> PromptSession:
    """Create a prompt session with slash-command autocomplete and
    a persistent bottom toolbar.

    The toolbar shows up to three lines:
      Line 1: turns + context %
      Line 2: ⏵⏵ mode on · ctrl+t team
      Line 3: team workers (if any active)
    """
    history_path = os.path.expanduser(history_file)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # Build completer
    extra_commands: dict[str, str] = {}
    if command_descriptions:
        extra_commands.update(command_descriptions)
    for name in (skill_names or []):
        if name not in extra_commands:
            extra_commands[name] = ""
    completer = SlashCommandCompleter(commands=extra_commands)

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

    session = PromptSession(
        message=[("class:prompt", "> ")],
        history=FileHistory(history_path),
        completer=completer,
        key_bindings=bindings,
        multiline=False,
        complete_while_typing=True,
        style=INPUT_STYLE,
        bottom_toolbar=bottom_toolbar,
    )

    return session
