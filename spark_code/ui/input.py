"""Input handling — slash-command dropdown, persistent bottom toolbar.

Uses prompt_toolkit's built-in bottom_toolbar for a reliable
always-visible footer below the input, matching Claude Code's layout.
"""

from __future__ import annotations

import os
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
    "/tokens": "Show token usage",
    "/stats": "Show session statistics",
    "/status": "Show session statistics (alias for /stats)",
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
}


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
        if name not in extra_commands:
            extra_commands[name] = ""
    slash_completer = SlashCommandCompleter(commands=extra_commands)
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
