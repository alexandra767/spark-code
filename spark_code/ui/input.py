"""Input handling — multi-line support, history, autocomplete."""

import os
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.completion import PathCompleter, WordCompleter, merge_completers


def create_session(history_file: str = "~/.spark/history",
                   skill_names: list[str] | None = None) -> PromptSession:
    """Create a prompt session with history and completions."""
    history_path = os.path.expanduser(history_file)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # Completers
    commands = skill_names or []
    commands.extend(["/help", "/clear", "/compact", "/config", "/model",
                     "/tokens", "/quit", "/exit"])
    command_completer = WordCompleter(commands, sentence=True)
    path_completer = PathCompleter()
    completer = merge_completers([command_completer, path_completer])

    # Key bindings
    bindings = KeyBindings()

    @bindings.add(Keys.Escape, Keys.Enter)
    def _(event):
        """Alt+Enter for newline (multi-line input)."""
        event.current_buffer.insert_text("\n")

    session = PromptSession(
        history=FileHistory(history_path),
        completer=completer,
        key_bindings=bindings,
        multiline=False,
        complete_while_typing=False,
    )

    return session
