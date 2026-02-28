"""Diff display for file edits."""

import difflib
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def render_diff(console: Console, file_path: str, old_string: str,
                new_string: str):
    """Show a colored diff of the proposed edit."""
    old_lines = old_string.splitlines(keepends=True)
    new_lines = new_string.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )

    text = Text()
    for line in diff:
        line = line.rstrip("\n")
        if line.startswith("+++") or line.startswith("---"):
            text.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        else:
            text.append(line + "\n")

    console.print(Panel(
        text,
        title=f"[yellow]Edit: {file_path}[/yellow]",
        border_style="yellow",
    ))


def render_file_created(console: Console, file_path: str, line_count: int):
    """Show that a new file was created."""
    console.print(Panel(
        f"[green]+ {line_count} lines[/green]",
        title=f"[green]New: {file_path}[/green]",
        border_style="green",
    ))
