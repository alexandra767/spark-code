"""Permission prompt UI components."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..ui.diff import render_diff


def show_edit_permission(console: Console, file_path: str,
                         old_string: str, new_string: str) -> str:
    """Show edit diff and ask for permission. Returns 'y', 'n', or 'a'."""
    render_diff(console, file_path, old_string, new_string)

    console.print("[yellow][Y]es[/yellow] / [red][N]o[/red] / [green][A]lways allow edits[/green]")
    while True:
        choice = console.input("> ").strip().lower()
        if choice in ("y", "yes", ""):
            return "y"
        elif choice in ("n", "no"):
            return "n"
        elif choice in ("a", "always"):
            return "a"
        console.print("[dim]Enter y, n, or a[/dim]")


def show_bash_permission(console: Console, command: str) -> str:
    """Show bash command and ask for permission."""
    console.print(Panel(
        Text(command, style="bold white"),
        title="[yellow]Run Command?[/yellow]",
        border_style="yellow",
    ))
    console.print("[yellow][Y]es[/yellow] / [red][N]o[/red] / [green][A]lways allow bash[/green]")
    while True:
        choice = console.input("> ").strip().lower()
        if choice in ("y", "yes", ""):
            return "y"
        elif choice in ("n", "no"):
            return "n"
        elif choice in ("a", "always"):
            return "a"
        console.print("[dim]Enter y, n, or a[/dim]")


def show_write_permission(console: Console, file_path: str,
                          content_preview: str) -> str:
    """Show file write and ask for permission."""
    lines = content_preview.split("\n")
    preview = "\n".join(lines[:20])
    if len(lines) > 20:
        preview += f"\n... ({len(lines)} total lines)"

    console.print(Panel(
        preview,
        title=f"[yellow]Create: {file_path}[/yellow]",
        border_style="yellow",
    ))
    console.print("[yellow][Y]es[/yellow] / [red][N]o[/red] / [green][A]lways allow writes[/green]")
    while True:
        choice = console.input("> ").strip().lower()
        if choice in ("y", "yes", ""):
            return "y"
        elif choice in ("n", "no"):
            return "n"
        elif choice in ("a", "always"):
            return "a"
        console.print("[dim]Enter y, n, or a[/dim]")
