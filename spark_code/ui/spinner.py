"""Progress spinners and status indicators."""

from contextlib import contextmanager

from rich.console import Console


@contextmanager
def thinking_spinner(console: Console, message: str = "Thinking..."):
    """Show a spinner while the model is generating."""
    with console.status(f"[bold cyan]{message}", spinner="dots"):
        yield


@contextmanager
def tool_spinner(console: Console, tool_name: str):
    """Show a spinner while a tool is executing."""
    with console.status(f"[cyan]⚡ Running {tool_name}...", spinner="dots"):
        yield


def show_progress(console: Console, current: int, total: int, label: str = ""):
    """Show simple progress indicator."""
    pct = (current / total * 100) if total > 0 else 0
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    console.print(f"  {label} [{bar}] {pct:.0f}% ({current}/{total})", end="\r")
