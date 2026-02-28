"""Output rendering — markdown, syntax highlighting, panels."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


def render_markdown(console: Console, text: str):
    """Render markdown text to terminal."""
    try:
        md = Markdown(text)
        console.print(md)
    except Exception:
        console.print(text)


def render_code(console: Console, code: str, language: str = "python",
                title: str = ""):
    """Render syntax-highlighted code block."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    if title:
        console.print(Panel(syntax, title=title, border_style="blue"))
    else:
        console.print(syntax)


def render_error(console: Console, message: str):
    """Render an error message."""
    console.print(Panel(
        Text(message, style="red"),
        title="[red]Error[/red]",
        border_style="red",
    ))


def render_success(console: Console, message: str):
    """Render a success message."""
    console.print(f"[green]✓[/green] {message}")


def render_warning(console: Console, message: str):
    """Render a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def render_tool_call(console: Console, name: str, args_str: str):
    """Render a tool call being executed."""
    console.print(f"  [cyan]⚡ {name}[/cyan]({args_str})")


def render_tool_result(console: Console, preview: str):
    """Render tool result preview."""
    console.print(f"  [dim]→ {preview}[/dim]")
