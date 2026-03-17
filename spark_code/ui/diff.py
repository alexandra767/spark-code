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


def render_inline_diff(console: Console, file_path: str,
                       old_string: str, new_string: str,
                       context_lines: int = 3):
    """Show a colored inline diff with context lines from the actual file.

    Reads the file, finds old_string, and shows ±context_lines around
    the change with red for deletions and green for additions.
    """
    # Try to read the file for context
    file_lines = []
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            file_lines = f.readlines()
    except OSError:
        # Fall back to simple diff without file context
        render_diff(console, file_path, old_string, new_string)
        return

    # Find the old_string location in the file
    file_content = "".join(file_lines)
    start_idx = file_content.find(old_string)
    if start_idx == -1:
        # Can't find it — fall back to simple diff
        render_diff(console, file_path, old_string, new_string)
        return

    # Calculate line numbers
    start_line = file_content[:start_idx].count("\n")
    old_line_count = old_string.count("\n") + (1 if old_string and not old_string.endswith("\n") else 0)

    # Build display with context
    text = Text()

    # Header
    text.append(f"  {file_path}\n", style="bold #d8dee9")

    # Before context
    ctx_start = max(0, start_line - context_lines)
    for i in range(ctx_start, start_line):
        line = file_lines[i].rstrip("\n") if i < len(file_lines) else ""
        text.append(f"  {i + 1:4d}  ", style="#4c566a")
        text.append(f"{line}\n", style="#8899aa")

    # Deleted lines (old)
    for j, line in enumerate(old_string.split("\n")):
        if j == old_string.count("\n") and not line:
            continue
        lineno = start_line + j + 1
        text.append(f"  {lineno:4d}  ", style="#4c566a")
        text.append(f"- {line}\n", style="#bf616a")

    # Added lines (new)
    for j, line in enumerate(new_string.split("\n")):
        if j == new_string.count("\n") and not line:
            continue
        text.append("       ", style="#4c566a")
        text.append(f"+ {line}\n", style="#a3be8c")

    # After context
    ctx_end = min(len(file_lines), start_line + old_line_count + context_lines)
    for i in range(start_line + old_line_count, ctx_end):
        line = file_lines[i].rstrip("\n") if i < len(file_lines) else ""
        text.append(f"  {i + 1:4d}  ", style="#4c566a")
        text.append(f"{line}\n", style="#8899aa")

    console.print(Panel(
        text,
        title="[bold #ebcb8b] Proposed Edit [/bold #ebcb8b]",
        border_style="#ebcb8b",
        padding=(0, 1),
    ))


def render_file_created(console: Console, file_path: str, line_count: int):
    """Show that a new file was created."""
    console.print(Panel(
        f"[green]+ {line_count} lines[/green]",
        title=f"[green]New: {file_path}[/green]",
        border_style="green",
    ))
