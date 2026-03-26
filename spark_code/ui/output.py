"""Output rendering — styled to match Claude Code's terminal output.

Uses Nord-inspired colors matching Claude Code's palette:
  - #88c0d0  cyan/teal for tool names, headers
  - #a3be8c  green for success, additions
  - #bf616a  red for errors, deletions
  - #ebcb8b  yellow for warnings
  - #5e81ac  blue for info, keys
  - #d8dee9  light gray for normal text
  - #4c566a  dark gray for dim/connectors
  - #eceff4  bright white for emphasis
  - #2e3440  dark bg for code blocks
"""

import os
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import CodeBlock
from rich.markdown import Markdown as _RichMarkdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONNECTOR = "\u23bf"  # ⎿

_MAX_RESULT_LINES = 5
_MAX_LINE_WIDTH = 120

# Claude Code's color tokens
_C_TOOL = "#88c0d0"       # tool name
_C_TOOL_BOLD = "bold #88c0d0"
_C_TEXT = "#d8dee9"        # normal text
_C_BRIGHT = "#eceff4"     # emphasis
_C_DIM = "#7b88a1"        # dim/connector (brightened for readability)
_C_MUTED = "#8899aa"      # muted info (brightened for readability)
_C_GREEN = "#a3be8c"      # success / additions
_C_RED = "#bf616a"        # error / deletions
_C_YELLOW = "#ebcb8b"     # warning
_C_BLUE = "#5e81ac"       # info / keys
_C_PATH = "#d8dee9"       # file paths
_C_CMD = "#eceff4"        # commands


# ---------------------------------------------------------------------------
# Custom Markdown — code blocks with line numbers & panel border
# ---------------------------------------------------------------------------

class _SparkCodeBlock(CodeBlock):
    """Code blocks with line numbers and a panel border — like Claude Code."""

    def __rich_console__(self, console, options):
        code = str(self.text).rstrip()
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            line_numbers=True,
            word_wrap=True,
            padding=(0, 1),
            background_color="#2e3440",
        )
        yield Panel(
            syntax,
            border_style=_C_DIM,
            padding=(0, 0),
        )


class Markdown(_RichMarkdown):
    """Markdown with enhanced code blocks (line numbers + panel)."""
    elements = {**_RichMarkdown.elements, "fence": _SparkCodeBlock, "code_block": _SparkCodeBlock}


# ---------------------------------------------------------------------------
# Tool-call display names & argument formatting
# ---------------------------------------------------------------------------

_TOOL_LABELS: dict[str, str] = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "bash": "Bash",
    "glob": "Glob",
    "grep": "Grep",
    "list_dir": "List",
    "web_search": "WebSearch",
    "web_fetch": "WebFetch",
}


def _tool_label(name: str) -> str:
    return _TOOL_LABELS.get(name, name)


def _format_tool_args(name: str, args: dict[str, Any]) -> Text:
    """Format tool arguments — Claude Code style per-tool formatting."""
    text = Text()

    if name == "read_file":
        path = args.get("file_path", "")
        text.append(_abbreviate_path(path), style=_C_PATH)
        extras = []
        if args.get("offset"):
            extras.append(f"offset={args['offset']}")
        if args.get("limit"):
            extras.append(f"limit={args['limit']}")
        if extras:
            text.append(f"  ({', '.join(extras)})", style=_C_DIM)

    elif name == "write_file":
        path = args.get("file_path", "")
        content = args.get("content", "")
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        text.append(_abbreviate_path(path), style=_C_PATH)
        text.append(f"  ({line_count} lines)", style=_C_DIM)

    elif name == "edit_file":
        path = args.get("file_path", "")
        text.append(_abbreviate_path(path), style=_C_PATH)
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        old_lines = old.count("\n") + (1 if old else 0)
        new_lines = new.count("\n") + (1 if new else 0)
        text.append("  (", style=_C_DIM)
        text.append(f"-{old_lines}", style=_C_RED)
        text.append(f" +{new_lines}", style=_C_GREEN)
        text.append(" lines)", style=_C_DIM)

    elif name == "bash":
        command = args.get("command", "")
        if len(command) > _MAX_LINE_WIDTH:
            command = command[:_MAX_LINE_WIDTH - 3] + "..."
        text.append(command, style=_C_CMD)
        if args.get("timeout") and args["timeout"] != 120:
            text.append(f"  (timeout={args['timeout']}s)", style=_C_DIM)

    elif name == "grep":
        pattern = args.get("pattern", "")
        text.append(pattern, style=f"bold {_C_BRIGHT}")
        path = args.get("path", "")
        if path:
            text.append("  in ", style=_C_DIM)
            text.append(_abbreviate_path(path), style=_C_PATH)
        file_glob = args.get("glob", "")
        if file_glob:
            text.append(f"  ({file_glob})", style=_C_DIM)

    elif name == "glob":
        pattern = args.get("pattern", "")
        text.append(pattern, style=f"bold {_C_BRIGHT}")
        path = args.get("path", "")
        if path:
            text.append("  in ", style=_C_DIM)
            text.append(_abbreviate_path(path), style=_C_PATH)

    elif name == "list_dir":
        path = args.get("path", args.get("directory", ""))
        text.append(_abbreviate_path(path) if path else ".", style=_C_PATH)

    elif name == "web_search":
        query = args.get("query", "")
        text.append(query, style=_C_BRIGHT)

    elif name == "web_fetch":
        url = args.get("url", "")
        if len(url) > _MAX_LINE_WIDTH:
            url = url[:_MAX_LINE_WIDTH - 3] + "..."
        text.append(url, style=_C_BLUE)

    else:
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "..."
            text.append(f"{k}: ", style=_C_DIM)
            text.append(v_str, style=_C_TEXT)
            text.append("\n")
        if text.plain.endswith("\n"):
            text.right_crop(1)

    return text


def _abbreviate_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


# ---------------------------------------------------------------------------
# Tool-call rendering
# ---------------------------------------------------------------------------

def render_tool_call(console: Console, name: str, args: dict[str, Any] | str = ""):
    """Render a tool invocation — Claude Code style."""
    label = _tool_label(name)

    if isinstance(args, str):
        args_text = Text(args, style=_C_TEXT)
    else:
        args_text = _format_tool_args(name, args)

    header = Text()
    header.append(f"  {label} ", style=_C_TOOL_BOLD)
    header.append_text(args_text)

    console.print(header)


def render_tool_result(console: Console, result: str, tool_name: str = ""):
    """Render tool output with ⎿ connector — colored result lines."""
    if not result:
        console.print(Text(f"  {_CONNECTOR} (no output)", style=_C_DIM))
        return

    lines = result.split("\n")

    # File reads — summary only
    if tool_name in ("read_file",):
        total_lines = len(lines) - 1
        if lines and lines[0].startswith("File:"):
            console.print(Text(f"  {_CONNECTOR} {lines[0]}", style=_C_DIM))
        else:
            console.print(Text(f"  {_CONNECTOR} {total_lines} lines", style=_C_DIM))
        return

    # Edits / writes — confirmation line
    if tool_name in ("edit_file", "write_file"):
        console.print(Text(f"  {_CONNECTOR} {lines[0]}", style=_C_GREEN))
        return

    # Default: show preview lines with syntax-aware coloring
    shown = lines[:_MAX_RESULT_LINES]
    remaining = len(lines) - len(shown)

    for line in shown:
        if len(line) > _MAX_LINE_WIDTH:
            line = line[:_MAX_LINE_WIDTH - 3] + "..."

        t = Text(f"  {_CONNECTOR} ")
        t.stylize(_C_DIM, 0, len(f"  {_CONNECTOR} "))

        # Color diff-like lines
        if line.startswith("+") and not line.startswith("+++"):
            t.append(line, style=_C_GREEN)
        elif line.startswith("-") and not line.startswith("---"):
            t.append(line, style=_C_RED)
        elif line.startswith("@@"):
            t.append(line, style=_C_BLUE)
        elif line.startswith("Error") or line.startswith("error"):
            t.append(line, style=_C_RED)
        else:
            t.append(line, style=_C_MUTED)

        console.print(t)

    if remaining > 0:
        console.print(Text(f"  {_CONNECTOR} ... ({remaining} more lines)", style=_C_DIM))


def render_tool_error(console: Console, name: str, message: str):
    console.print(Text(f"  {_CONNECTOR} {message}", style=_C_RED))


def render_tool_denied(console: Console, name: str):
    console.print(Text(f"  {_CONNECTOR} Permission denied by user", style=_C_RED))


# ---------------------------------------------------------------------------
# Markdown / code rendering
# ---------------------------------------------------------------------------

def render_markdown(console: Console, text: str):
    try:
        md = Markdown(text, code_theme="nord-darker")
        console.print(md)
    except Exception:
        console.print(text)


def render_code(console: Console, code: str, language: str = "python",
                title: str = ""):
    """Render syntax-highlighted code block with nord theme."""
    syntax = Syntax(code, language, theme="nord-darker", line_numbers=True,
                    background_color="#2e3440")
    if title:
        console.print(Panel(syntax, title=f"[{_C_BLUE}]{title}[/{_C_BLUE}]",
                            border_style=_C_DIM))
    else:
        console.print(syntax)


# ---------------------------------------------------------------------------
# Status messages
# ---------------------------------------------------------------------------

def render_error(console: Console, message: str):
    console.print(Panel(
        Text(message, style=_C_RED),
        title=f"[{_C_RED}]Error[/{_C_RED}]",
        border_style=_C_RED,
    ))


def render_success(console: Console, message: str):
    console.print(Text(f"  {message}", style=_C_GREEN))


def render_warning(console: Console, message: str):
    console.print(Text(f"  {message}", style=_C_YELLOW))


def render_info(console: Console, message: str):
    console.print(Text(f"  {message}", style=_C_DIM))


# ---------------------------------------------------------------------------
# Persistent status footer (printed between turns)
# ---------------------------------------------------------------------------

def render_status_footer(console: Console, model_name: str = "",
                         provider: str = "", perm_mode: str = "ask",
                         tokens: int = 0, turns: int = 0,
                         max_tokens: int = 0):
    """Print persistent footer matching Claude Code's two-line layout.

    Line 1: info left  |  context right
    Line 2: ⏵⏵ mode
    """
    width = console.width

    # ── Line 1: info + context ──
    line1_left = Text()
    if turns > 0:
        line1_left.append(f"  {turns} turns", style=_C_MUTED)

    line1_right = Text()
    if max_tokens > 0 and tokens > 0:
        pct = max(0, 100 - (tokens / max_tokens * 100))
        line1_right.append(f"Context left until auto-compact: {pct:.0f}%", style=_C_MUTED)
    elif tokens > 0:
        line1_right.append(f"{tokens:,} tokens", style=_C_MUTED)

    # ── Line 2: mode ──
    line2 = Text()
    line2.append("  ⏵⏵ ", style=_C_GREEN)
    line2.append(f"{perm_mode} mode on", style=_C_TEXT)

    # Print
    console.print(Rule(style=_C_DIM))

    # Line 1 — left + right justified
    pad1 = width - len(line1_left.plain) - len(line1_right.plain) - 2
    if pad1 < 1:
        pad1 = 1
    row1 = Text()
    row1.append_text(line1_left)
    row1.append(" " * pad1)
    row1.append_text(line1_right)
    console.print(row1)

    # Line 2 — mode
    console.print(line2)


# ---------------------------------------------------------------------------
# Streaming renderer
# ---------------------------------------------------------------------------

class StreamingRenderer:
    """Live-updating markdown renderer for streamed text.

    Uses Rich Live display to render markdown in real-time as chunks
    arrive, giving syntax-highlighted code blocks and styled text
    just like Claude Code.

    When live_mode=False (used for background workers), skips the Live
    display entirely — just buffers text and prints on flush. This avoids
    Rich's "Only one live display may be active at once" error.
    """

    def __init__(self, console: Console, live_mode: bool = True):
        self._console = console
        self._buffer: list[str] = []
        self._flushed = False
        self._live: Live | None = None
        self._last_render: float = 0.0
        self._live_mode = live_mode
        self._start_time: float = 0.0

    @property
    def elapsed(self) -> float:
        if self._start_time > 0:
            return time.monotonic() - self._start_time
        return 0.0

    def start(self):
        """Start the live display with a spinner."""
        self._start_time = time.monotonic()
        if not self._live_mode:
            return
        self._live = Live(
            console=self._console,
            refresh_per_second=4,
            transient=True,  # Clear on stop — prevents duplication
            get_renderable=self._get_renderable,
        )
        self._live.start()

    def _make_spinner(self):
        """Create spinner with current elapsed time."""
        elapsed = self.elapsed
        if elapsed > 0.5:
            return Spinner("dots", text=Text(f" Generating... ({elapsed:.1f}s)", style=f"bold {_C_TOOL}"))
        return Spinner("dots", text=Text(" Generating...", style=f"bold {_C_TOOL}"))

    def _get_renderable(self):
        """Called by Live on each refresh cycle."""
        full = "".join(self._buffer)
        if not full.strip():
            return self._make_spinner()
        try:
            return Markdown(full, code_theme="nord-darker")
        except Exception:
            return Text(full)

    def feed(self, chunk: str):
        """Feed a text chunk — updates live markdown display."""
        if self._flushed:
            return
        self._buffer.append(chunk)

        # Throttle renders to ~8/sec to avoid flicker
        now = time.monotonic()
        if now - self._last_render >= 0.125:
            self._last_render = now
            self._render()

    def _render(self):
        """Force a re-render of the live display."""
        if not self._live:
            return
        # Live auto-refreshes via _get_renderable, but we force it
        # on feed() to ensure text appears immediately
        self._live.refresh()

    def stop(self):
        """Stop the live display immediately (e.g. before tool calls)."""
        if self._live:
            self._live.stop()
            self._live = None

    def flush(self):
        """Final render and stop the live display."""
        if self._flushed:
            return
        self._flushed = True

        full_text = "".join(self._buffer)

        # Stop live display first (transient=True clears it)
        if self._live:
            self._live.stop()
            self._live = None

        # Print final markdown cleanly — one time, no duplication
        if full_text.strip():
            try:
                self._console.print(Markdown(full_text, code_theme="nord-darker"))
            except Exception:
                self._console.print(full_text)

    def get_text(self) -> str:
        return "".join(self._buffer)

    def reset(self):
        self._buffer.clear()
        self._flushed = False
        self._live = None
        self._last_render = 0.0
