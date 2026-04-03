"""Spark Code CLI — entry point."""

import asyncio
import base64
import mimetypes
import os
import subprocess
import sys

import click
from rich.box import ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import Agent
from .branches import BranchManager
from .config import ensure_dirs, get, load_config, set_config
from .context import AGENTIC_PROMPT, SYSTEM_PROMPT, Context
from .custom_tools import CustomToolRegistry
from .hooks import HookManager
from .mcp.client import MCPClient
from .mcp.registry import find_mcp_configs
from .memory import Memory
from .model import PROVIDERS, ModelClient
from .permissions import PermissionManager
from .pinned import PinnedFiles
from .plan_executor import execute_plan
from .projectplan import extract_keywords, fetch_rag_context
from .project_detect import detect_project_type
from .skills.base import SkillRegistry
from .snippets import SnippetLibrary
from .stats import SessionStats
from .task_store import TaskStore
from .team import TeamManager
from .tool_cache import ToolCache
from .tools.base import ToolRegistry
from .tools.bash import BashTool
from .tools.edit_file import EditFileTool
from .tools.glob_search import GlobTool
from .tools.grep_search import GrepTool
from .tools.list_dir import ListDirTool
from .tools.read_file import ReadFileTool
from .tools.spawn_worker import SpawnWorkerTool
from .tools.wait_for_workers import WaitForWorkersTool
from .tools.web_fetch import WebFetchTool
from .tools.web_search import WebSearchTool
from .tools.write_file import WriteFileTool
from .tools.rag_search import RagSearchTool
from .ui.hotkeys import TeamStatusMonitor
from .ui.input import create_session
from .ui.theme import get_theme
from .platform_info import format_platform_prompt
from .watcher import FileWatcher


async def _warmup_model(model):
    """Send a tiny request to force model into VRAM."""
    try:
        async for _ in model.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[], stream=True
        ):
            break  # One chunk confirms model loaded
    except Exception:
        pass  # Silent — don't block startup


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    try:
        subprocess.run(
            ["pbcopy"], input=text.encode(), check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def load_spark_md() -> str:
    """Load SPARK.md or .spark/SPARK.md from the project root.

    Returns the file content, or empty string if not found.
    """
    for candidate in ("SPARK.md", os.path.join(".spark", "SPARK.md")):
        path = os.path.join(os.getcwd(), candidate)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError:
                pass
    return ""


def _get_git_info() -> str:
    """Get git branch + dirty status for the banner.

    Returns e.g. "main ✓" or "feature/foo *", or "" if not a git repo.
    """
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if branch.returncode != 0:
            return ""
        branch_name = branch.stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=3,
        )
        dirty = bool(status.stdout.strip())
        icon = " *" if dirty else " ✓"
        return f"{branch_name}{icon}"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _make_session_label(context) -> str:
    """Extract a short label from the first user message in the session."""
    import re
    for msg in context.messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Clean up: remove newlines, strip, truncate
                label = content.strip().split("\n")[0][:50]
                # Sanitize for filename: lowercase, replace non-alnum with hyphens
                label = re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')
                return label[:40]  # Keep filename reasonable
    return ""


def _get_latest_session() -> str:
    """Return the path to the most recent session file, or empty string."""
    history_dir = os.path.expanduser("~/.spark/history")
    if not os.path.isdir(history_dir):
        return ""
    sessions = sorted(
        [f for f in os.listdir(history_dir) if f.endswith(".json")],
        reverse=True,
    )
    if sessions:
        return os.path.join(history_dir, sessions[0])
    return ""


def _notify_done():
    """Play a notification sound (macOS bell)."""
    try:
        # macOS: use afplay with system sound
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # Fallback: terminal bell
        print("\a", end="", flush=True)


# Shell command prefixes — when user types these, run directly via bash
_SHELL_PREFIXES = (
    "python ", "python3 ", "pip ", "pip3 ",
    "node ", "npm ", "npx ", "yarn ", "bun ",
    "cargo ", "go ", "make ", "cmake ",
    "docker ", "docker-compose ",
    "git ", "gh ", "ls ", "cat ", "cd ", "mkdir ", "rm ", "cp ", "mv ",
    "curl ", "wget ", "chmod ", "brew ",
    "swift ", "swiftc ", "javac ", "java ",
    "gcc ", "g++ ", "clang ",
    "pytest ", "jest ", "ruby ",
    "./",
)


def _is_shell_command(text: str) -> bool:
    """Check if input looks like a direct shell command."""
    return text.startswith(_SHELL_PREFIXES)


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg")

# ---------------------------------------------------------------------------
# Provider info — shared by /providers and --setup
# ---------------------------------------------------------------------------

_PROVIDER_INFO = [
    {
        "name": "gemini",
        "label": "Google Gemini",
        "model": "gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY",
        "signup": "https://aistudio.google.com/apikey",
    },
    {
        "name": "openai",
        "label": "OpenAI",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "signup": "https://platform.openai.com/api-keys",
    },
    {
        "name": "groq",
        "label": "Groq",
        "model": "llama-3.3-70b-versatile",
        "env_var": "GROQ_API_KEY",
        "signup": "https://console.groq.com/keys",
    },
    {
        "name": "deepseek",
        "label": "DeepSeek",
        "model": "deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
        "signup": "https://platform.deepseek.com/api_keys",
    },
    {
        "name": "openrouter",
        "label": "OpenRouter",
        "model": "anthropic/claude-sonnet-4",
        "env_var": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/keys",
    },
    {
        "name": "ollama",
        "label": "Ollama (local)",
        "model": "qwen3.5:122b",
        "env_var": "",
        "signup": "https://ollama.ai",
    },
    {
        "name": "sglang",
        "label": "SGLang (fast local)",
        "model": "Qwen3.5-122B-A10B-NVFP4",
        "env_var": "",
        "signup": "https://github.com/sgl-project/sglang",
    },
]


def _is_image_drop(text: str) -> tuple[str, str]:
    """Detect if input is a dragged-in image file path.

    macOS pastes the full path when you drag a file into the terminal.
    Sometimes paths are quoted or escaped.

    Returns (file_path, remaining_text) or ("", "") if not an image.
    """
    # Clean up: strip quotes, unescape spaces
    cleaned = text.strip().strip("'\"")
    cleaned = cleaned.replace("\\ ", " ")

    # Check if it starts with a path-like string
    # Could be absolute (/Users/...) or home-relative (~/...)
    parts = cleaned.split(maxsplit=1)
    candidate = parts[0] if parts else cleaned

    # Also check the full cleaned string (path might have spaces)
    for path_str in [cleaned, candidate]:
        lower = path_str.lower()
        if any(lower.endswith(ext) for ext in _IMAGE_EXTENSIONS):
            expanded = os.path.expanduser(path_str)
            if os.path.isfile(expanded):
                remaining = parts[1] if len(parts) > 1 and path_str == candidate else ""
                return expanded, remaining

    return "", ""


def _detect_file_mentions(text: str) -> list[str]:
    """Extract file paths mentioned in user input for auto-reading.

    Looks for patterns like: fix auth.py, look at src/main.rs, edit config.yaml
    Returns list of existing file paths.
    """
    import re
    # Match common file path patterns (word.ext or path/to/file.ext)
    pattern = r'(?:^|\s)((?:[\w./~-]+/)?[\w.-]+\.(?:py|js|ts|jsx|tsx|rs|go|java|kt|swift|rb|c|cpp|h|hpp|css|html|yaml|yml|toml|json|md|txt|sh|sql|env))\b'
    matches = re.findall(pattern, text)
    found = []
    for match in matches:
        path = os.path.expanduser(match)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if os.path.isfile(path) and path not in found:
            found.append(path)
    return found


def _is_error_paste(text: str) -> bool:
    """Detect if pasted text looks like an error/traceback."""
    indicators = [
        "Traceback (most recent call last)",
        "Error:", "error:", "ERROR:",
        "TypeError:", "ValueError:", "KeyError:", "AttributeError:",
        "ImportError:", "ModuleNotFoundError:", "FileNotFoundError:",
        "SyntaxError:", "IndentationError:", "NameError:",
        "npm ERR!", "FAIL ", "FAILED",
        "panic:", "fatal error:",
        "Exception in thread",
        "at Object.<anonymous>",
        "Cannot find module",
        "undefined is not",
    ]
    line_count = text.count("\n")
    # Multi-line text with error indicators
    return line_count >= 2 and any(ind in text for ind in indicators)


def build_tools() -> ToolRegistry:
    """Register all built-in tools."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(BashTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(ListDirTool())
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    registry.register(RagSearchTool())
    return registry


_SPARK_LOGO = """\
    [bold #ebcb8b]▄▄[/]
   [bold #ebcb8b]▟██[/]
  [bold #ebcb8b]▟██▀[/]
 [bold #ebcb8b]▟████▙[/]
  [bold #ebcb8b]▀██▄[/]
   [bold #ebcb8b]▀██▄[/]
    [bold #ebcb8b]▀▀[/]\
"""


def print_banner(console: Console, config: dict, mcp_count: int = 0,
                 skill_count: int = 0, spark_md_loaded: bool = False,
                 project_type: str = ""):
    """Print startup banner — two-column layout matching Claude Code."""
    model_name = get(config, "model", "name", default="unknown")
    provider = get(config, "model", "provider", default="")
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]

    git_info = _get_git_info()

    # Left column: logo + identity
    left = Text()
    left.append("\n")
    for line in _SPARK_LOGO.strip().split("\n"):
        left.append("  ")
        left.append_text(Text.from_markup(line))
        left.append("\n")
    left.append("\n")
    left.append(f"  {model_name}", style="#88c0d0")
    if provider:
        left.append(f" · {provider}", style="#4c566a")
    left.append(f"\n  {cwd}", style="#4c566a")
    if git_info:
        left.append(f"  ({git_info})", style="#a3be8c" if "✓" in git_info else "#ebcb8b")

    # Right column: tips and info
    right = Text()
    right.append("Tips for getting started\n", style="bold #eceff4")
    right.append("Run ", style="#8899aa")
    right.append("/help", style="bold #d8dee9")
    right.append(" for available commands\n", style="#8899aa")

    right.append("─" * 35 + "\n", style="#3b4252")

    right.append("Capabilities\n", style="bold #eceff4")
    right.append("Read, write, and edit files\n", style="#8899aa")
    right.append("Run shell commands\n", style="#8899aa")
    right.append("Search code with glob/grep\n", style="#8899aa")
    right.append("Web search and fetch\n", style="#8899aa")
    right.append("Send images with /image\n", style="#8899aa")

    extras = (mcp_count > 0 or skill_count > 0 or spark_md_loaded or project_type)
    if extras:
        right.append("─" * 35 + "\n", style="#3b4252")
        if spark_md_loaded:
            right.append("SPARK.md loaded\n", style="#a3be8c")
        if project_type:
            right.append(f"{project_type}\n", style="#88c0d0")
        if mcp_count > 0:
            right.append(f"MCP: {mcp_count} tool(s) loaded\n", style="#a3be8c")
        if skill_count > 0:
            right.append(f"Skills: {skill_count} available\n", style="#a3be8c")

    # Build two-column table
    table = Table(show_header=False, show_edge=False, box=None,
                  padding=(0, 2), expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_row(left, right)

    title = Text()
    title.append(f" Spark Code v{__version__} ", style="bold #ebcb8b")

    console.print(Panel(table, title=title, border_style="#4c566a",
                        box=ROUNDED, padding=(0, 1)))
    console.print()


def handle_slash_command(cmd: str, context: Context, console: Console,
                         config: dict, skills: SkillRegistry,
                         model: ModelClient,
                         permissions: PermissionManager | None = None,
                         team_manager: TeamManager | None = None,
                         task_store: TaskStore | None = None,
                         memory: Memory | None = None,
                         stats: SessionStats | None = None,
                         pinned: PinnedFiles | None = None,
                         snippets: SnippetLibrary | None = None) -> str | None:
    """Handle slash commands.
    Returns None if handled (no agent needed), or a prompt string for the agent.
    Returns "__ASYNC__" for commands that schedule async work (team spawn).
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        help_text = """## Commands

**Session**
- `/help` — Show this help
- `/clear` — Clear conversation
- `/compact` — Summarize conversation to save context
- `/tokens` — Show token usage
- `/stats` — Show session statistics
- `/cost` — Show session costs and budget
- `/analytics` — Detailed analytics dashboard
- `/history` — List and resume past sessions
- `/search <query>` — Search past sessions by keyword
- `/export` — Export session as markdown
- `/share [gist]` — Export as shareable HTML (or markdown for gist)
- `/fork` — Save current + start fresh context

**Branching**
- `/branch [name]` — Create/list conversation branches
- `/switch <name>` — Switch to a branch
- `/branches` — List all branches

**Code & Files**
- `/diff` — Show git diff with syntax highlighting
- `/undo [N]` — Undo last N file operations (`/undo list` to see stack)
- `/pin <file>` — Pin a file to always stay in context
- `/unpin <file>` — Remove a pinned file
- `/watch <cmd>` — Auto-run command on file changes (`/watch off` to stop)
- `/checkpoint` — Create a restorable checkpoint (git stash)
- `/rollback [N]` — Restore from a checkpoint
- `/continue` — Resume from last checkpoint
- `/retry` — Re-send the last message
- `/clean` — Delete files created this session
- `/apply <url>` — Apply code from a URL (gist, PR, diff)

**Model & Config**
- `/config` — Show current config / `/config set <key> <value>`
- `/model` — Show model info / `/model list` / `/model <provider>`
- `/providers` — Show API providers with signup URLs
- `/profile` — Benchmark model (TTFT, tokens/sec)
- `/benchmark` — Measure model speed (TTFT, tok/s)
- `/mode [ask|auto|trust]` — Switch permission mode
- `/trust` / `/auto` / `/ask` — Quick mode switch
- `/yolo` — Toggle agent mode (autonomous + trust all)

**Team & Planning**
- `/team <prompt>` — Spawn a background worker
- `/team status` — Show worker status
- `/team stop [id]` — Stop a worker (or all)
- `/team msg <name> <msg>` — Send a message to a worker
- `/tasks` — Show the shared task list
- `/messages` — Check messages from workers
- `/plan <prompt>` — Create plan.md / `/plan show` / `/plan go`
- `/projectplan <prompt>` — RAG-researched plan / `/projectplan show` / `/projectplan go`

**Project**
- `/publish [name]` — Create GitHub repo and push
- `/new <name> [desc]` — Scaffold a new project
- `/run [command]` — Run the project (auto-detect)
- `/git sync|pr|log|stash` — Smart git commands
- `/memory` — View/add memory / `/memory edit`
- `/image <path> [prompt]` — Send an image

**Knowledge Base**
- `/docs <query>` — Search indexed docs (Swift, SwiftUI, HIG, App Store Guidelines, CNN)

**Extensibility**
- `/teach <name> <desc> -- <cmd>` — Create a custom tool
- `/snippet save <name> <prompt>` — Save a reusable prompt
- `/snippet <name>` — Run a saved snippet
- `/quit` or `/exit` — Exit

## Skills"""
        for skill in skills.all():
            help_text += f"\n- `/{skill.name}` — {skill.description}"
        console.print(Markdown(help_text))
        return None

    elif command == "/clear":
        context.clear()
        os.system("clear" if os.name != "nt" else "cls")
        print_banner(console, config,
                     skill_count=len(skills.all()),
                     spark_md_loaded=bool(load_spark_md()),
                     project_type=detect_project_type(os.getcwd()))
        return None

    elif command == "/compact":
        compact_msg = context.compact()
        if compact_msg:
            console.print(f"  [#ebcb8b]⚡ {compact_msg}[/#ebcb8b]")
        else:
            console.print("[green]Nothing to compact.[/green]")
        return None

    elif command == "/config":
        import yaml
        if not args:
            console.print(Markdown(f"```yaml\n{yaml.dump(config, default_flow_style=False)}```"))
            return None

        sub_parts = args.strip().split(maxsplit=2)
        sub_cmd = sub_parts[0].lower()

        if sub_cmd == "set":
            if len(sub_parts) < 3:
                console.print("[#ebcb8b]Usage: /config set <key> <value>[/#ebcb8b]")
                console.print("[#8899aa]  /config set model.temperature 0.5[/#8899aa]")
                console.print("[#8899aa]  /config set permissions.mode trust[/#8899aa]")
                console.print("[#8899aa]  /config set ui.notification_sound true[/#8899aa]")
                return None
            key_path = sub_parts[1]
            value = sub_parts[2]
            ok, msg = set_config(config, key_path, value)
            if ok:
                console.print(f"[#a3be8c]{msg}[/#a3be8c]")
            else:
                console.print(f"[#bf616a]{msg}[/#bf616a]")
            return None

        elif sub_cmd == "reset":
            from .config import DEFAULT_CONFIG
            for key, val in DEFAULT_CONFIG.items():
                config[key] = val
            console.print("[#a3be8c]Config reset to defaults (in-memory only)[/#a3be8c]")
            return None

        else:
            console.print(Markdown(f"```yaml\n{yaml.dump(config, default_flow_style=False)}```"))
            return None

    elif command == "/model":
        if not args:
            # Show current model info
            console.print(f"Model: {get(config, 'model', 'name')}")
            console.print(f"Provider: {get(config, 'model', 'provider', default='unknown')}")
            console.print(f"Endpoint: {get(config, 'model', 'endpoint')}")
            console.print(f"Temperature: {get(config, 'model', 'temperature')}")
            console.print(f"Input tokens: {model.total_input_tokens:,}")
            console.print(f"Output tokens: {model.total_output_tokens:,}")
            return None

        sub = args.strip().lower()

        if sub == "list":
            # Show available providers from config
            providers = config.get("providers", {})
            current = get(config, "model", "provider", default="")
            if not providers:
                console.print("[#8899aa]No providers configured in config.yaml[/#8899aa]")
                console.print("[#8899aa]Built-in providers: ollama, gemini, openai[/#8899aa]")
                return None
            console.print("[bold #eceff4]Available providers:[/bold #eceff4]")
            for name, pconf in providers.items():
                marker = " [#a3be8c]← active[/#a3be8c]" if name == current else ""
                model_name = pconf.get("model", "?")
                console.print(f"  [#88c0d0]{name}[/#88c0d0]  {model_name}{marker}")
            console.print("[#8899aa]Switch with: /model <provider>[/#8899aa]")
            return None

        # Switch provider — return signal for main loop
        return f"__MODEL_SWITCH__{sub}"

    elif command == "/providers":
        table = Table(
            title="API Providers",
            border_style="#4c566a",
            show_header=True,
            header_style="bold #88c0d0",
        )
        table.add_column("Provider", style="#88c0d0")
        table.add_column("Default Model", style="#d8dee9")
        table.add_column("API Key", style="#d8dee9")
        table.add_column("Signup URL", style="#5e81ac")
        for p in _PROVIDER_INFO:
            if p["env_var"]:
                key_status = (
                    "[#a3be8c]SET[/#a3be8c]"
                    if os.environ.get(p["env_var"])
                    else "[#bf616a]NOT SET[/#bf616a]"
                )
                key_col = f"{p['env_var']}  {key_status}"
            else:
                key_col = "[#8899aa]none needed[/#8899aa]"
            table.add_row(p["label"], p["model"], key_col, p["signup"])
        console.print(table)
        console.print()
        console.print("[bold #eceff4]To set a key:[/bold #eceff4]")
        console.print("  [#88c0d0]1.[/#88c0d0] Run [bold]spark --setup[/bold] (interactive wizard)")
        console.print("  [#88c0d0]2.[/#88c0d0] Or add to your shell profile (~/.zshrc):")
        console.print('     [#a3be8c]export GEMINI_API_KEY="your-key-here"[/#a3be8c]')
        console.print("     then restart your terminal or run [#a3be8c]source ~/.zshrc[/#a3be8c]")
        return None

    elif command == "/tokens":
        tokens = context.estimate_tokens()
        max_tokens = context.max_tokens
        pct = tokens / max_tokens * 100 if max_tokens else 0
        console.print(f"Context: ~{tokens:,} / {max_tokens:,} tokens ({pct:.0f}%)")
        console.print(f"Turns: {context.turn_count}")
        console.print(f"API usage: {model.total_input_tokens:,} in / {model.total_output_tokens:,} out")
        return None

    elif command in ("/stats", "/status"):
        if not stats:
            console.print("[#8899aa]No stats available[/#8899aa]")
            return None
        table = Table(title="Session Statistics", border_style="#4c566a",
                      show_header=True, header_style="bold #88c0d0")
        table.add_column("Metric", style="#d8dee9")
        table.add_column("Value", style="#eceff4")

        table.add_row("Duration", stats.format_duration())
        table.add_row("Turns", str(context.turn_count))

        # Generation speed
        speed_str = stats.format_speed()
        if speed_str:
            table.add_row("Generation speed", speed_str)

        # Token counts
        if stats.input_tokens > 0 or stats.output_tokens > 0:
            table.add_row("Tokens in / out",
                           f"{stats.input_tokens:,} / {stats.output_tokens:,}")
        elif hasattr(model, 'total_input_tokens'):
            table.add_row("Tokens in / out",
                           f"{model.total_input_tokens:,} / {model.total_output_tokens:,}")

        # Cost
        cost_str = stats.format_cost()
        if cost_str:
            provider_name = get(config, "model", "provider", default="")
            table.add_row("Session cost", f"{cost_str} ({provider_name})")
        else:
            provider_name = get(config, "model", "provider", default="local")
            table.add_row("Session cost", f"$0.00 ({provider_name})")

        # Tool breakdown
        table.add_row("Total tool calls", str(stats.total_tool_calls))
        if stats.tool_calls:
            for tool_name, count in sorted(stats.tool_calls.items(),
                                            key=lambda x: -x[1]):
                table.add_row(f"  {tool_name}", str(count))

        # Files
        created = len(stats.files_created) if hasattr(stats, 'files_created') else 0
        table.add_row("Files",
                       f"{created} created, "
                       f"{len(stats.files_read)} read, "
                       f"{len(stats.files_edited)} edited")
        table.add_row("Commands run", str(stats.commands_run))

        # Workers
        if team_manager:
            workers = team_manager.workers
            total = len(workers)
            if total > 0:
                completed = sum(1 for w in workers.values() if w.status == "completed")
                failed = sum(1 for w in workers.values() if w.status == "failed")
                worker_str = f"{total} spawned, {completed} completed"
                if failed:
                    worker_str += f", {failed} failed"
                table.add_row("Workers", worker_str)

        if pinned and pinned.count > 0:
            table.add_row("Pinned files", str(pinned.count))
        console.print(table)
        return None

    elif command == "/diff":
        # Run git diff and show with syntax highlighting
        diff_args = args.strip() if args else ""
        git_cmd = ["git", "diff"] + (diff_args.split() if diff_args else [])
        try:
            result = subprocess.run(
                git_cmd, capture_output=True, text=True, timeout=10,
            )
            diff_output = result.stdout
            if not diff_output:
                console.print("[#8899aa]No changes to show.[/#8899aa]")
                return None
            syntax = Syntax(diff_output, "diff", theme="nord-darker",
                            line_numbers=False, background_color="#2e3440")
            console.print(Panel(syntax, title="[bold #88c0d0] git diff [/bold #88c0d0]",
                                border_style="#4c566a", box=ROUNDED, padding=(0, 1)))
        except subprocess.TimeoutExpired:
            console.print("[#bf616a]git diff timed out[/#bf616a]")
        except FileNotFoundError:
            console.print("[#bf616a]git not found[/#bf616a]")
        return None

    elif command == "/memory":
        if memory is None:
            console.print("[#bf616a]Memory system not available[/#bf616a]")
            return None

        if not args:
            # Show global + project memory
            global_mem = memory.load_global()
            project_mem = memory.load_project()
            if global_mem:
                console.print(Panel(
                    Markdown(global_mem),
                    title="[bold #88c0d0] Global Memory [/bold #88c0d0]",
                    border_style="#4c566a", box=ROUNDED, padding=(1, 2),
                ))
            if project_mem:
                console.print(Panel(
                    Markdown(project_mem),
                    title="[bold #88c0d0] Project Memory [/bold #88c0d0]",
                    border_style="#4c566a", box=ROUNDED, padding=(1, 2),
                ))
            if not global_mem and not project_mem:
                console.print("[#8899aa]No memory entries yet. Use /memory add <entry>[/#8899aa]")
            return None

        sub_parts = args.strip().split(maxsplit=1)
        sub_cmd = sub_parts[0].lower()

        if sub_cmd == "add":
            entry = sub_parts[1] if len(sub_parts) > 1 else ""
            if not entry:
                console.print("[#ebcb8b]Usage: /memory add <entry>[/#ebcb8b]")
                return None
            memory.append_project(entry)
            console.print("[#a3be8c]Added to project memory[/#a3be8c]")
            return None

        elif sub_cmd == "edit":
            return (
                "Read the project memory file at .spark/memory/MEMORY.md "
                "(create it if it doesn't exist). Show its contents to the user "
                "and ask what they'd like to change. Then edit it accordingly."
            )

        else:
            # Treat as add
            memory.append_project(args.strip())
            console.print("[#a3be8c]Added to project memory[/#a3be8c]")
            return None

    elif command == "/yolo":
        # Toggle agent mode — swap system prompt and set trust
        is_agentic = context.system_prompt.startswith("You are Spark Code, a fully autonomous")
        if is_agentic:
            # Switch back to normal mode
            new_prompt = context.system_prompt.replace(AGENTIC_PROMPT, SYSTEM_PROMPT)
            context.system_prompt = new_prompt
            if permissions:
                permissions.mode = "auto"
            console.print("[#8899aa]Agent mode off — back to normal[/#8899aa]")
        else:
            # Switch to agent mode
            new_prompt = context.system_prompt.replace(SYSTEM_PROMPT, AGENTIC_PROMPT)
            context.system_prompt = new_prompt
            if permissions:
                permissions.mode = "trust"
            console.print("[bold #ebcb8b]⚡ Agent mode on[/bold #ebcb8b] [#8899aa]— autonomous execution, all tools trusted[/#8899aa]")
        return None

    elif command == "/image":
        if not args:
            console.print("[#ebcb8b]Usage: /image <file_path> [prompt][/#ebcb8b]")
            console.print("[#8899aa]Example: /image ~/Desktop/screenshot.png what's wrong with this UI?[/#8899aa]")
            return None
        # Parse: first token is path, rest is prompt
        img_parts = args.split(maxsplit=1)
        img_path = os.path.expanduser(img_parts[0])
        img_prompt = img_parts[1] if len(img_parts) > 1 else "Describe this image."
        if not os.path.exists(img_path):
            console.print(f"[#bf616a]File not found: {img_path}[/#bf616a]")
            return None
        # Read and encode image
        mime_type = mimetypes.guess_type(img_path)[0] or "image/png"
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        size_kb = len(img_data) * 3 / 4 / 1024
        console.print(f"  [#88c0d0]Image[/#88c0d0] [#d8dee9]{os.path.basename(img_path)}[/#d8dee9] [#4c566a]({size_kb:.0f} KB, {mime_type})[/#4c566a]")
        # Store image in context and return prompt for agent
        context.add_user_with_image(img_prompt, img_data, mime_type)
        return "__IMAGE_SENT__"  # Signal to skip add_user in agent

    elif command in ("/mode", "/trust", "/auto", "/ask"):
        if permissions is None:
            console.print("[#bf616a]Permissions not available[/#bf616a]")
            return None
        valid_modes = {"ask", "auto", "trust"}
        if command == "/mode":
            if args and args.strip() in valid_modes:
                new_mode = args.strip()
            else:
                console.print(f"[#88c0d0]Current mode:[/#88c0d0] [#d8dee9]{permissions.mode}[/#d8dee9]")
                console.print("[#8899aa]Usage: /mode <ask|auto|trust>  or  shift+tab to cycle[/#8899aa]")
                console.print("[#8899aa]  ask   — confirm every tool call[/#8899aa]")
                console.print("[#8899aa]  auto  — allow reads, ask for writes[/#8899aa]")
                console.print("[#8899aa]  trust — allow all tool calls[/#8899aa]")
                console.print("[#8899aa]  plan  — plan before executing (via shift+tab)[/#8899aa]")
                return None
        else:
            new_mode = command[1:]  # /trust -> trust, /auto -> auto, /ask -> ask
        permissions.mode = new_mode
        config["permissions"]["mode"] = new_mode
        console.print(f"[#a3be8c]Switched to {new_mode} mode[/#a3be8c]")
        return None

    elif command in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    elif command == "/team":
        if not team_manager:
            console.print("[#bf616a]Team system not available[/#bf616a]")
            return None

        if not args:
            console.print("[#ebcb8b]Usage: /team <prompt> — spawn a worker[/#ebcb8b]")
            console.print("[#8899aa]  /team status     — show all workers[/#8899aa]")
            console.print("[#8899aa]  /team stop [id]  — stop a worker[/#8899aa]")
            console.print("[#8899aa]  /team msg <name> <message> — message a worker[/#8899aa]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        if sub_cmd == "status":
            workers = team_manager.status()
            if not workers:
                console.print("[#8899aa]No workers spawned yet.[/#8899aa]")
                return None
            for w in workers:
                status_icon = {
                    "running": "[#ebcb8b]⟳ running[/#ebcb8b]",
                    "completed": "[#a3be8c]✓ completed[/#a3be8c]",
                    "failed": "[#bf616a]✗ failed[/#bf616a]",
                }.get(w["status"], w["status"])
                console.print(
                    f"  [#88c0d0]#{w['id']}[/#88c0d0] "
                    f"[#d8dee9]{w['name']}[/#d8dee9]  "
                    f"{status_icon}  "
                    f"[#8899aa]{w['prompt'][:60]}[/#8899aa]"
                )
            return None

        elif sub_cmd == "stop":
            stop_id = sub[1] if len(sub) > 1 else ""
            return f"__TEAM_STOP__{stop_id}"

        elif sub_cmd == "msg":
            # /team msg worker-1 Hey, how's it going?
            msg_rest = sub[1] if len(sub) > 1 else ""
            msg_parts = msg_rest.split(maxsplit=1)
            if len(msg_parts) < 2:
                console.print("[#ebcb8b]Usage: /team msg <worker-name> <message>[/#ebcb8b]")
                return None
            target_name = msg_parts[0]
            msg_content = msg_parts[1]
            result = team_manager.deliver_message("lead", target_name, msg_content)
            console.print(f"[#a3be8c]{result}[/#a3be8c]")
            return None

        else:
            # Spawn a worker — the full args is the prompt
            return f"__TEAM_SPAWN__{args}"

    elif command == "/messages":
        if not team_manager:
            console.print("[#bf616a]Team system not available[/#bf616a]")
            return None
        msgs = team_manager.get_lead_messages()
        if not msgs:
            console.print("[#8899aa]No new messages.[/#8899aa]")
            return None
        for m in msgs:
            console.print(
                f"  [#5e81ac][{m.from_name}][/#5e81ac] "
                f"[#d8dee9]{m.content}[/#d8dee9]"
            )
        return None

    elif command == "/tasks":
        if not task_store:
            console.print("[#bf616a]Task store not available[/#bf616a]")
            return None
        tasks = task_store.list()
        if not tasks:
            console.print("[#8899aa]No tasks yet. Spawn a worker with /team <prompt>[/#8899aa]")
            return None
        for t in tasks:
            status_icon = {
                "pending": "[#ebcb8b]○[/#ebcb8b]",
                "in_progress": "[#ebcb8b]⟳[/#ebcb8b]",
                "completed": "[#a3be8c]✓[/#a3be8c]",
                "failed": "[#bf616a]✗[/#bf616a]",
            }.get(t.status, " ")
            assigned = f"  [#4c566a]({t.assigned_to})[/#4c566a]" if t.assigned_to else ""
            console.print(
                f"  [#88c0d0]#{t.id}[/#88c0d0]  "
                f"{status_icon} {t.status:<12}  "
                f"[#d8dee9]{t.description[:60]}[/#d8dee9]"
                f"{assigned}"
            )
        return None

    elif command == "/plan":
        if not args:
            console.print("[#ebcb8b]Usage: /plan <prompt> — create a plan[/#ebcb8b]")
            console.print("[#8899aa]  /plan show    — show current plan[/#8899aa]")
            console.print("[#8899aa]  /plan copy    — copy plan to clipboard[/#8899aa]")
            console.print("[#8899aa]  /plan go      — execute the approved plan[/#8899aa]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        plan_path = os.path.join(os.getcwd(), "plan.md")

        if sub_cmd == "show":
            if not os.path.exists(plan_path):
                console.print("[#8899aa]No plan.md found. Create one with /plan <prompt>[/#8899aa]")
                return None
            with open(plan_path) as f:
                content = f.read()
            console.print()
            console.print(Panel(
                Markdown(content),
                title="[bold #88c0d0] plan.md [/bold #88c0d0]",
                border_style="#4c566a",
                box=ROUNDED,
                padding=(1, 2),
            ))
            _copy_to_clipboard(content)
            console.print()
            console.print("[#a3be8c]  ✓ Copied to clipboard  ·  /plan go to execute  ·  /plan <prompt> to redo[/#a3be8c]")
            return None

        elif sub_cmd == "copy":
            if not os.path.exists(plan_path):
                console.print("[#8899aa]No plan.md found. Create one with /plan <prompt>[/#8899aa]")
                return None
            with open(plan_path) as f:
                content = f.read()
            if _copy_to_clipboard(content):
                console.print("[#a3be8c]✓ Plan copied to clipboard[/#a3be8c]")
            else:
                console.print("[#bf616a]Could not copy — pbcopy not available[/#bf616a]")
            return None

        elif sub_cmd == "go":
            if not os.path.exists(plan_path):
                console.print("[#bf616a]No plan.md found. Create one first with /plan <prompt>[/#bf616a]")
                return None
            with open(plan_path) as f:
                plan_content = f.read()
            # System-driven execution — return signal for main loop
            return f"__PLAN_EXECUTE__{plan_content}"

        else:
            # /plan <prompt> — create a plan
            plan_prompt = args
            create_plan_prompt = (
                f"The user wants you to create a plan before executing. "
                f"Their request: {plan_prompt}\n\n"
                "RULES:\n"
                "1. Use read-only tools to explore the codebase first (read_file, glob, grep, list_dir)\n"
                "2. Do NOT create project files yet — only create plan.md\n"
                "3. Create a detailed implementation plan\n"
                "4. You MUST use the write_file tool to save the plan to 'plan.md' in the current directory\n"
                "   This is the ONE file you must write. Do not skip this step.\n\n"
                "The plan.md should include:\n"
                "- Summary of what will be done\n"
                "- Numbered steps with clear descriptions\n"
                "- Files that will be modified or created\n"
                "- Which steps can be done in parallel (mark them clearly)\n"
                "- Any risks or considerations\n\n"
                "After writing plan.md, tell the user: Review the plan with /plan show, then /plan go to execute."
            )
            return create_plan_prompt

    elif command == "/projectplan":
        if not args:
            console.print("[#ebcb8b]Usage: /projectplan <prompt> — create a RAG-researched plan[/#ebcb8b]")
            console.print("[#8899aa]  /projectplan show    — show current plan[/#8899aa]")
            console.print("[#8899aa]  /projectplan copy    — copy plan to clipboard[/#8899aa]")
            console.print("[#8899aa]  /projectplan go      — execute the approved plan[/#8899aa]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        pp_path = os.path.join(os.getcwd(), "projectplan.md")

        if sub_cmd == "show":
            if not os.path.exists(pp_path):
                console.print("[#8899aa]No projectplan.md found. Create one with /projectplan <prompt>[/#8899aa]")
                return None
            with open(pp_path) as f:
                content = f.read()
            console.print()
            console.print(Panel(
                Markdown(content),
                title="[bold #88c0d0] projectplan.md [/bold #88c0d0]",
                border_style="#4c566a",
                box=ROUNDED,
                padding=(1, 2),
            ))
            _copy_to_clipboard(content)
            console.print()
            console.print("[#a3be8c]  ✓ Copied to clipboard  ·  /projectplan go to execute  ·  /projectplan <prompt> to redo[/#a3be8c]")
            return None

        elif sub_cmd == "copy":
            if not os.path.exists(pp_path):
                console.print("[#8899aa]No projectplan.md found. Create one with /projectplan <prompt>[/#8899aa]")
                return None
            with open(pp_path) as f:
                content = f.read()
            if _copy_to_clipboard(content):
                console.print("[#a3be8c]✓ Project plan copied to clipboard[/#a3be8c]")
            else:
                console.print("[#bf616a]Could not copy — pbcopy not available[/#bf616a]")
            return None

        elif sub_cmd == "go":
            if not os.path.exists(pp_path):
                console.print("[#bf616a]No projectplan.md found. Create one first with /projectplan <prompt>[/#bf616a]")
                return None
            with open(pp_path) as f:
                plan_content = f.read()
            return f"__PLAN_EXECUTE__{plan_content}"

        else:
            # /projectplan <prompt> — research RAG, then create plan
            plan_prompt = args
            project_type = detect_project_type(os.getcwd())

            # Extract keywords and fetch RAG context
            keywords = extract_keywords(plan_prompt)
            console.print(f"[#88c0d0]▸ Researching docs for: {', '.join(keywords) or plan_prompt}[/#88c0d0]")

            rag_context = fetch_rag_context(keywords, project_type, prompt=plan_prompt)

            if rag_context:
                ref_count = rag_context.count("[Ref ")
                console.print(f"[#a3be8c]  ✓ Found {ref_count} reference(s) from knowledge base[/#a3be8c]")
            else:
                console.print(f"[#ebcb8b]  ⚠ No RAG results (service down or no matches)[/#ebcb8b]")
            # Discover MCP tools
            mcp_section = ""
            mcp_configs = find_mcp_configs()
            if mcp_configs:
                mcp_lines = ["## Available MCP Tools\n",
                             "You MUST incorporate these tools into the plan steps where applicable. "
                             "For each tool, add a step or sub-step that uses it.\n"]
                for server_name, server_conf in mcp_configs.items():
                    desc = server_conf.get("description", "")
                    tools_list = server_conf.get("tools", [])
                    mcp_lines.append(f"**{server_name}**" + (f" — {desc}" if desc else ""))
                    if tools_list:
                        for t in tools_list:
                            if isinstance(t, dict):
                                mcp_lines.append(f"  - `{t.get('name', '?')}`: {t.get('description', '')}")
                            else:
                                mcp_lines.append(f"  - `{t}`")
                    mcp_lines.append("")
                mcp_section = "\n".join(mcp_lines) + "\n"
                console.print(f"[#a3be8c]  ✓ Found {len(mcp_configs)} MCP server(s)[/#a3be8c]")

            # Detect if directory is empty (new project)
            is_empty_dir = not any(
                f for f in os.listdir(os.getcwd())
                if not f.startswith(".") and f not in ("projectplan.md", "plan.md")
            )

            console.print(f"[#88c0d0]▸ {'Writing' if is_empty_dir else 'Exploring codebase and writing'} projectplan.md...[/#88c0d0]")
            console.print(f"[#4c566a]  (This may take a minute with large models. Ctrl+C to cancel)[/#4c566a]")

            rag_section = ""
            if rag_context:
                rag_section = (
                    "Pre-researched documentation — include as '## Reference Material' in the plan. "
                    "Tag steps with [see Ref N].\n\n"
                    f"{rag_context}\n\n---\n\n"
                )

            # Build MCP instruction if tools are available
            mcp_instruction = ""
            if mcp_section:
                mcp_instruction = (
                    "## MCP Tools section is REQUIRED. Add a dedicated step for each "
                    "MCP tool that applies to this project. For example, if an image generation "
                    "tool is available and the project involves images/photos, add a step that "
                    "uses it.\n\n"
                )

            if is_empty_dir:
                # Lean prompt for new projects — no codebase to explore
                create_plan_prompt = (
                    f"Create projectplan.md for a NEW project.\n"
                    f"Request: {plan_prompt}\n\n"
                    f"{rag_section}"
                    f"{mcp_section}"
                    f"{mcp_instruction}"
                    "This is an EMPTY directory — do NOT explore files. "
                    "Write projectplan.md IMMEDIATELY using write_file.\n\n"
                    "Format: ## Reference Material (if docs above), ---, "
                    "## Summary, ## Steps (numbered, tag with [see Ref N]), "
                    "## Parallelization, ## Files, ## Risks\n\n"
                    "Keep the plan concise — max 8 steps. Write the file NOW."
                )
            else:
                # Full prompt for existing projects
                create_plan_prompt = (
                    f"Create projectplan.md for this project.\n"
                    f"Request: {plan_prompt}\n"
                    f"Project type: {project_type or 'unknown'}\n\n"
                    f"{rag_section}"
                    f"{mcp_section}"
                    f"{mcp_instruction}"
                    "RULES:\n"
                    "1. Explore the codebase first (glob, grep, read_file)\n"
                    "2. Only create projectplan.md — no other files\n"
                    "3. Use write_file to save projectplan.md\n\n"
                    "Format: ## Reference Material (if docs above), ---, "
                    "## Summary, ## Steps (numbered, tag with [see Ref N]), "
                    "## Parallelization, ## Files, ## Risks\n\n"
                    "Do NOT use rag_search — docs are pre-researched.\n"
                    "After writing, say: Review with /projectplan show, then /projectplan go to execute."
                )
            return create_plan_prompt

    elif command == "/publish":
        # Create a GitHub repo and push the project
        raw_args = args.strip() if args else ""

        # Parse --private flag first
        visibility = "--public"
        if "--private" in raw_args:
            visibility = "--private"
            raw_args = raw_args.replace("--private", "").strip()

        # Auto-detect repo name from current directory if not provided
        repo_name = raw_args
        if not repo_name:
            import re
            repo_name = os.path.basename(os.getcwd())
            # Sanitize: lowercase, replace spaces/underscores with hyphens, strip non-alnum
            repo_name = repo_name.lower().replace(" ", "-").replace("_", "-")
            repo_name = re.sub(r'[^a-z0-9\-]', '', repo_name)
            repo_name = re.sub(r'-+', '-', repo_name).strip('-')
            console.print(f"[#88c0d0]Publishing as [bold]\"{repo_name}\"[/bold] on GitHub ({visibility.lstrip('-')})[/#88c0d0]")

        return (
            f"Create a GitHub repository called \"{repo_name}\" and push the current project. Steps:\n\n"
            f"1. Read the source files in this directory to understand the project\n"
            f"2. If no README.md exists, use write_file to create a short README.md with:\n"
            f"   - Project name and one-line description\n"
            f"   - How to install and run it\n"
            f"   - Keep it concise (under 30 lines)\n"
            f"3. If no .gitignore exists, use write_file to create one appropriate for the languages used\n"
            f"   (e.g. Python: __pycache__/, .venv/, .env; Node: node_modules/, dist/; etc.)\n"
            f"4. Check if this directory already has a git repo. If not, run `git init`\n"
            f"5. Stage all project files with `git add .`\n"
            f"6. Commit with a descriptive message based on what the project does\n"
            f"7. Create the GitHub repo: `gh repo create {repo_name} {visibility} --source=. --push`\n"
            f"8. Show the repo URL when done\n\n"
            f"Use write_file for creating README.md and .gitignore. Use bash for git commands.\n"
            f"Do not ask questions — just do it."
        )

    elif command == "/new":
        # Scaffold a new project
        raw_args = args.strip() if args else ""
        if not raw_args:
            console.print("[#ebcb8b]Usage: /new <project-name> [description][/#ebcb8b]")
            console.print("[#8899aa]  Scaffolds a new project with git[/#8899aa]")
            console.print("[#8899aa]  Example: /new my-app[/#8899aa]")
            console.print("[#8899aa]  Example: /new weather-api a FastAPI weather service[/#8899aa]")
            return None

        parts = raw_args.split(None, 1)
        project_name = parts[0]
        description = parts[1] if len(parts) > 1 else ""

        desc_hint = ""
        if description:
            desc_hint = (
                f"The user described this project as: \"{description}\"\n"
                f"Use this to pick the right language, framework, and project structure.\n\n"
            )

        console.print(f"[#88c0d0]Scaffolding [bold]\"{project_name}\"[/bold]...[/#88c0d0]")

        return (
            f"Create a new project called \"{project_name}\". Steps:\n\n"
            f"{desc_hint}"
            f"1. Create the directory: `mkdir -p {project_name}`\n"
            f"2. Run `git init` inside it\n"
            f"3. Use write_file to create these files inside {project_name}/:\n"
            f"   - A main entry point file (e.g. main.py, index.js, main.go — pick based on the description)\n"
            f"   - A .gitignore appropriate for the language\n"
            f"   - A short README.md with the project name and description\n"
            f"4. If the language needs a dependency file (requirements.txt, package.json, go.mod), create it\n"
            f"5. Show the file tree when done\n"
            f"6. End with: \"Use `/publish` to push to GitHub when ready.\"\n\n"
            f"Use write_file for all file creation. Use bash only for mkdir and git init.\n"
            f"Do not ask questions — just do it."
        )

    elif command == "/run":
        # Run/launch the project in the current directory
        run_cmd = args.strip() if args else ""

        if run_cmd:
            # User specified a command: /run python app.py
            return f"__RUN_CMD__{run_cmd}"

        # Auto-detect: ask the agent to figure out how to run the project
        return (
            "Look at the files in the current directory and figure out how to "
            "run this project. Check for:\n"
            "- package.json (npm start / npm run dev)\n"
            "- pyproject.toml or setup.py (python -m or entry point)\n"
            "- main.py, app.py, server.py, index.js, etc.\n"
            "- Makefile (make run)\n"
            "- docker-compose.yml (docker compose up)\n\n"
            "Then run it using the bash tool. If it's a web server, "
            "note the URL. If it's a CLI app, show the output."
        )

    elif command == "/history":
        from datetime import datetime as _dt
        history_dir = os.path.expanduser("~/.spark/history")
        if not os.path.isdir(history_dir):
            console.print("[#8899aa]No saved sessions yet.[/#8899aa]")
            return None
        sessions = sorted(
            [f for f in os.listdir(history_dir) if f.endswith(".json")],
            reverse=True,
        )
        if not sessions:
            console.print("[#8899aa]No saved sessions yet.[/#8899aa]")
            return None
        if args.strip():
            # Resume a specific session
            target = args.strip()
            matches = [s for s in sessions if target in s]
            if not matches:
                console.print(f"[#bf616a]No session matching '{target}'[/#bf616a]")
                return None
            session_path = os.path.join(history_dir, matches[0])
            if context.load(session_path):
                meta = Context.read_metadata(session_path)
                label = meta.get("label", "")
                label_display = f"  [#d8dee9]{label}[/#d8dee9]" if label else ""
                console.print(f"[#a3be8c]Resumed session: {matches[0]}{label_display}[/#a3be8c]")
            else:
                console.print("[#bf616a]Failed to load session[/#bf616a]")
            return None
        # List recent sessions with metadata
        console.print("[bold #eceff4]Recent sessions:[/bold #eceff4]")
        for s in sessions[:10]:
            session_path = os.path.join(history_dir, s)
            meta = Context.read_metadata(session_path)
            # Time ago
            time_str = ""
            ts = meta.get("timestamp", "")
            if ts:
                try:
                    session_time = _dt.fromisoformat(ts)
                    delta = _dt.now() - session_time
                    if delta.days > 0:
                        time_str = f"{delta.days}d ago"
                    elif delta.seconds >= 3600:
                        time_str = f"{delta.seconds // 3600}h ago"
                    elif delta.seconds >= 60:
                        time_str = f"{delta.seconds // 60}m ago"
                    else:
                        time_str = "just now"
                except (ValueError, TypeError):
                    pass
            turns = meta.get("turn_count", 0)
            label = meta.get("label", "")
            cwd = meta.get("cwd", "")
            home = os.path.expanduser("~")
            if cwd.startswith(home):
                cwd = "~" + cwd[len(home):]

            # Build display line
            parts = []
            if time_str:
                parts.append(f"[#8899aa]{time_str:<8}[/#8899aa]")
            if turns:
                parts.append(f"[#8899aa]{turns} turns[/#8899aa]")
            if label:
                parts.append(f"[#d8dee9]{label}[/#d8dee9]")
            if cwd:
                parts.append(f"[#4c566a]{cwd}[/#4c566a]")

            name = s.replace(".json", "")
            detail = "  ·  ".join(p for p in parts) if parts else ""
            console.print(f"  [#88c0d0]{name}[/#88c0d0]  {detail}")
        console.print("[#8899aa]Use /history <name> to resume  ·  spark --resume to continue last[/#8899aa]")
        return None

    elif command == "/undo":
        undo_dir = os.path.expanduser("~/.spark/.undo")
        if not os.path.isdir(undo_dir):
            console.print("[#8899aa]Nothing to undo.[/#8899aa]")
            return None
        undo_files = sorted(os.listdir(undo_dir), reverse=True)
        if not undo_files:
            console.print("[#8899aa]Nothing to undo.[/#8899aa]")
            return None

        # Parse count: /undo 3 → undo last 3
        count = 1
        if args and args.strip().isdigit():
            count = min(int(args.strip()), len(undo_files))
        elif args and args.strip() == "list":
            # Show undo stack
            import json as _json
            console.print("[bold #eceff4]Undo stack:[/bold #eceff4]")
            for i, uf in enumerate(undo_files[:20], 1):
                try:
                    with open(os.path.join(undo_dir, uf), encoding="utf-8") as f:
                        data = _json.load(f)
                    home = os.path.expanduser("~")
                    p = data["path"]
                    display = "~" + p[len(home):] if p.startswith(home) else p
                    console.print(f"  [#88c0d0]{i}.[/#88c0d0] [#d8dee9]{display}[/#d8dee9]")
                except Exception:
                    console.print(f"  [#88c0d0]{i}.[/#88c0d0] [#8899aa](corrupt entry)[/#8899aa]")
            return None

        restored = 0
        import json as _json
        for uf in undo_files[:count]:
            undo_meta_path = os.path.join(undo_dir, uf)
            try:
                with open(undo_meta_path, encoding="utf-8") as f:
                    undo_data = _json.load(f)
                original_path = undo_data["path"]
                original_content = undo_data["content"]
                with open(original_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                os.remove(undo_meta_path)
                home = os.path.expanduser("~")
                display = "~" + original_path[len(home):] if original_path.startswith(home) else original_path
                console.print(f"[#a3be8c]Restored: {display}[/#a3be8c]")
                restored += 1
            except Exception as e:
                console.print(f"[#bf616a]Undo failed: {e}[/#bf616a]")
        if restored > 1:
            console.print(f"[#a3be8c]Undid {restored} operations[/#a3be8c]")
        return None

    elif command == "/pin":
        if not pinned:
            console.print("[#bf616a]Pin system not available[/#bf616a]")
            return None
        if not args:
            files = pinned.list()
            if not files:
                console.print("[#8899aa]No pinned files. Use /pin <file_path>[/#8899aa]")
            else:
                console.print("[bold #eceff4]Pinned files:[/bold #eceff4]")
                home = os.path.expanduser("~")
                for f in files:
                    display = "~" + f[len(home):] if f.startswith(home) else f
                    console.print(f"  [#88c0d0]{display}[/#88c0d0]")
            return None
        ok, msg = pinned.pin(args.strip())
        style = "#a3be8c" if ok else "#bf616a"
        console.print(f"[{style}]{msg}[/{style}]")
        return None

    elif command == "/unpin":
        if not pinned:
            console.print("[#bf616a]Pin system not available[/#bf616a]")
            return None
        if not args:
            console.print("[#ebcb8b]Usage: /unpin <file_path>[/#ebcb8b]")
            return None
        ok, msg = pinned.unpin(args.strip())
        style = "#a3be8c" if ok else "#bf616a"
        console.print(f"[{style}]{msg}[/{style}]")
        return None

    elif command == "/git":
        if not args:
            console.print("[#ebcb8b]Usage: /git <command>[/#ebcb8b]")
            console.print("[#8899aa]  /git sync   — pull, then push[/#8899aa]")
            console.print("[#8899aa]  /git pr     — create a PR with AI-generated description[/#8899aa]")
            console.print("[#8899aa]  /git stash  — stash changes[/#8899aa]")
            console.print("[#8899aa]  /git log    — show recent commits[/#8899aa]")
            return None
        sub = args.strip().lower()
        if sub == "sync":
            return "__RUN_CMD__git pull --rebase && git push"
        elif sub == "stash":
            return "__RUN_CMD__git stash"
        elif sub == "log":
            return "__RUN_CMD__git log --oneline -15"
        elif sub == "pr":
            return (
                "Create a pull request for the current branch. Steps:\n"
                "1. Run `git branch --show-current` to get branch name\n"
                "2. Run `git log main..HEAD --oneline` to see commits\n"
                "3. Run `git diff main...HEAD --stat` to see changed files\n"
                "4. Generate a concise PR title and description based on the changes\n"
                "5. Run `gh pr create --title \"<title>\" --body \"<description>\"` to create the PR\n"
                "6. Show the PR URL when done"
            )
        else:
            # Pass through as git command
            return f"__RUN_CMD__git {args.strip()}"

    elif command == "/fork":
        # Save current session and start fresh
        from datetime import datetime as _dt
        try:
            history_dir = os.path.expanduser("~/.spark/history")
            os.makedirs(history_dir, exist_ok=True)
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            label = _make_session_label(context)
            suffix = f"_{label}" if label else ""
            save_path = os.path.join(history_dir, f"{ts}{suffix}_fork.json")
            context.save(save_path, label=f"{label} (forked)", cwd=os.getcwd())
            console.print("[#a3be8c]Session saved as fork. Starting fresh context.[/#a3be8c]")
            console.print(f"[#8899aa]Resume the fork with /history {ts}[/#8899aa]")
            context.clear()
        except Exception as e:
            console.print(f"[#bf616a]Fork failed: {e}[/#bf616a]")
        return None

    elif command == "/snippet":
        if not snippets:
            console.print("[#bf616a]Snippet system not available[/#bf616a]")
            return None
        if not args:
            all_snippets = snippets.list()
            if not all_snippets:
                console.print("[#8899aa]No snippets. Use /snippet save <name> <prompt>[/#8899aa]")
            else:
                console.print("[bold #eceff4]Saved snippets:[/bold #eceff4]")
                for name, prompt in all_snippets.items():
                    preview = prompt[:60].replace("\n", " ")
                    console.print(f"  [#88c0d0]{name}[/#88c0d0]  [#8899aa]{preview}[/#8899aa]")
            return None
        sub_parts = args.strip().split(maxsplit=2)
        if sub_parts[0].lower() == "save":
            if len(sub_parts) < 3:
                console.print("[#ebcb8b]Usage: /snippet save <name> <prompt>[/#ebcb8b]")
                return None
            msg = snippets.add(sub_parts[1], sub_parts[2])
            console.print(f"[#a3be8c]{msg}[/#a3be8c]")
            return None
        elif sub_parts[0].lower() == "remove":
            if len(sub_parts) < 2:
                console.print("[#ebcb8b]Usage: /snippet remove <name>[/#ebcb8b]")
                return None
            msg = snippets.remove(sub_parts[1])
            console.print(f"[#8899aa]{msg}[/#8899aa]")
            return None
        else:
            # Run a snippet by name
            prompt = snippets.get(sub_parts[0])
            if prompt:
                return prompt
            console.print(f"[#bf616a]Snippet not found: {sub_parts[0]}[/#bf616a]")
            return None

    elif command == "/export":
        # Export session as markdown
        if context.turn_count == 0:
            console.print("[#8899aa]Nothing to export — no conversation yet.[/#8899aa]")
            return None
        lines = ["# Spark Code Session Export\n"]
        for msg in context.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and isinstance(content, str):
                lines.append(f"## User\n\n{content}\n")
            elif role == "assistant" and isinstance(content, str) and content:
                lines.append(f"## Assistant\n\n{content}\n")
            elif role == "tool":
                name = msg.get("name", "tool")
                result_preview = (content[:200] + "...") if len(content or "") > 200 else content
                lines.append(f"**Tool: {name}**\n```\n{result_preview}\n```\n")
        export_text = "\n".join(lines)
        export_path = os.path.join(os.getcwd(), "session_export.md")
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(export_text)
            console.print(f"[#a3be8c]Exported to {export_path}[/#a3be8c]")
            _copy_to_clipboard(export_text)
            console.print("[#8899aa]Also copied to clipboard[/#8899aa]")
        except OSError as e:
            console.print(f"[#bf616a]Export failed: {e}[/#bf616a]")
        return None

    elif command == "/cost":
        # Show running cost information
        in_tokens = model.total_input_tokens
        out_tokens = model.total_output_tokens
        cost = model.estimated_cost
        provider = get(config, "model", "provider", default="ollama")
        console.print(f"[bold #eceff4]Session Cost[/bold #eceff4]")
        console.print(f"  [#88c0d0]Provider:[/#88c0d0] [#d8dee9]{provider}[/#d8dee9]")
        console.print(f"  [#88c0d0]Input tokens:[/#88c0d0] [#d8dee9]{in_tokens:,}[/#d8dee9]")
        console.print(f"  [#88c0d0]Output tokens:[/#88c0d0] [#d8dee9]{out_tokens:,}[/#d8dee9]")
        if cost > 0:
            console.print(f"  [#88c0d0]Estimated cost:[/#88c0d0] [#a3be8c]${cost:.4f}[/#a3be8c]")
            budget = get(config, "budget_limit", default=0)
            if budget and cost > float(budget):
                console.print(f"  [#bf616a]Over budget limit (${budget})![/#bf616a]")
            elif budget:
                remaining = float(budget) - cost
                console.print(f"  [#8899aa]Budget remaining: ${remaining:.4f}[/#8899aa]")
        else:
            console.print(f"  [#8899aa]Local model — no cost[/#8899aa]")
        return None

    elif command == "/watch":
        if not args:
            console.print("[#ebcb8b]Usage: /watch <command> — auto-run on file changes[/#ebcb8b]")
            console.print("[#8899aa]  /watch pytest       — run tests on change[/#8899aa]")
            console.print("[#8899aa]  /watch npm test     — run JS tests on change[/#8899aa]")
            console.print("[#8899aa]  /watch off          — stop watching[/#8899aa]")
            return None
        return f"__WATCH__{args.strip()}"

    elif command == "/checkpoint":
        # Create a git stash checkpoint
        try:
            result = subprocess.run(
                ["git", "stash", "push", "-m",
                 f"spark-checkpoint-{int(__import__('time').time())}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if "No local changes" in output:
                    console.print("[#8899aa]No changes to checkpoint.[/#8899aa]")
                else:
                    console.print(f"[#a3be8c]Checkpoint created: {output}[/#a3be8c]")
                    # Immediately restore working state (stash keeps a copy)
                    subprocess.run(
                        ["git", "stash", "pop", "--index"],
                        capture_output=True, text=True, timeout=10,
                    )
                    console.print("[#8899aa]Working state preserved. Use /rollback to restore.[/#8899aa]")
            else:
                console.print(f"[#bf616a]Checkpoint failed: {result.stderr.strip()}[/#bf616a]")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            console.print("[#bf616a]Git not available[/#bf616a]")
        return None

    elif command == "/rollback":
        # Show and restore from git stash checkpoints
        try:
            result = subprocess.run(
                ["git", "stash", "list"],
                capture_output=True, text=True, timeout=10,
            )
            stashes = [l for l in result.stdout.strip().split("\n")
                       if l and "spark-checkpoint" in l]
            if not stashes:
                console.print("[#8899aa]No checkpoints found. Use /checkpoint first.[/#8899aa]")
                return None

            if args and args.strip().isdigit():
                idx = int(args.strip())
                restore_result = subprocess.run(
                    ["git", "stash", "pop", f"stash@{{{idx}}}"],
                    capture_output=True, text=True, timeout=10,
                )
                if restore_result.returncode == 0:
                    console.print(f"[#a3be8c]Rolled back to checkpoint {idx}[/#a3be8c]")
                else:
                    console.print(f"[#bf616a]{restore_result.stderr.strip()}[/#bf616a]")
            else:
                console.print("[bold #eceff4]Checkpoints:[/bold #eceff4]")
                for i, stash in enumerate(stashes[:10]):
                    console.print(f"  [#88c0d0]{i}.[/#88c0d0] [#d8dee9]{stash}[/#d8dee9]")
                console.print("[#8899aa]Use /rollback <number> to restore[/#8899aa]")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            console.print("[#bf616a]Git not available[/#bf616a]")
        return None

    elif command == "/profile":
        # Model performance benchmark
        return "__PROFILE__"

    elif command == "/benchmark":
        return "__BENCHMARK__"

    elif command == "/apply":
        if not args:
            console.print("[#ebcb8b]Usage: /apply <url> — apply code from a URL[/#ebcb8b]")
            console.print("[#8899aa]  Supports: GitHub gist, PR diff, raw file URLs[/#8899aa]")
            return None
        url = args.strip()
        return (
            f"Fetch the content from this URL and apply it to the local codebase:\n"
            f"URL: {url}\n\n"
            f"Steps:\n"
            f"1. Use web_fetch to get the content from the URL\n"
            f"2. If it's a diff/patch, apply it using edit_file\n"
            f"3. If it's a full file, write it using write_file\n"
            f"4. If it's a gist with multiple files, create each one\n"
            f"5. Show what was applied"
        )

    elif command == "/teach":
        if not args:
            console.print("[#ebcb8b]Usage: /teach <name> <description> -- <command>[/#ebcb8b]")
            console.print("[#8899aa]  Example: /teach deploy \"Deploy to production\" -- ssh prod 'cd /app && git pull'[/#8899aa]")
            console.print("[#8899aa]  Example: /teach lint \"Run linter\" -- ruff check --fix .[/#8899aa]")
            return None
        return f"__TEACH__{args.strip()}"

    elif command == "/branch":
        return f"__BRANCH__{args.strip() if args else ''}"

    elif command == "/switch":
        if not args:
            console.print("[#ebcb8b]Usage: /switch <branch-name>[/#ebcb8b]")
            return None
        return f"__SWITCH__{args.strip()}"

    elif command == "/branches":
        return "__BRANCHES__"

    elif command == "/share":
        if context.turn_count == 0:
            console.print("[#8899aa]Nothing to share — no conversation yet.[/#8899aa]")
            return None
        return f"__SHARE__{args.strip() if args else ''}"

    elif command in ("/search", "/hsearch"):
        if not args:
            console.print("[#ebcb8b]Usage: /search <query> — search past sessions[/#ebcb8b]")
            return None
        return f"__SEARCH__{args.strip()}"

    elif command == "/analytics":
        return "__ANALYTICS__"

    elif command == "/retry":
        return "__RETRY__"

    elif command == "/continue":
        return "__CONTINUE__"

    elif command == "/docs":
        if not args.strip():
            console.print("[#ebcb8b]Usage: /docs <query> — search indexed knowledge base[/#ebcb8b]")
            console.print("[#8899aa]Searches: Swift docs, SwiftUI, Apple HIG, App Store Guidelines, CNN, and more[/#8899aa]")
            return None
        return f"Use rag_search to find information about: {args.strip()}. Show the most relevant results with source citations."

    elif command == "/clean":
        return "__CLEAN__"

    # Check skills (skip /plan since we handle it above)
    skill = skills.get(command)
    if skill:
        if skill.requires_args and not args.strip():
            console.print(f"[#ebcb8b]Usage: {command} <prompt>[/#ebcb8b]")
            console.print(f"[#8899aa]{skill.description}[/#8899aa]")
            return None
        return skill.get_prompt(args)

    console.print(f"[yellow]Unknown command: {command}. Type /help for available commands.[/yellow]")
    return None


async def run_interactive(config: dict, resume_session: str = "",
                          continue_prompt: str = ""):
    """Run interactive CLI session.

    resume_session: path to a session JSON to resume on startup
    continue_prompt: if set, resume last session and send this prompt
    """
    theme_name = get(config, "ui", "theme", default="dark")
    console = Console(theme=get_theme(theme_name))
    ensure_dirs()

    # Initialize skills
    skills = SkillRegistry()
    skills.load_all()

    # Initialize pinned files and snippets
    pinned = PinnedFiles()
    snippet_lib = SnippetLibrary()

    # Initialize MCP
    mcp_client = MCPClient()
    mcp_configs = get(config, "mcp_servers", default={})
    mcp_tools = []
    if mcp_configs:
        console.print("[dim]Connecting to MCP servers...[/dim]")
        mcp_tools = await mcp_client.connect_all(mcp_configs)

    # Initialize memory
    memory = Memory(
        global_path=get(config, "memory", "global_path", default="~/.spark/memory"),
    )
    memory_context = memory.load_all()
    agentic = config.get("_agentic", False)
    system_prompt = AGENTIC_PROMPT if agentic else SYSTEM_PROMPT

    # Load SPARK.md project instructions
    spark_md = load_spark_md()
    if spark_md:
        system_prompt += f"\n\n# Project Instructions (SPARK.md)\n{spark_md}"

    if memory_context:
        system_prompt += f"\n\n{memory_context}"

    # Smart project detection
    project_type = detect_project_type(os.getcwd())
    if project_type:
        system_prompt += f"\n\nThis is a {project_type}."

    # Print banner
    print_banner(console, config, mcp_count=len(mcp_tools),
                 skill_count=len(skills.all()),
                 spark_md_loaded=bool(spark_md),
                 project_type=project_type)

    # Initialize components
    model = ModelClient(
        endpoint=get(config, "model", "endpoint"),
        model=get(config, "model", "name"),
        temperature=get(config, "model", "temperature", default=0.7),
        max_tokens=get(config, "model", "max_tokens", default=8192),
        api_key=get(config, "model", "api_key", default=""),
        provider=get(config, "model", "provider", default="ollama"),
        timeout=float(get(config, "model", "timeout", default=300)),
    )

    # Startup connection check (non-blocking)
    try:
        ok, msg = await model.ping()
        if ok:
            console.print(f"  [#a3be8c]{msg}[/#a3be8c]")
            # Preload model into VRAM in background
            asyncio.create_task(_warmup_model(model))
        else:
            console.print(f"  [#ebcb8b]{msg}[/#ebcb8b]")
            console.print("  [#8899aa]Session will start anyway — requests may fail until server is available[/#8899aa]")
    except Exception:
        pass  # Don't block startup

    if agentic:
        console.print("  [bold #ebcb8b]⚡ Agent mode[/bold #ebcb8b] [#8899aa]— fully autonomous, trust all tools[/#8899aa]")
    console.print()

    # Initialize session stats
    session_stats = SessionStats()
    session_stats.set_cost_rates(
        input_rate=get(config, "model", "cost_per_million_input", default=0),
        output_rate=get(config, "model", "cost_per_million_output", default=0),
    )

    platform_prompt = format_platform_prompt(os.getcwd())
    provider_prompt = get(config, "model", "system_prompt", default="")

    context = Context(
        system_prompt=system_prompt,
        max_tokens=get(config, "model", "context_window", default=32768),
        platform_prompt=platform_prompt,
        provider_prompt=provider_prompt,
    )
    tools = build_tools()

    # Register MCP tools
    for mcp_tool in mcp_tools:
        tools.register(mcp_tool)

    permissions = PermissionManager(
        mode=get(config, "permissions", "mode", default="ask"),
        always_allow=get(config, "permissions", "always_allow", default=[]),
    )
    # Progress tracking for toolbar
    current_tool = {"name": "", "detail": ""}

    def _on_tool_start(tool_name, args):
        """Update toolbar with current tool being executed."""
        detail = ""
        if tool_name == "read_file":
            detail = args.get("file_path", "")
        elif tool_name == "bash":
            cmd = args.get("command", "")
            detail = cmd[:40] + "..." if len(cmd) > 40 else cmd
        elif tool_name in ("glob", "grep"):
            detail = args.get("pattern", "")
        elif tool_name in ("write_file", "edit_file"):
            detail = args.get("file_path", "")
        home = os.path.expanduser("~")
        if detail.startswith(home):
            detail = "~" + detail[len(home):]
        current_tool["name"] = tool_name
        current_tool["detail"] = detail

    # Initialize tool cache
    tool_cache = ToolCache(ttl=120.0, max_entries=300)

    # Initialize hooks
    hook_manager = HookManager(config)

    # Initialize branch manager
    branch_manager = BranchManager()

    # Initialize custom tools
    custom_tool_registry = CustomToolRegistry()
    for ct in custom_tool_registry.all():
        tools.register(ct)

    # Initialize file watcher (lazy — started by /watch)
    file_watcher = None

    agent = Agent(model, context, tools, permissions, console,
                  stats=session_stats, on_tool_start=_on_tool_start,
                  tool_cache=tool_cache, hooks=hook_manager)

    # Initialize team system — optionally use a faster model for workers
    task_store = TaskStore()
    worker_model_obj = None
    worker_model_name = get(config, "model", "worker_model", default="")
    if worker_model_name:
        worker_endpoint = get(config, "model", "endpoint", default="")
        worker_model_obj = ModelClient(
            endpoint=worker_endpoint,
            model=worker_model_name,
            temperature=get(config, "model", "temperature", default=0.3),
            max_tokens=get(config, "model", "max_tokens", default=8192),
            api_key=get(config, "model", "api_key", default=""),
            provider=get(config, "model", "provider", default="ollama"),
        )
        console.print(f"  [#88c0d0]Workers will use: {worker_model_name}[/#88c0d0]")
    team_manager = TeamManager(model, tools, console, task_store,
                               stats=session_stats, worker_model=worker_model_obj)

    # Give the lead agent the ability to spawn workers
    spawn_tool = SpawnWorkerTool()
    spawn_tool.set_team_manager(team_manager)
    tools.register(spawn_tool)

    # Register wait_for_workers tool (only useful when team is available)
    tools.register(WaitForWorkersTool(team=team_manager))

    # Resume session if requested
    if resume_session:
        if context.load(resume_session):
            meta = Context.read_metadata(resume_session)
            label = meta.get("label", "")
            turns = meta.get("turn_count", 0)
            label_str = f" — {label}" if label else ""
            console.print(f"  [#a3be8c]Resumed session ({turns} turns{label_str})[/#a3be8c]")
            console.print()
        else:
            console.print(f"  [#ebcb8b]Could not load session: {resume_session}[/#ebcb8b]")

    # Callbacks for the prompt_toolkit footer (always visible below input)
    def status_callback():
        """Line 1: model + turns + context %."""
        tokens = context.estimate_tokens()
        turns = context.turn_count
        max_tok = context.max_tokens

        parts = []
        # Show current model name
        model_name = get(config, "model", "name", default="")
        provider_name = get(config, "model", "provider", default="")
        if model_name:
            label = f"  {model_name}"
            if provider_name:
                label += f" ({provider_name})"
            parts.append(("class:bottom-toolbar.team", label))
            parts.append(("class:bottom-toolbar.info", "  "))

        if turns > 0:
            parts.append(("class:bottom-toolbar.info", f"{turns} turns"))
        else:
            parts.append(("class:bottom-toolbar.info", ""))

        # Tokens/sec and cost — right after turns
        if session_stats:
            speed_str = session_stats.format_speed()
            if speed_str:
                parts.append(("class:bottom-toolbar.info", f"  {speed_str}"))

            cost_str = session_stats.format_cost()
            if cost_str:
                parts.append(("class:bottom-toolbar.info", f"  {cost_str}"))

        # Context percentage (right side)
        if max_tok > 0 and tokens > 0:
            pct = max(0, 100 - (tokens / max_tok * 100))
            spacer = " " * 10
            parts.append(("class:bottom-toolbar.context", f"{spacer}Context: {pct:.0f}%"))
        elif tokens > 0:
            parts.append(("class:bottom-toolbar.context", f"    {tokens:,} tokens"))

        return parts

    # Track plan mode separately from permission mode
    plan_mode = {"active": False}

    def mode_switch():
        """Cycle modes: ask → auto → trust → plan → ask (Shift+Tab)."""
        from .ui.input import _MODE_CYCLE
        if plan_mode["active"]:
            current = "plan"
        else:
            current = permissions.mode
        idx = _MODE_CYCLE.index(current) if current in _MODE_CYCLE else 0
        next_mode = _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]
        if next_mode == "plan":
            plan_mode["active"] = True
            permissions.mode = "trust"  # plan mode auto-trusts tools
        else:
            plan_mode["active"] = False
            permissions.mode = next_mode
        config["permissions"]["mode"] = next_mode

    def mode_callback():
        """Line 2: ⏵⏵ mode on · shift+tab to switch · ctrl+t team."""
        if plan_mode["active"]:
            display_mode = "plan"
        else:
            display_mode = permissions.mode
        parts = [
            ("class:bottom-toolbar.mode", "  ⏵⏵ "),
            ("class:bottom-toolbar.mode-text", f"{display_mode} mode on"),
            ("class:bottom-toolbar.info", "  ·  "),
            ("class:bottom-toolbar.info", "shift+tab to switch"),
        ]
        if team_manager.workers:
            active = team_manager.active_count
            total = len(team_manager.workers)
            parts.append(("class:bottom-toolbar.info", "  ·  "))
            parts.append(("class:bottom-toolbar.team", "ctrl+t "))
            parts.append(("class:bottom-toolbar.team-text", f"team ({active}/{total})"))
        return parts

    def team_callback():
        """Line 3+: live worker status (only shown when workers exist)."""
        if not team_manager.workers:
            return None

        # Count by status
        running = sum(1 for w in team_manager.workers.values() if w.status == "running")
        done = sum(1 for w in team_manager.workers.values() if w.status == "completed")
        failed = sum(1 for w in team_manager.workers.values() if w.status == "failed")

        parts = []
        parts.append(("class:bottom-toolbar.team", "  ▸ Team: "))

        status_parts = []
        if running:
            status_parts.append(("class:bottom-toolbar.worker-running", f"{running} running"))
        if done:
            status_parts.append(("class:bottom-toolbar.worker-done", f"{done} done"))
        if failed:
            status_parts.append(("class:bottom-toolbar.worker-failed", f"{failed} failed"))

        for i, (style, text) in enumerate(status_parts):
            if i > 0:
                parts.append(("class:bottom-toolbar.info", " · "))
            parts.append((style, text))

        return parts

    def team_display():
        """Ctrl+T handler — print detailed team status to console."""
        workers = team_manager.status()
        if not workers:
            console.print("\n[#8899aa]  No workers. Use /team <prompt> to spawn one.[/#8899aa]")
            return

        console.print()
        console.print("[bold #88c0d0]  ▸ Team Members[/bold #88c0d0]")
        console.print("[#4c566a]  ─────────────────────────────────────────[/#4c566a]")

        for w in workers:
            if w["status"] == "running":
                icon = "[#ebcb8b]⟳[/#ebcb8b]"
                status_text = "[#ebcb8b]running[/#ebcb8b]"
            elif w["status"] == "completed":
                icon = "[#a3be8c]✓[/#a3be8c]"
                status_text = "[#a3be8c]completed[/#a3be8c]"
            else:
                icon = "[#bf616a]✗[/#bf616a]"
                status_text = "[#bf616a]failed[/#bf616a]"

            console.print(
                f"  {icon} [bold #d8dee9]{w['name']}[/bold #d8dee9]  "
                f"{status_text}"
            )
            console.print(
                f"    [#8899aa]Task: {w['prompt'][:70]}[/#8899aa]"
            )
            if w["result"] and w["status"] != "running":
                result_preview = w["result"][:100].replace("\n", " ")
                console.print(
                    f"    [#4c566a]Result: {result_preview}[/#4c566a]"
                )

        # Messages
        msgs = team_manager.get_lead_messages()
        if msgs:
            console.print()
            console.print("[bold #5e81ac]  ▸ Messages[/bold #5e81ac]")
            console.print("[#4c566a]  ─────────────────────────────────────────[/#4c566a]")
            for m in msgs:
                console.print(
                    f"  [#5e81ac][{m.from_name}][/#5e81ac] "
                    f"[#d8dee9]{m.content[:80]}[/#d8dee9]"
                )

        console.print("[#4c566a]  ─────────────────────────────────────────[/#4c566a]")
        console.print()

    # Auto-display team status during agent execution + Ctrl+T signal handler
    team_monitor = TeamStatusMonitor(
        team_manager.status, console, display_fn=team_display
    )

    # Input session with history, autocomplete, and status footer
    session = create_session(
        skill_names=skills.names(),
        status_callback=status_callback,
        mode_callback=mode_callback,
        team_callback=team_callback,
        team_display_callback=team_display,
        mode_switch_callback=mode_switch,
    )

    # Notification sound config
    notify_enabled = get(config, "ui", "notification_sound", default=True)
    import time as _time

    async def _run_with_notify(coro):
        """Run a coroutine as a cancellable task. Ctrl+C cancels generation."""
        import signal as _signal

        start = _time.monotonic()
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)

        prev_handler = _signal.getsignal(_signal.SIGINT)

        def _cancel_on_sigint(sig, frame):
            agent.cancel()
            task.cancel()
            loop.call_soon_threadsafe(lambda: None)

        _signal.signal(_signal.SIGINT, _cancel_on_sigint)

        # Ensure terminal ISIG is set so Ctrl+C generates SIGINT.
        # prompt_toolkit may leave it cleared after session.prompt()
        # runs in a thread via run_in_executor.
        try:
            import termios as _termios
            _fd = sys.stdin.fileno()
            _attrs = _termios.tcgetattr(_fd)
            if not (_attrs[3] & _termios.ISIG):
                _attrs[3] |= _termios.ISIG
                _termios.tcsetattr(_fd, _termios.TCSANOW, _attrs)
        except Exception:
            pass

        try:
            result = await task
            if notify_enabled and (_time.monotonic() - start) > 5.0:
                _notify_done()
            return result
        except asyncio.CancelledError:
            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
            return ""
        finally:
            _signal.signal(_signal.SIGINT, prev_handler)

    # Handle --continue prompt (send first prompt automatically)
    if continue_prompt:
        try:
            team_monitor.start()
            await _run_with_notify(agent.run(continue_prompt))
        except KeyboardInterrupt:
            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
        except Exception as e:
            console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
        finally:
            team_monitor.stop()

    last_user_message = ""

    try:
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt(),
                )
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("\n[dim]Use /quit to exit[/dim]")
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            # Detect dragged-in image files
            img_path, img_remaining = _is_image_drop(user_input)
            if img_path:
                img_prompt = img_remaining.strip() if img_remaining else "Describe this image and help with anything you see."
                mime_type = mimetypes.guess_type(img_path)[0] or "image/png"
                try:
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    size_kb = len(img_data) * 3 / 4 / 1024
                    console.print(f"  [#88c0d0]Image[/#88c0d0] [#d8dee9]{os.path.basename(img_path)}[/#d8dee9] [#4c566a]({size_kb:.0f} KB, {mime_type})[/#4c566a]")
                    context.add_user_with_image(img_prompt, img_data, mime_type)
                    try:
                        team_monitor.start()
                        await _run_with_notify(agent.run_without_user_add())
                    except KeyboardInterrupt:
                        console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                    finally:
                        team_monitor.stop()
                except OSError as e:
                    console.print(f"[#bf616a]Could not read image: {e}[/#bf616a]")
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                result = handle_slash_command(
                    user_input, context, console, config, skills, model,
                    permissions=permissions,
                    team_manager=team_manager,
                    task_store=task_store,
                    memory=memory,
                    stats=session_stats,
                    pinned=pinned,
                    snippets=snippet_lib,
                )
                if result is None:
                    continue
                if result == "__IMAGE_SENT__":
                    # Image already added to context — run agent without add_user
                    try:
                        team_monitor.start()
                        await _run_with_notify(agent.run_without_user_add())
                    except KeyboardInterrupt:
                        console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                    finally:
                        team_monitor.stop()
                elif result.startswith("__TEAM_SPAWN__"):
                    prompt = result[len("__TEAM_SPAWN__"):]
                    try:
                        await team_manager.spawn(prompt)
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error spawning worker: {e}[/#bf616a]")
                elif result.startswith("__PLAN_EXECUTE__"):
                    plan_content = result[len("__PLAN_EXECUTE__"):]
                    try:
                        team_monitor.start()
                        await execute_plan(
                            plan_content, team_manager, agent, console
                        )
                    except KeyboardInterrupt:
                        console.print("\n[#ebcb8b]Plan interrupted[/#ebcb8b]")
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                    finally:
                        team_monitor.stop()
                elif result.startswith("__RUN_CMD__"):
                    run_command = result[len("__RUN_CMD__"):]
                    run_prompt = (
                        f"Run this command in the current directory:\n\n"
                        f"```\n{run_command}\n```\n\n"
                        f"Use the bash tool. Show the output."
                    )
                    try:
                        team_monitor.start()
                        await _run_with_notify(agent.run(run_prompt))
                    except KeyboardInterrupt:
                        console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                    finally:
                        team_monitor.stop()
                elif result.startswith("__TEAM_STOP__"):
                    stop_id = result[len("__TEAM_STOP__"):]
                    try:
                        if stop_id:
                            ok = await team_manager.stop(stop_id)
                            if not ok:
                                console.print(f"[#ebcb8b]Worker #{stop_id} not found[/#ebcb8b]")
                        else:
                            await team_manager.stop_all()
                            console.print("[#a3be8c]All workers stopped[/#a3be8c]")
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error stopping worker: {e}[/#bf616a]")
                elif result.startswith("__WATCH__"):
                    watch_cmd = result[len("__WATCH__"):]
                    if watch_cmd.lower() == "off":
                        if file_watcher and file_watcher.is_running:
                            await file_watcher.stop()
                            file_watcher = None
                        else:
                            console.print("[#8899aa]No watcher running.[/#8899aa]")
                    else:
                        if file_watcher and file_watcher.is_running:
                            await file_watcher.stop()
                        file_watcher = FileWatcher(watch_cmd, console)
                        await file_watcher.start()

                elif result == "__PROFILE__":
                    # Model performance benchmark
                    import time as _t
                    console.print("[#88c0d0]Running benchmark...[/#88c0d0]")
                    bench_prompt = "Write a Python function that checks if a number is prime. Include type hints."
                    bench_ctx = Context(system_prompt="You are a coding assistant.", max_tokens=4096)
                    bench_ctx.add_user(bench_prompt)
                    start_t = _t.monotonic()
                    first_token_t = None
                    total_chars = 0
                    async for chunk in model.chat(bench_ctx.get_messages(), tools=None, stream=True):
                        if chunk["type"] == "text":
                            if first_token_t is None:
                                first_token_t = _t.monotonic()
                            total_chars += len(chunk["content"])
                    end_t = _t.monotonic()
                    ttft = (first_token_t - start_t) if first_token_t else 0
                    total_time = end_t - start_t
                    est_tokens = total_chars / 4
                    tps = est_tokens / total_time if total_time > 0 else 0
                    console.print(f"  [#88c0d0]TTFT:[/#88c0d0] [#d8dee9]{ttft:.2f}s[/#d8dee9]")
                    console.print(f"  [#88c0d0]Total time:[/#88c0d0] [#d8dee9]{total_time:.2f}s[/#d8dee9]")
                    console.print(f"  [#88c0d0]Output:[/#88c0d0] [#d8dee9]~{int(est_tokens)} tokens ({total_chars} chars)[/#d8dee9]")
                    console.print(f"  [#88c0d0]Speed:[/#88c0d0] [#a3be8c]{tps:.1f} tokens/sec[/#a3be8c]")
                    console.print(f"  [#88c0d0]Model:[/#88c0d0] [#d8dee9]{get(config, 'model', 'name')}[/#d8dee9]")

                elif result == "__BENCHMARK__":
                    import time as _btime
                    model_name = get(config, "model", "name", default="unknown")
                    console.print(f"  [#88c0d0]Benchmarking {model_name}...[/#88c0d0]")
                    bench_prompt = "Write a Python function that checks if a number is prime."
                    bench_msgs = [{"role": "user", "content": bench_prompt}]
                    first_token_time = None
                    token_count = 0
                    start = _btime.monotonic()
                    try:
                        async for chunk in model.chat(
                            messages=bench_msgs, tools=[], stream=True
                        ):
                            if chunk["type"] == "text":
                                if first_token_time is None:
                                    first_token_time = _btime.monotonic()
                                token_count += max(1, len(chunk["content"].split()))
                            elif chunk["type"] == "done":
                                break
                        end = _btime.monotonic()
                        ttft = (first_token_time - start) if first_token_time else 0
                        total_time = end - start
                        gen_time = (end - first_token_time) if first_token_time else 0
                        speed = token_count / gen_time if gen_time > 0 else 0
                        console.print(f"  [#a3be8c]Time to first token: {ttft:.1f}s[/#a3be8c]")
                        console.print(f"  [#a3be8c]Generation speed: {speed:.1f} tok/s[/#a3be8c]")
                        console.print(f"  [#a3be8c]Total: {token_count} tokens in {total_time:.1f}s[/#a3be8c]")
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        console.print(f"\n  [#ebcb8b]Benchmark cancelled.[/#ebcb8b]")
                    except Exception as e:
                        console.print(f"  [#bf616a]Benchmark failed: {e}[/#bf616a]")
                    continue

                elif result.startswith("__TEACH__"):
                    teach_args = result[len("__TEACH__"):]
                    # Parse: name "description" -- command
                    if " -- " in teach_args:
                        before, command_str = teach_args.split(" -- ", 1)
                        parts = before.strip().split(maxsplit=1)
                        tool_name = parts[0] if parts else ""
                        tool_desc = parts[1].strip('"\'') if len(parts) > 1 else f"Custom tool: {tool_name}"
                    else:
                        parts = teach_args.strip().split(maxsplit=2)
                        tool_name = parts[0] if parts else ""
                        tool_desc = parts[1].strip('"\'') if len(parts) > 1 else f"Custom tool: {tool_name}"
                        command_str = parts[2] if len(parts) > 2 else ""

                    if not tool_name or not command_str:
                        console.print("[#ebcb8b]Usage: /teach <name> <description> -- <command>[/#ebcb8b]")
                    else:
                        ct = custom_tool_registry.add(tool_name, tool_desc, command_str)
                        tools.register(ct)
                        console.print(f"[#a3be8c]Taught: {tool_name} — {tool_desc}[/#a3be8c]")
                        console.print(f"[#8899aa]Command: {command_str}[/#8899aa]")

                elif result.startswith("__BRANCH__"):
                    branch_args = result[len("__BRANCH__"):]
                    if not branch_args:
                        # List branches
                        branches = branch_manager.list_branches()
                        if not branches:
                            console.print("[#8899aa]No branches yet. Use /branch <name>[/#8899aa]")
                        else:
                            console.print("[bold #eceff4]Branches:[/bold #eceff4]")
                            for b in branches:
                                marker = " [#a3be8c]<- current[/#a3be8c]" if b["current"] else ""
                                console.print(
                                    f"  [#88c0d0]{b['name']}[/#88c0d0]  "
                                    f"[#8899aa]{b['turns']} turns[/#8899aa]{marker}")
                    else:
                        msg = branch_manager.create_branch(branch_args, context, os.getcwd())
                        console.print(f"[#a3be8c]{msg}[/#a3be8c]")

                elif result.startswith("__SWITCH__"):
                    branch_name = result[len("__SWITCH__"):]
                    # Save current branch first
                    branch_manager.save_branch(branch_manager.current, context, os.getcwd())
                    ok, msg = branch_manager.switch_branch(branch_name, context)
                    style = "#a3be8c" if ok else "#bf616a"
                    console.print(f"[{style}]{msg}[/{style}]")

                elif result == "__BRANCHES__":
                    branches = branch_manager.list_branches()
                    if not branches:
                        console.print("[#8899aa]No branches. Use /branch <name> to create one.[/#8899aa]")
                    else:
                        console.print("[bold #eceff4]Conversation branches:[/bold #eceff4]")
                        for b in branches:
                            marker = " [#a3be8c]<- current[/#a3be8c]" if b["current"] else ""
                            parent = f" [#4c566a](from {b['parent']})[/#4c566a]" if b.get("parent") else ""
                            console.print(
                                f"  [#88c0d0]{b['name']}[/#88c0d0]  "
                                f"[#8899aa]{b['turns']} turns[/#8899aa]"
                                f"{parent}{marker}")
                        console.print("[#8899aa]Use /switch <name> to switch branches[/#8899aa]")

                elif result.startswith("__SHARE__"):
                    share_format = result[len("__SHARE__"):].strip() or "html"
                    lines = ["<!DOCTYPE html><html><head>",
                             "<meta charset='utf-8'>",
                             "<title>Spark Code Session</title>",
                             "<style>",
                             "body{font-family:monospace;background:#2e3440;color:#d8dee9;max-width:800px;margin:0 auto;padding:20px}",
                             ".user{color:#88c0d0;font-weight:bold}.assistant{color:#d8dee9}",
                             ".tool{color:#a3be8c;font-style:italic}pre{background:#3b4252;padding:10px;border-radius:4px;overflow-x:auto}",
                             "h1{color:#ebcb8b}h2{color:#88c0d0;border-bottom:1px solid #4c566a}",
                             "</style></head><body>",
                             "<h1>Spark Code Session</h1>"]
                    for msg in context.messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "") or ""
                        if role == "user" and isinstance(content, str):
                            lines.append(f"<h2 class='user'>User</h2><p>{content[:2000]}</p>")
                        elif role == "assistant" and isinstance(content, str) and content:
                            lines.append(f"<h2 class='assistant'>Assistant</h2><p>{content[:2000]}</p>")
                        elif role == "tool":
                            name = msg.get("name", "tool")
                            preview = (content[:500] + "...") if len(content) > 500 else content
                            lines.append(f"<p class='tool'>Tool: {name}</p><pre>{preview}</pre>")
                    lines.append("</body></html>")
                    html_content = "\n".join(lines)

                    if share_format == "gist":
                        # Export as markdown for gist
                        md_lines = [f"# Spark Code Session\n"]
                        for msg in context.messages:
                            role = msg.get("role", "")
                            content = msg.get("content", "") or ""
                            if role == "user" and isinstance(content, str):
                                md_lines.append(f"## User\n\n{content[:2000]}\n")
                            elif role == "assistant" and isinstance(content, str):
                                md_lines.append(f"## Assistant\n\n{content[:2000]}\n")
                        md_text = "\n".join(md_lines)
                        share_path = os.path.join(os.getcwd(), "session_share.md")
                        with open(share_path, "w") as f:
                            f.write(md_text)
                        console.print(f"[#a3be8c]Saved to {share_path}[/#a3be8c]")
                        console.print("[#8899aa]Create a gist: gh gist create session_share.md[/#8899aa]")
                    else:
                        share_path = os.path.join(os.getcwd(), "session_share.html")
                        with open(share_path, "w") as f:
                            f.write(html_content)
                        console.print(f"[#a3be8c]Saved to {share_path}[/#a3be8c]")

                elif result.startswith("__SEARCH__"):
                    query = result[len("__SEARCH__"):].lower()
                    history_dir = os.path.expanduser("~/.spark/history")
                    if not os.path.isdir(history_dir):
                        console.print("[#8899aa]No saved sessions.[/#8899aa]")
                    else:
                        import json as _json
                        matches = []
                        for f in sorted(os.listdir(history_dir), reverse=True):
                            if not f.endswith(".json"):
                                continue
                            path = os.path.join(history_dir, f)
                            try:
                                with open(path, encoding="utf-8") as fh:
                                    data = _json.load(fh)
                                for msg in data.get("messages", []):
                                    content = msg.get("content", "")
                                    if isinstance(content, str) and query in content.lower():
                                        label = data.get("label", f.replace(".json", ""))
                                        # Extract matching snippet
                                        idx = content.lower().index(query)
                                        start = max(0, idx - 40)
                                        end = min(len(content), idx + len(query) + 40)
                                        snippet = content[start:end].replace("\n", " ")
                                        matches.append((label, snippet, f))
                                        break
                            except Exception:
                                pass
                            if len(matches) >= 10:
                                break

                        if not matches:
                            console.print(f"[#8899aa]No sessions matching '{query}'[/#8899aa]")
                        else:
                            console.print(f"[bold #eceff4]Sessions matching '{query}':[/bold #eceff4]")
                            for label, snippet, fname in matches:
                                console.print(f"  [#88c0d0]{label}[/#88c0d0]")
                                console.print(f"    [#8899aa]...{snippet}...[/#8899aa]")
                            console.print("[#8899aa]Use /history <name> to resume[/#8899aa]")

                elif result == "__ANALYTICS__":
                    if not stats:
                        console.print("[#8899aa]No stats available.[/#8899aa]")
                    else:
                        # Generate analytics report
                        console.print()
                        console.print("[bold #eceff4]Session Analytics[/bold #eceff4]")
                        console.print("[#4c566a]─────────────────────────────────[/#4c566a]")
                        console.print(f"  [#88c0d0]Duration:[/#88c0d0] {stats.format_duration()}")
                        console.print(f"  [#88c0d0]Turns:[/#88c0d0] {context.turn_count}")
                        console.print(f"  [#88c0d0]Total tool calls:[/#88c0d0] {stats.total_tool_calls}")
                        console.print()

                        # Tool usage breakdown
                        if stats.tool_calls:
                            console.print("  [bold #eceff4]Tool Usage[/bold #eceff4]")
                            max_count = max(stats.tool_calls.values())
                            for name, count in sorted(stats.tool_calls.items(), key=lambda x: -x[1]):
                                bar_len = int(count / max_count * 20) if max_count else 0
                                bar = "[#a3be8c]" + "█" * bar_len + "[/#a3be8c]" + "[#4c566a]" + "░" * (20 - bar_len) + "[/#4c566a]"
                                console.print(f"    {name:<15} {bar} {count}")

                        # Files touched
                        all_files = stats.files_read | stats.files_written | stats.files_edited
                        if all_files:
                            console.print()
                            console.print("  [bold #eceff4]Files Touched[/bold #eceff4]")
                            home = os.path.expanduser("~")
                            for fpath in sorted(all_files)[:15]:
                                display = "~" + fpath[len(home):] if fpath.startswith(home) else fpath
                                tags = []
                                if fpath in stats.files_read:
                                    tags.append("[#88c0d0]R[/#88c0d0]")
                                if fpath in stats.files_written:
                                    tags.append("[#a3be8c]W[/#a3be8c]")
                                if fpath in stats.files_edited:
                                    tags.append("[#ebcb8b]E[/#ebcb8b]")
                                console.print(f"    {''.join(tags)} {display}")

                        # Token usage
                        console.print()
                        console.print("  [bold #eceff4]Tokens[/bold #eceff4]")
                        console.print(f"    Input:  {model.total_input_tokens:,}")
                        console.print(f"    Output: {model.total_output_tokens:,}")
                        cost = model.estimated_cost
                        if cost > 0:
                            console.print(f"    Cost:   ${cost:.4f}")

                        # Cache stats
                        if tool_cache:
                            cs = tool_cache.stats
                            console.print()
                            console.print("  [bold #eceff4]Cache[/bold #eceff4]")
                            console.print(f"    Entries: {cs['entries']}  Hits: {cs['hits']}  Misses: {cs['misses']}  Rate: {cs['hit_rate']}")

                elif result == "__RETRY__":
                    if not last_user_message:
                        console.print("  [#ebcb8b]No previous message to retry.[/#ebcb8b]")
                    else:
                        console.print(f"  [#88c0d0]Retrying: {last_user_message[:60]}...[/#88c0d0]")
                        try:
                            team_monitor.start()
                            await _run_with_notify(agent.run(last_user_message))
                        except KeyboardInterrupt:
                            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                        except Exception as e:
                            console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                        finally:
                            team_monitor.stop()
                    continue

                elif result == "__CONTINUE__":
                    from spark_code.agent import load_checkpoint, CHECKPOINT_DIR
                    checkpoint_path = str(CHECKPOINT_DIR / "latest.json")
                    data = load_checkpoint(checkpoint_path)
                    if not data:
                        console.print("  [#ebcb8b]No checkpoint found.[/#ebcb8b]")
                    else:
                        context.messages = data["messages"]
                        saved_cwd = data.get("cwd", "")
                        if saved_cwd and saved_cwd != os.getcwd():
                            console.print(f"  [#ebcb8b]Note: CWD changed. Checkpoint was in {saved_cwd}[/#ebcb8b]")
                        context.messages.append({
                            "role": "system",
                            "content": f"Session resumed from checkpoint at round {data['round_count']}/{Agent.MAX_TOOL_ROUNDS}. Continue where you left off."
                        })
                        console.print(f"  [#a3be8c]Checkpoint restored ({len(data['messages'])} messages). Resuming...[/#a3be8c]")
                        try:
                            team_monitor.start()
                            await _run_with_notify(agent.run_without_user_add())
                        except KeyboardInterrupt:
                            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                        except Exception as e:
                            console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                        finally:
                            team_monitor.stop()

                elif result == "__CLEAN__":
                    if not session_stats or not session_stats.files_created:
                        console.print("  [#ebcb8b]No files were created this session.[/#ebcb8b]")
                    else:
                        existing = []
                        for f in sorted(session_stats.files_created):
                            if os.path.exists(f):
                                try:
                                    lines = len(open(f).readlines())
                                except Exception:
                                    lines = 0
                                existing.append((f, lines))
                        if not existing:
                            console.print("  [#ebcb8b]All created files have already been deleted.[/#ebcb8b]")
                        else:
                            console.print("  [#88c0d0]Files created this session:[/#88c0d0]")
                            for path, lines in existing:
                                short = path.replace(os.getcwd() + "/", "")
                                console.print(f"    {short} ({lines} lines)")
                            console.print()
                            answer = await session.prompt_async("  Delete all? [y/N/select] ")
                            answer = answer.strip().lower()
                            if answer == "y":
                                for path, _ in existing:
                                    try:
                                        os.remove(path)
                                        console.print(f"  [#a3be8c]Deleted {os.path.basename(path)}[/#a3be8c]")
                                    except Exception as e:
                                        console.print(f"  [#bf616a]Error deleting {path}: {e}[/#bf616a]")
                            elif answer == "select":
                                for path, lines in existing:
                                    short = path.replace(os.getcwd() + "/", "")
                                    ans = await session.prompt_async(f"  Delete {short}? [y/N] ")
                                    if ans.strip().lower() == "y":
                                        try:
                                            os.remove(path)
                                            console.print(f"  [#a3be8c]Deleted {os.path.basename(path)}[/#a3be8c]")
                                        except Exception as e:
                                            console.print(f"  [#bf616a]Error: {e}[/#bf616a]")
                            else:
                                console.print("  Cancelled.")
                    continue

                elif result.startswith("__MODEL_SWITCH__"):
                    provider_name = result[len("__MODEL_SWITCH__"):]
                    providers = config.get("providers", {})
                    if provider_name not in providers:
                        # Check built-in providers
                        if provider_name in PROVIDERS:
                            console.print(f"[#ebcb8b]Provider '{provider_name}' is built-in but not configured in your config.yaml[/#ebcb8b]")
                            console.print("[#8899aa]Add it to your config.yaml under 'providers:' to use it[/#8899aa]")
                        else:
                            console.print(f"[#bf616a]Unknown provider: {provider_name}[/#bf616a]")
                            if providers:
                                console.print(f"[#8899aa]Available: {', '.join(providers.keys())}[/#8899aa]")
                        continue

                    pconf = providers[provider_name]
                    # Close old client
                    await model.close()
                    # Create new client
                    model = ModelClient(
                        endpoint=pconf.get("endpoint", "http://localhost:11434"),
                        model=pconf.get("model", "unknown"),
                        temperature=pconf.get("temperature", 0.7),
                        max_tokens=pconf.get("max_tokens", 8192),
                        api_key=pconf.get("api_key", ""),
                        provider=provider_name,
                        timeout=float(pconf.get("timeout", 300)),
                    )
                    # Update config
                    config["model"]["name"] = pconf.get("model", "unknown")
                    config["model"]["endpoint"] = pconf.get("endpoint", "http://localhost:11434")
                    config["model"]["provider"] = provider_name

                    # Update agent's model reference
                    agent.model = model
                    console.print(f"[#a3be8c]Switched to {provider_name} ({pconf.get('model', '?')})[/#a3be8c]")
                    console.print("[#8899aa]Conversation context preserved[/#8899aa]")
                else:
                    # Skill returned a prompt — send to agent
                    is_projectplan = "projectplan.md" in result
                    user_input = result
                    try:
                        team_monitor.start()
                        await _run_with_notify(agent.run(user_input))
                    except KeyboardInterrupt:
                        console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                    finally:
                        team_monitor.stop()
                    # Post-run status for /projectplan
                    if is_projectplan:
                        pp_path = os.path.join(os.getcwd(), "projectplan.md")
                        if os.path.exists(pp_path):
                            console.print(f"\n[#a3be8c]▸ projectplan.md created successfully[/#a3be8c]")
                            console.print(f"[#88c0d0]  /projectplan show  to review  ·  /projectplan go  to execute[/#88c0d0]")
                        else:
                            console.print(f"\n[#bf616a]▸ projectplan.md was NOT created — model may have failed[/#bf616a]")
                            console.print(f"[#8899aa]  Try again or use /projectplan <prompt> with a simpler request[/#8899aa]")
            elif _is_shell_command(user_input):
                # Direct shell command — run via agent with explicit instruction
                run_prompt = (
                    f"Run this command now with the bash tool:\n\n"
                    f"```\n{user_input}\n```\n\n"
                    f"Just run it and show the output. Do not ask questions."
                )
                try:
                    team_monitor.start()
                    await _run_with_notify(agent.run(run_prompt))
                except KeyboardInterrupt:
                    console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                except Exception as e:
                    console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                finally:
                    team_monitor.stop()
            else:
                # Refresh pinned files before each turn
                if pinned.count > 0:
                    pinned.refresh()
                    pinned_ctx = pinned.get_context()
                    # Inject pinned files into the system prompt temporarily
                    if pinned_ctx and pinned_ctx not in context.system_prompt:
                        context.system_prompt = context.system_prompt.rstrip() + "\n\n" + pinned_ctx

                last_user_message = user_input

                # Auto-detect error pastes and enhance the prompt
                if _is_error_paste(user_input):
                    user_input = (
                        f"The user pasted an error. Diagnose it and suggest a fix.\n\n"
                        f"```\n{user_input}\n```"
                    )

                # Auto-read files mentioned in the prompt
                mentioned = _detect_file_mentions(user_input)
                if mentioned:
                    file_context = []
                    for fpath in mentioned[:3]:  # Max 3 auto-reads
                        try:
                            with open(fpath, encoding="utf-8", errors="replace") as f:
                                content = f.read(10000)
                            home = os.path.expanduser("~")
                            display = "~" + fpath[len(home):] if fpath.startswith(home) else fpath
                            file_context.append(f"Contents of {display}:\n```\n{content}\n```")
                            console.print(f"  [#4c566a]Auto-read: {display}[/#4c566a]")
                        except OSError:
                            pass
                    if file_context:
                        user_input += "\n\n" + "\n\n".join(file_context)

                # In plan mode, wrap prompt to plan first
                # Skip wrapping for short conversational messages
                if plan_mode["active"] and len(user_input.split()) > 3:
                    user_input = (
                        f"The user wants you to PLAN before executing. "
                        f"Their request: {user_input}\n\n"
                        "IMPORTANT: Do NOT execute changes yet. Instead:\n"
                        "1. Use read-only tools to explore the codebase\n"
                        "2. Create a detailed step-by-step plan\n"
                        "3. Present the plan and ask for approval before executing\n"
                        "4. Only after the user approves, execute the plan\n"
                    )
                # Run agent
                try:
                    team_monitor.start()
                    await _run_with_notify(agent.run(user_input))
                except KeyboardInterrupt:
                    console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                except Exception as e:
                    console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                finally:
                    team_monitor.stop()


            # Context limit warning
            tokens = context.estimate_tokens()
            if context.max_tokens > 0:
                pct = tokens / context.max_tokens * 100
                if pct > 75 and pct <= 80:
                    console.print(f"[#ebcb8b]Context usage: {pct:.0f}% — consider /compact[/#ebcb8b]")

            # Auto-compact if getting large
            if context.estimate_tokens() > context.max_tokens * 0.8:
                compact_msg = context.compact()
                if compact_msg:
                    console.print(f"  [#ebcb8b]⚡ {compact_msg}[/#ebcb8b]")

    finally:
        # Auto-save conversation history with label and cwd
        if context.turn_count > 0:
            try:
                from datetime import datetime
                history_dir = os.path.expanduser("~/.spark/history")
                os.makedirs(history_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                label = _make_session_label(context)
                suffix = f"_{label}" if label else ""
                context.save(
                    os.path.join(history_dir, f"{ts}{suffix}.json"),
                    label=label,
                    cwd=os.getcwd(),
                )
            except Exception:
                pass  # Don't crash on save failure

        await team_manager.stop_all()
        await model.close()
        await mcp_client.disconnect_all()
        console.print("[dim]Session ended.[/dim]")


def _run_setup():
    """Interactive setup wizard for first-time configuration."""
    import yaml as _yaml

    from .config import GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_FILE

    console = Console()
    console.print()
    console.print("[bold #ebcb8b]  Spark Code Setup[/bold #ebcb8b]")
    console.print("[#4c566a]  ─────────────────────────────────────────[/#4c566a]")
    console.print()

    # Provider presets — built from shared _PROVIDER_INFO with setup-specific extras
    _SETUP_EXTRAS = {
        "gemini": {
            "label": "Google Gemini (recommended — fast, free tier)",
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
            "context_window": 1000000,
        },
        "openai": {
            "label": "OpenAI (GPT-4o, GPT-4o-mini)",
            "endpoint": "https://api.openai.com/v1",
            "context_window": 128000,
        },
        "groq": {
            "label": "Groq (fast Llama 3.3 inference, free tier)",
            "endpoint": "https://api.groq.com/openai/v1",
            "context_window": 128000,
        },
        "deepseek": {
            "label": "DeepSeek (cheap, strong at code)",
            "endpoint": "https://api.deepseek.com/v1",
            "context_window": 64000,
        },
        "openrouter": {
            "label": "OpenRouter (100+ models, one API key)",
            "endpoint": "https://openrouter.ai/api/v1",
            "context_window": 200000,
        },
        "ollama": {
            "label": "Ollama (local, no API key needed)",
            "endpoint": "http://localhost:11434",
            "context_window": 32768,
        },
    }
    presets = {}
    for i, p in enumerate(_PROVIDER_INFO, 1):
        presets[str(i)] = {**p, **_SETUP_EXTRAS[p["name"]]}

    console.print("  Choose a provider:\n")
    for key, preset in presets.items():
        console.print(f"    [#88c0d0]{key}[/#88c0d0]  {preset['label']}")
    console.print()

    choice = input("  Enter number (1-6): ").strip()
    if choice not in presets:
        console.print("[#bf616a]  Invalid choice. Run 'spark --setup' again.[/#bf616a]")
        return

    preset = presets[choice]
    console.print(f"\n  [#a3be8c]Selected: {preset['label']}[/#a3be8c]\n")

    # Get API key
    api_key_value = ""
    env_var = preset["env_var"]
    if env_var:
        console.print(f"  Get your API key at: [#5e81ac]{preset['signup']}[/#5e81ac]\n")
        api_key_value = input(f"  Paste your {env_var}: ").strip()
        if not api_key_value:
            console.print("[#ebcb8b]  No key entered. You can set it later in ~/.zshrc[/#ebcb8b]")
    else:
        console.print("  [#8899aa]No API key needed for local Ollama.[/#8899aa]")

    # Optional: custom model
    console.print(f"\n  Default model: [#88c0d0]{preset['model']}[/#88c0d0]")
    custom_model = input("  Custom model (or Enter to keep default): ").strip()
    if custom_model:
        preset["model"] = custom_model

    # Build config
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing config if any
    existing = {}
    if GLOBAL_CONFIG_FILE.exists():
        with open(GLOBAL_CONFIG_FILE) as f:
            existing = _yaml.safe_load(f) or {}

    # Set up provider
    if "providers" not in existing:
        existing["providers"] = {}

    provider_conf = {
        "endpoint": preset["endpoint"],
        "model": preset["model"],
        "context_window": preset["context_window"],
        "max_tokens": 8192,
        "temperature": 0.7,
    }
    if env_var:
        provider_conf["api_key"] = f"${{{env_var}}}"

    existing["providers"][preset["name"]] = provider_conf
    existing["active_provider"] = preset["name"]

    # Write config
    with open(GLOBAL_CONFIG_FILE, "w") as f:
        _yaml.dump(existing, f, default_flow_style=False)

    console.print(f"\n  [#a3be8c]✓ Config saved to {GLOBAL_CONFIG_FILE}[/#a3be8c]")

    # Set env var
    if env_var and api_key_value:
        shell_rc = os.path.expanduser("~/.zshrc")
        export_line = f'export {env_var}="{api_key_value}"'
        # Check if already set
        try:
            with open(shell_rc, encoding="utf-8") as f:
                rc_content = f.read()
            if env_var not in rc_content:
                with open(shell_rc, "a", encoding="utf-8") as f:
                    f.write(f"\n# Spark Code — {preset['name']}\n{export_line}\n")
                console.print(f"  [#a3be8c]✓ {env_var} added to {shell_rc}[/#a3be8c]")
            else:
                console.print(f"  [#8899aa]{env_var} already in {shell_rc}[/#8899aa]")
        except OSError:
            console.print(f"  [#ebcb8b]Add this to your {shell_rc}:[/#ebcb8b]")
            console.print(f"    {export_line}")

        # Set for current session too
        os.environ[env_var] = api_key_value

    console.print("\n  [bold #a3be8c]Setup complete![/bold #a3be8c]")
    console.print("  Run [bold]spark[/bold] to start coding.\n")
    if env_var and api_key_value:
        console.print("  [#8899aa]Run 'source ~/.zshrc' or open a new terminal first.[/#8899aa]\n")


@click.command()
@click.option("--endpoint", "-e", help="Model API endpoint URL")
@click.option("--model", "-m", "model_name", help="Model name")
@click.option("--provider", "-p", help="Provider: ollama, gemini, openai")
@click.option("--trust", is_flag=True, help="Trust mode (allow all tool calls)")
@click.option("--auto", "auto_mode", is_flag=True, help="Auto mode (allow reads, ask for writes)")
@click.option("--yolo", is_flag=True, help="Full agent mode (trust + autonomous execution)")
@click.option("--resume", "-r", is_flag=True, help="Resume the most recent session")
@click.option("--continue", "-c", "continue_prompt", default="", help="Resume last session and send a prompt")
@click.option("--setup", is_flag=True, help="Run interactive setup wizard")
@click.option("--version", "-v", is_flag=True, help="Show version")
@click.argument("prompt", nargs=-1, required=False)
def main(endpoint, model_name, provider, trust, auto_mode, yolo, resume,
         continue_prompt, setup, version, prompt):
    """Spark Code — Your local AI coding assistant."""
    if version:
        click.echo(f"Spark Code v{__version__}")
        return

    if setup:
        _run_setup()
        return

    # Load config with provider selection
    config = load_config(os.getcwd(), provider=provider)

    # CLI overrides
    if endpoint:
        config["model"]["endpoint"] = endpoint
    if model_name:
        config["model"]["name"] = model_name
    if yolo or trust:
        config["permissions"]["mode"] = "trust"
    elif auto_mode:
        config["permissions"]["mode"] = "auto"

    # Agentic mode flag
    config["_agentic"] = yolo

    # Resume / continue mode
    resume_session = ""
    if resume or continue_prompt:
        resume_session = _get_latest_session()
        if not resume_session:
            click.echo("No previous session to resume.")
            if not continue_prompt:
                # Fall through to interactive mode without resume
                pass

    # One-shot mode (skip if --resume or --continue)
    if prompt and not resume and not continue_prompt:
        prompt_text = " ".join(prompt)
        asyncio.run(_one_shot(config, prompt_text))
        return

    # Interactive mode
    asyncio.run(run_interactive(
        config,
        resume_session=resume_session,
        continue_prompt=continue_prompt,
    ))


async def _one_shot(config: dict, prompt: str):
    """Run a single prompt and exit."""
    console = Console(theme=get_theme())
    model = ModelClient(
        endpoint=get(config, "model", "endpoint"),
        model=get(config, "model", "name"),
        temperature=get(config, "model", "temperature", default=0.7),
        max_tokens=get(config, "model", "max_tokens", default=8192),
        api_key=get(config, "model", "api_key", default=""),
        provider=get(config, "model", "provider", default="ollama"),
        timeout=float(get(config, "model", "timeout", default=300)),
    )
    context = Context()
    tools = build_tools()
    permissions = PermissionManager(mode="auto")
    agent = Agent(model, context, tools, permissions, console)

    try:
        await agent.run(prompt)
    finally:
        await model.close()


if __name__ == "__main__":
    main()
