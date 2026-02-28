"""Spark Code CLI — entry point."""

import asyncio
import base64
import mimetypes
import os
import sys

import click
from rich.box import ROUNDED
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import __version__, __app_name__
from .config import load_config, get, ensure_dirs
from .model import ModelClient
from .context import Context, SYSTEM_PROMPT
from .agent import Agent
from .memory import Memory
from .permissions import PermissionManager
from .tools.base import ToolRegistry
from .tools.read_file import ReadFileTool
from .tools.write_file import WriteFileTool
from .tools.edit_file import EditFileTool
from .tools.bash import BashTool
from .tools.glob_search import GlobTool
from .tools.grep_search import GrepTool
from .tools.list_dir import ListDirTool
from .tools.web_search import WebSearchTool
from .tools.web_fetch import WebFetchTool
from .tools.spawn_worker import SpawnWorkerTool
from .skills.base import SkillRegistry
from .task_store import TaskStore
from .team import TeamManager
from .plan_executor import execute_plan
from .mcp.client import MCPClient
from .ui.input import create_session
from .ui.hotkeys import TeamStatusMonitor
from .ui.theme import get_theme


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    import subprocess
    try:
        proc = subprocess.run(
            ["pbcopy"], input=text.encode(), check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


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
                 skill_count: int = 0):
    """Print startup banner — two-column layout matching Claude Code."""
    model_name = get(config, "model", "name", default="unknown")
    provider = get(config, "model", "provider", default="")
    perm_mode = get(config, "permissions", "mode", default="ask")
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]

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

    # Right column: tips and info
    right = Text()
    right.append("Tips for getting started\n", style="bold #eceff4")
    right.append("Run ", style="#666666")
    right.append("/help", style="bold #d8dee9")
    right.append(" for available commands\n", style="#666666")

    right.append("─" * 35 + "\n", style="#3b4252")

    right.append("Capabilities\n", style="bold #eceff4")
    right.append("Read, write, and edit files\n", style="#666666")
    right.append("Run shell commands\n", style="#666666")
    right.append("Search code with glob/grep\n", style="#666666")
    right.append("Web search and fetch\n", style="#666666")
    right.append("Send images with /image\n", style="#666666")

    if mcp_count > 0 or skill_count > 0:
        right.append("─" * 35 + "\n", style="#3b4252")
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
                         task_store: TaskStore | None = None) -> str | None:
    """Handle slash commands.
    Returns None if handled (no agent needed), or a prompt string for the agent.
    Returns "__ASYNC__" for commands that schedule async work (team spawn).
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        help_text = """## Commands
- `/help` — Show this help
- `/clear` — Clear conversation
- `/compact` — Summarize conversation to save context
- `/config` — Show current config
- `/model` — Show model info
- `/tokens` — Show token usage
- `/image <path> [prompt]` — Send an image with optional prompt
- `/mode [ask|auto|trust]` — Switch permission mode
- `/trust` — Switch to trust mode (allow all tools)
- `/auto` — Switch to auto mode (allow reads, ask for writes)
- `/ask` — Switch to ask mode (confirm all tools)
- `/team <prompt>` — Spawn a background worker agent
- `/team status` — Show worker status
- `/team stop [id]` — Stop a worker (or all)
- `/team msg <name> <message>` — Send a message to a worker
- `/plan <prompt>` — Create a plan.md before executing
- `/plan show` — Show the current plan
- `/plan go` — Execute the approved plan
- `/run [command]` — Run the project (auto-detects or specify command)
- `/tasks` — Show the shared task list
- `/messages` — Check messages from workers
- `/quit` or `/exit` — Exit

## Skills"""
        for skill in skills.all():
            help_text += f"\n- `/{skill.name}` — {skill.description}"
        console.print(Markdown(help_text))
        return None

    elif command == "/clear":
        context.clear()
        console.print("[green]Conversation cleared.[/green]")
        return None

    elif command == "/compact":
        before = context.estimate_tokens()
        context.compact()
        after = context.estimate_tokens()
        console.print(f"[green]Compacted: ~{before:,} → ~{after:,} tokens[/green]")
        return None

    elif command == "/config":
        import yaml
        console.print(Markdown(f"```yaml\n{yaml.dump(config, default_flow_style=False)}```"))
        return None

    elif command == "/model":
        console.print(f"Model: {get(config, 'model', 'name')}")
        console.print(f"Endpoint: {get(config, 'model', 'endpoint')}")
        console.print(f"Temperature: {get(config, 'model', 'temperature')}")
        console.print(f"Input tokens: {model.total_input_tokens:,}")
        console.print(f"Output tokens: {model.total_output_tokens:,}")
        return None

    elif command == "/tokens":
        tokens = context.estimate_tokens()
        max_tokens = context.max_tokens
        pct = tokens / max_tokens * 100 if max_tokens else 0
        console.print(f"Context: ~{tokens:,} / {max_tokens:,} tokens ({pct:.0f}%)")
        console.print(f"Turns: {context.turn_count}")
        console.print(f"API usage: {model.total_input_tokens:,} in / {model.total_output_tokens:,} out")
        return None

    elif command == "/image":
        if not args:
            console.print("[#ebcb8b]Usage: /image <file_path> [prompt][/#ebcb8b]")
            console.print("[#666666]Example: /image ~/Desktop/screenshot.png what's wrong with this UI?[/#666666]")
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
                console.print("[#666666]Usage: /mode <ask|auto|trust>[/#666666]")
                console.print("[#666666]  ask   — confirm every tool call[/#666666]")
                console.print("[#666666]  auto  — allow reads, ask for writes[/#666666]")
                console.print("[#666666]  trust — allow all tool calls[/#666666]")
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
            console.print("[#666666]  /team status     — show all workers[/#666666]")
            console.print("[#666666]  /team stop [id]  — stop a worker[/#666666]")
            console.print("[#666666]  /team msg <name> <message> — message a worker[/#666666]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        if sub_cmd == "status":
            workers = team_manager.status()
            if not workers:
                console.print("[#666666]No workers spawned yet.[/#666666]")
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
                    f"[#666666]{w['prompt'][:60]}[/#666666]"
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
            console.print("[#666666]No new messages.[/#666666]")
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
            console.print("[#666666]No tasks yet. Spawn a worker with /team <prompt>[/#666666]")
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
            console.print("[#666666]  /plan show    — show current plan[/#666666]")
            console.print("[#666666]  /plan copy    — copy plan to clipboard[/#666666]")
            console.print("[#666666]  /plan go      — execute the approved plan[/#666666]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        plan_path = os.path.join(os.getcwd(), "plan.md")

        if sub_cmd == "show":
            if not os.path.exists(plan_path):
                console.print("[#666666]No plan.md found. Create one with /plan <prompt>[/#666666]")
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
                console.print("[#666666]No plan.md found. Create one with /plan <prompt>[/#666666]")
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

    # Check skills (skip /plan since we handle it above)
    skill = skills.get(command)
    if skill:
        if skill.requires_args and not args.strip():
            console.print(f"[#ebcb8b]Usage: {command} <prompt>[/#ebcb8b]")
            console.print(f"[#666666]{skill.description}[/#666666]")
            return None
        return skill.get_prompt(args)

    console.print(f"[yellow]Unknown command: {command}. Type /help for available commands.[/yellow]")
    return None


async def run_interactive(config: dict):
    """Run interactive CLI session."""
    console = Console(theme=get_theme())
    ensure_dirs()

    # Initialize skills
    skills = SkillRegistry()
    skills.load_all()

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
    system_prompt = SYSTEM_PROMPT
    if memory_context:
        system_prompt += f"\n\n{memory_context}"

    # Print banner
    print_banner(console, config, mcp_count=len(mcp_tools),
                 skill_count=len(skills.all()))

    # Initialize components
    model = ModelClient(
        endpoint=get(config, "model", "endpoint"),
        model=get(config, "model", "name"),
        temperature=get(config, "model", "temperature", default=0.7),
        max_tokens=get(config, "model", "max_tokens", default=8192),
        api_key=get(config, "model", "api_key", default=""),
        provider=get(config, "model", "provider", default="ollama"),
    )
    context = Context(
        system_prompt=system_prompt,
        max_tokens=get(config, "model", "context_window", default=32768),
    )
    tools = build_tools()

    # Register MCP tools
    for mcp_tool in mcp_tools:
        tools.register(mcp_tool)

    permissions = PermissionManager(
        mode=get(config, "permissions", "mode", default="ask"),
        always_allow=get(config, "permissions", "always_allow", default=[]),
    )
    agent = Agent(model, context, tools, permissions, console)

    # Initialize team system
    task_store = TaskStore()
    team_manager = TeamManager(model, tools, console, task_store)

    # Give the lead agent the ability to spawn workers
    spawn_tool = SpawnWorkerTool()
    spawn_tool.set_team_manager(team_manager)
    tools.register(spawn_tool)

    # Callbacks for the prompt_toolkit footer (always visible below input)
    model_name = get(config, "model", "name", default="unknown")
    provider_name = get(config, "model", "provider", default="")

    def status_callback():
        """Line 1: turns + context %."""
        tokens = context.estimate_tokens()
        turns = context.turn_count
        max_tok = context.max_tokens

        parts = []
        if turns > 0:
            parts.append(("class:bottom-toolbar.info", f"  {turns} turns"))
        else:
            parts.append(("class:bottom-toolbar.info", "  "))

        # Context percentage
        if max_tok > 0 and tokens > 0:
            pct = max(0, 100 - (tokens / max_tok * 100))
            spacer = " " * 40
            parts.append(("class:bottom-toolbar.context", f"{spacer}Context left until auto-compact: {pct:.0f}%"))
        elif tokens > 0:
            parts.append(("class:bottom-toolbar.context", f"    {tokens:,} tokens"))

        return parts

    def mode_callback():
        """Line 2: ⏵⏵ mode on · ctrl+t team."""
        parts = [
            ("class:bottom-toolbar.mode", "  ⏵⏵ "),
            ("class:bottom-toolbar.mode-text", f"{permissions.mode} mode on"),
            ("class:bottom-toolbar.info", "  ·  "),
            ("class:bottom-toolbar.info", "esc to interrupt"),
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
            console.print("\n[#666666]  No workers. Use /team <prompt> to spawn one.[/#666666]")
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
                f"    [#666666]Task: {w['prompt'][:70]}[/#666666]"
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
    )

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

            # Handle slash commands
            if user_input.startswith("/"):
                result = handle_slash_command(
                    user_input, context, console, config, skills, model,
                    permissions=permissions,
                    team_manager=team_manager,
                    task_store=task_store,
                )
                if result is None:
                    continue
                if result == "__IMAGE_SENT__":
                    # Image already added to context — run agent without add_user
                    try:
                        team_monitor.start()
                        await agent.run_without_user_add()
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
                        await agent.run(run_prompt)
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
                else:
                    # Skill returned a prompt — send to agent
                    user_input = result
                    try:
                        team_monitor.start()
                        await agent.run(user_input)
                    except KeyboardInterrupt:
                        console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                    finally:
                        team_monitor.stop()
            elif _is_shell_command(user_input):
                # Direct shell command — run via agent with explicit instruction
                run_prompt = (
                    f"Run this command now with the bash tool:\n\n"
                    f"```\n{user_input}\n```\n\n"
                    f"Just run it and show the output. Do not ask questions."
                )
                try:
                    team_monitor.start()
                    await agent.run(run_prompt)
                except KeyboardInterrupt:
                    console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                except Exception as e:
                    console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                finally:
                    team_monitor.stop()
            else:
                # Run agent
                try:
                    team_monitor.start()
                    await agent.run(user_input)
                except KeyboardInterrupt:
                    console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                except Exception as e:
                    console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                finally:
                    team_monitor.stop()


            # Auto-compact if getting large
            if context.estimate_tokens() > context.max_tokens * 0.8:
                console.print("[#666666]Auto-compacting conversation...[/#666666]")
                context.compact()

    finally:
        await team_manager.stop_all()
        await model.close()
        await mcp_client.disconnect_all()
        console.print("[dim]Session ended.[/dim]")


@click.command()
@click.option("--endpoint", "-e", help="Model API endpoint URL")
@click.option("--model", "-m", "model_name", help="Model name")
@click.option("--provider", "-p", help="Provider: ollama, gemini, openai")
@click.option("--trust", is_flag=True, help="Trust mode (allow all tool calls)")
@click.option("--auto", "auto_mode", is_flag=True, help="Auto mode (allow reads, ask for writes)")
@click.option("--version", "-v", is_flag=True, help="Show version")
@click.argument("prompt", nargs=-1, required=False)
def main(endpoint, model_name, provider, trust, auto_mode, version, prompt):
    """Spark Code — Your local AI coding assistant."""
    if version:
        click.echo(f"Spark Code v{__version__}")
        return

    # Load config with provider selection
    config = load_config(os.getcwd(), provider=provider)

    # CLI overrides
    if endpoint:
        config["model"]["endpoint"] = endpoint
    if model_name:
        config["model"]["name"] = model_name
    if trust:
        config["permissions"]["mode"] = "trust"
    elif auto_mode:
        config["permissions"]["mode"] = "auto"

    # One-shot mode
    if prompt:
        prompt_text = " ".join(prompt)
        asyncio.run(_one_shot(config, prompt_text))
        return

    # Interactive mode
    asyncio.run(run_interactive(config))


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
