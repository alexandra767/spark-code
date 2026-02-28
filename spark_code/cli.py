"""Spark Code CLI — entry point."""

import asyncio
import os
import sys

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from . import __version__, __app_name__
from .config import load_config, get, ensure_dirs
from .model import ModelClient
from .context import Context
from .agent import Agent
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


def print_banner(console: Console, config: dict):
    """Print startup banner."""
    model_name = get(config, "model", "name", default="unknown")
    endpoint = get(config, "model", "endpoint", default="unknown")

    banner = Text()
    banner.append("⚡ ", style="bold yellow")
    banner.append(f"{__app_name__} ", style="bold white")
    banner.append(f"v{__version__}", style="dim")
    banner.append(f"\n   Model: {model_name}", style="cyan")
    banner.append(f"\n   Endpoint: {endpoint}", style="dim")
    banner.append(f"\n   Dir: {os.getcwd()}", style="dim")
    banner.append("\n   Type /help for commands", style="dim")

    console.print(Panel(banner, border_style="yellow", padding=(0, 1)))
    console.print()


def handle_slash_command(cmd: str, context: Context, console: Console,
                         config: dict) -> bool:
    """Handle slash commands. Returns True if handled."""
    cmd = cmd.strip().lower()

    if cmd == "/help":
        console.print(Markdown("""## Commands
- `/help` — Show this help
- `/clear` — Clear conversation
- `/compact` — Summarize conversation to save context
- `/config` — Show current config
- `/model` — Show model info
- `/tokens` — Show token usage
- `/commit` — Generate git commit
- `/review` — Review code changes
- `/test` — Run tests
- `/quit` or `/exit` — Exit
"""))
        return True

    elif cmd == "/clear":
        context.clear()
        console.print("[green]Conversation cleared.[/green]")
        return True

    elif cmd == "/compact":
        before = context.estimate_tokens()
        context.compact()
        after = context.estimate_tokens()
        console.print(f"[green]Compacted: ~{before:,} → ~{after:,} tokens[/green]")
        return True

    elif cmd == "/config":
        import yaml
        console.print(Markdown(f"```yaml\n{yaml.dump(config, default_flow_style=False)}```"))
        return True

    elif cmd == "/model":
        console.print(f"Model: {get(config, 'model', 'name')}")
        console.print(f"Endpoint: {get(config, 'model', 'endpoint')}")
        console.print(f"Temperature: {get(config, 'model', 'temperature')}")
        return True

    elif cmd == "/tokens":
        tokens = context.estimate_tokens()
        console.print(f"Estimated tokens: ~{tokens:,}")
        console.print(f"Turns: {context.turn_count}")
        return True

    elif cmd in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    # Skill-based commands
    elif cmd == "/commit":
        # Inject commit skill prompt
        return False  # Let agent handle it with modified input

    elif cmd == "/review":
        return False

    elif cmd == "/test":
        return False

    return False


SKILL_PROMPTS = {
    "/commit": "Look at the current git diff (staged and unstaged changes), then create a concise, meaningful commit message. Show me the message before committing.",
    "/review": "Review the recent code changes (git diff) for bugs, security issues, and improvements. Be specific about what to fix.",
    "/test": "Find and run the project's test suite. Report results and suggest fixes for any failures.",
    "/explain": "Explain how the current project/codebase is structured. Read key files to understand the architecture.",
}


async def run_interactive(config: dict):
    """Run interactive CLI session."""
    console = Console()
    ensure_dirs()
    print_banner(console, config)

    # Initialize components
    model = ModelClient(
        endpoint=get(config, "model", "endpoint"),
        model=get(config, "model", "name"),
        temperature=get(config, "model", "temperature", default=0.7),
        max_tokens=get(config, "model", "max_tokens", default=4096),
    )
    context = Context(max_tokens=get(config, "model", "context_window", default=32768))
    tools = build_tools()
    permissions = PermissionManager(
        mode=get(config, "permissions", "mode", default="ask"),
        always_allow=get(config, "permissions", "always_allow", default=[]),
    )
    agent = Agent(model, context, tools, permissions, console)

    # Input session with history
    history_file = os.path.expanduser("~/.spark/history")
    session = PromptSession(history=FileHistory(history_file))

    try:
        while True:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt(
                        "\n> ",
                        multiline=False,
                    ),
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
                if handle_slash_command(user_input, context, console, config):
                    continue
                # Check for skill prompts
                cmd = user_input.split()[0].lower()
                if cmd in SKILL_PROMPTS:
                    user_input = SKILL_PROMPTS[cmd]

            # Run agent
            try:
                await agent.run(user_input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

            # Auto-compact if getting large
            if context.estimate_tokens() > context.max_tokens * 0.8:
                console.print("[dim]Auto-compacting conversation...[/dim]")
                context.compact()

    finally:
        await model.close()
        console.print("[dim]Session ended.[/dim]")


@click.command()
@click.option("--endpoint", "-e", help="Model API endpoint URL")
@click.option("--model", "-m", "model_name", help="Model name")
@click.option("--trust", is_flag=True, help="Trust mode (allow all tool calls)")
@click.option("--version", "-v", is_flag=True, help="Show version")
@click.argument("prompt", nargs=-1, required=False)
def main(endpoint, model_name, trust, version, prompt):
    """⚡ Spark Code — Your local AI coding assistant."""
    if version:
        click.echo(f"Spark Code v{__version__}")
        return

    # Load config
    config = load_config(os.getcwd())

    # CLI overrides
    if endpoint:
        config["model"]["endpoint"] = endpoint
    if model_name:
        config["model"]["name"] = model_name
    if trust:
        config["permissions"]["mode"] = "trust"

    # One-shot mode
    if prompt:
        prompt_text = " ".join(prompt)
        asyncio.run(_one_shot(config, prompt_text))
        return

    # Interactive mode
    asyncio.run(run_interactive(config))


async def _one_shot(config: dict, prompt: str):
    """Run a single prompt and exit."""
    console = Console()
    model = ModelClient(
        endpoint=get(config, "model", "endpoint"),
        model=get(config, "model", "name"),
        temperature=get(config, "model", "temperature", default=0.7),
        max_tokens=get(config, "model", "max_tokens", default=4096),
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
