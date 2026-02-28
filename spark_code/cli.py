"""Spark Code CLI — entry point."""

import asyncio
import os
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
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
from .skills.base import SkillRegistry
from .mcp.client import MCPClient
from .ui.input import create_session
from .ui.theme import get_theme


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


def print_banner(console: Console, config: dict, mcp_count: int = 0,
                 skill_count: int = 0):
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
    if mcp_count > 0:
        banner.append(f"\n   MCP: {mcp_count} server tool(s) loaded", style="dim green")
    if skill_count > 0:
        banner.append(f"\n   Skills: {skill_count} available", style="dim green")
    banner.append("\n   Type /help for commands", style="dim")

    console.print(Panel(banner, border_style="yellow", padding=(0, 1)))
    console.print()


def handle_slash_command(cmd: str, context: Context, console: Console,
                         config: dict, skills: SkillRegistry,
                         model: ModelClient) -> str | None:
    """Handle slash commands.
    Returns None if handled (no agent needed), or a prompt string for the agent.
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

    elif command in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    # Check skills
    skill = skills.get(command)
    if skill:
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
        max_tokens=get(config, "model", "max_tokens", default=4096),
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

    # Input session with history and autocomplete
    session = create_session(skill_names=skills.names())

    try:
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: session.prompt("\n> "),
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
                    user_input, context, console, config, skills, model
                )
                if result is None:
                    continue
                # Skill returned a prompt — send to agent
                user_input = result

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
        await mcp_client.disconnect_all()
        console.print("[dim]Session ended.[/dim]")


@click.command()
@click.option("--endpoint", "-e", help="Model API endpoint URL")
@click.option("--model", "-m", "model_name", help="Model name")
@click.option("--trust", is_flag=True, help="Trust mode (allow all tool calls)")
@click.option("--auto", "auto_mode", is_flag=True, help="Auto mode (allow reads, ask for writes)")
@click.option("--version", "-v", is_flag=True, help="Show version")
@click.argument("prompt", nargs=-1, required=False)
def main(endpoint, model_name, trust, auto_mode, version, prompt):
    """Spark Code — Your local AI coding assistant."""
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
