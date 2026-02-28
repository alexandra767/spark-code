"""Permission system for tool execution."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


class PermissionManager:
    """Manages tool execution permissions."""

    def __init__(self, mode: str = "ask", always_allow: list[str] | None = None):
        """
        Modes:
          - ask: prompt for every tool call (except always_allow)
          - auto: allow read-only tools, ask for writes
          - trust: allow everything
        """
        self.mode = mode
        self.always_allow = set(always_allow or [])
        self.session_allow = set()  # Tools approved this session
        self.console = Console()

    def check(self, tool_name: str, is_read_only: bool, details: str = "") -> bool:
        """Check if tool execution is allowed. Returns True if allowed."""
        # Trust mode: always allow
        if self.mode == "trust":
            return True

        # Always-allowed tools
        if tool_name in self.always_allow or tool_name in self.session_allow:
            return True

        # Auto mode: allow read-only
        if self.mode == "auto" and is_read_only:
            return True

        # Ask the user
        return self._prompt_user(tool_name, details)

    def _prompt_user(self, tool_name: str, details: str) -> bool:
        """Show permission prompt and get user decision."""
        content = f"[bold]{tool_name}[/bold]"
        if details:
            content += f"\n\n{details}"

        self.console.print(Panel(
            content,
            title="[yellow]Permission Required[/yellow]",
            border_style="yellow",
        ))

        choice = Prompt.ask(
            "[Y]es / [N]o / [A]lways allow this tool",
            choices=["y", "n", "a"],
            default="y",
        )

        if choice == "a":
            self.session_allow.add(tool_name)
            return True
        return choice == "y"
