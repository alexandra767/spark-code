"""Hooks system — run commands before/after tool calls.

Configuration in ~/.spark/config.yaml or .spark/config.yaml:

hooks:
  after_write_file:
    - pattern: "*.py"
      command: "ruff check --fix {path}"
    - pattern: "*.js"
      command: "eslint --fix {path}"
  after_edit_file:
    - pattern: "*.py"
      command: "ruff check --fix {path}"
  before_bash:
    - command: "echo 'Running: {command}'"
"""

import asyncio
import fnmatch
import logging
import os

logger = logging.getLogger(__name__)


class Hook:
    """A single hook definition."""

    def __init__(self, command: str, pattern: str = "*", timeout: int = 30):
        self.command = command
        self.pattern = pattern
        self.timeout = timeout

    def matches(self, path: str) -> bool:
        """Check if hook pattern matches the given path."""
        if self.pattern == "*":
            return True
        basename = os.path.basename(path)
        return fnmatch.fnmatch(basename, self.pattern)

    async def run(self, context: dict[str, str]) -> tuple[bool, str]:
        """Execute the hook command with context substitution.

        context keys: path, command, old_string, new_string, pattern, etc.
        Returns (success, output).
        """
        cmd = self.command
        for key, value in context.items():
            cmd = cmd.replace(f"{{{key}}}", str(value))

        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            return process.returncode == 0, output
        except asyncio.TimeoutError:
            return False, f"Hook timed out after {self.timeout}s"
        except Exception as e:
            return False, f"Hook error: {e}"


class HookManager:
    """Manages pre/post hooks for tool calls."""

    def __init__(self, config: dict | None = None):
        self._hooks: dict[str, list[Hook]] = {}
        if config:
            self.load(config)

    def load(self, config: dict):
        """Load hooks from config dict."""
        hooks_conf = config.get("hooks", {})
        if not hooks_conf:
            return

        for event_name, hook_list in hooks_conf.items():
            if not isinstance(hook_list, list):
                continue
            self._hooks[event_name] = []
            for h in hook_list:
                if isinstance(h, dict) and "command" in h:
                    self._hooks[event_name].append(Hook(
                        command=h["command"],
                        pattern=h.get("pattern", "*"),
                        timeout=h.get("timeout", 30),
                    ))

    def has_hooks(self, event: str) -> bool:
        return bool(self._hooks.get(event))

    async def run_hooks(self, event: str, context: dict[str, str],
                        console=None) -> list[tuple[bool, str]]:
        """Run all hooks for an event. Returns list of (success, output)."""
        hooks = self._hooks.get(event, [])
        if not hooks:
            return []

        results = []
        path = context.get("path", context.get("file_path", ""))

        for hook in hooks:
            if path and not hook.matches(path):
                continue
            success, output = await hook.run(context)
            results.append((success, output))

            if console and output:
                from rich.text import Text
                style = "#a3be8c" if success else "#ebcb8b"
                console.print(Text(f"  hook: {output[:120]}", style=style))

        return results

    def get_events(self) -> list[str]:
        """List all configured hook events."""
        return list(self._hooks.keys())

    @property
    def count(self) -> int:
        return sum(len(hooks) for hooks in self._hooks.values())
