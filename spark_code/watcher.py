"""File watcher — auto-run commands when files change.

Usage: /watch pytest
       /watch npm test
       /watch off
"""

import asyncio
import os
import time

from rich.console import Console
from rich.text import Text

# Nord palette
_C_TOOL = "#88c0d0"
_C_GREEN = "#a3be8c"
_C_RED = "#bf616a"
_C_DIM = "#4c566a"
_C_YELLOW = "#ebcb8b"

# File extensions to watch
_WATCH_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".kt",
    ".swift", ".rb", ".c", ".cpp", ".h", ".hpp", ".css", ".html",
    ".yaml", ".yml", ".toml", ".json", ".sql", ".sh",
}

# Directories to ignore
_IGNORE_DIRS = {
    "__pycache__", "node_modules", ".git", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".next", ".nuxt", "target", ".cargo",
}


class FileWatcher:
    """Watches for file changes and runs a command."""

    def __init__(self, command: str, console: Console, directory: str = "."):
        self.command = command
        self.console = console
        self.directory = os.path.abspath(directory)
        self._task: asyncio.Task | None = None
        self._running = False
        self._run_count = 0

    def _scan(self) -> dict[str, float]:
        """Scan watched files and return path -> mtime."""
        snapshot = {}
        for root, dirs, files in os.walk(self.directory):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext in _WATCH_EXTENSIONS:
                    path = os.path.join(root, f)
                    try:
                        snapshot[path] = os.path.getmtime(path)
                    except OSError:
                        pass
        return snapshot

    async def start(self):
        """Start watching."""
        if self._running:
            return
        self._running = True
        self.console.print(Text(
            f"  Watching for changes... (command: {self.command})",
            style=_C_TOOL))
        self.console.print(Text(
            "  Use /watch off to stop", style=_C_DIM))
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self):
        """Stop watching."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.console.print(Text("  Watcher stopped", style=_C_DIM))

    @property
    def is_running(self) -> bool:
        return self._running

    async def _watch_loop(self):
        """Main watch loop — poll for changes every 2 seconds."""
        prev_snapshot = self._scan()
        try:
            while self._running:
                await asyncio.sleep(2)
                if not self._running:
                    break

                current = self._scan()
                changed = self._diff(prev_snapshot, current)
                if changed:
                    self._run_count += 1
                    # Show changed files
                    for path in changed[:5]:
                        rel = os.path.relpath(path, self.directory)
                        self.console.print(Text(
                            f"  Changed: {rel}", style=_C_YELLOW))
                    if len(changed) > 5:
                        self.console.print(Text(
                            f"  ... and {len(changed) - 5} more", style=_C_DIM))

                    # Run the command
                    self.console.print(Text(
                        f"  Running: {self.command} (#{self._run_count})",
                        style=_C_TOOL))
                    await self._run_command()

                prev_snapshot = current

        except asyncio.CancelledError:
            pass

    def _diff(self, old: dict, new: dict) -> list[str]:
        """Find changed/added/removed files."""
        changed = []
        all_paths = set(old) | set(new)
        for path in all_paths:
            if path not in old:
                changed.append(path)  # new file
            elif path not in new:
                changed.append(path)  # deleted
            elif old[path] != new[path]:
                changed.append(path)  # modified
        return changed

    async def _run_command(self):
        """Execute the watch command."""
        try:
            process = await asyncio.create_subprocess_shell(
                self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.directory,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=120
            )
            output = stdout.decode("utf-8", errors="replace").strip()

            if process.returncode == 0:
                # Show last few lines on success
                lines = output.split("\n")
                for line in lines[-5:]:
                    self.console.print(Text(f"  {line}", style=_C_GREEN))
            else:
                # Show more on failure
                lines = output.split("\n")
                for line in lines[-10:]:
                    self.console.print(Text(f"  {line}", style=_C_RED))
                self.console.print(Text(
                    f"  Exit code: {process.returncode}", style=_C_RED))

        except asyncio.TimeoutError:
            self.console.print(Text("  Command timed out", style=_C_RED))
        except Exception as e:
            self.console.print(Text(f"  Error: {e}", style=_C_RED))
