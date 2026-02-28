"""Bash execution tool."""

import asyncio
import os
from .base import Tool


class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command and return its output (stdout + stderr). Use for git, npm, pip, tests, and other system commands."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)",
                },
                "background": {
                    "type": "boolean",
                    "description": "Launch detached in background (for GUI apps, servers). Default: false, auto-detected for GUI apps.",
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str, timeout: int = 120,
                      background: bool = False, **kw) -> str:
        # Background mode — launch detached (for GUI apps, servers, etc.)
        if background or self._is_gui_command(command):
            return await self._run_background(command)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            if output:
                output += "\n"
            output += stderr.decode("utf-8", errors="replace")

        exit_code = process.returncode
        if exit_code != 0:
            output += f"\n\nExit code: {exit_code}"

        return output.strip() if output.strip() else f"Command completed (exit code {exit_code})"

    def _is_gui_command(self, command: str) -> bool:
        """Detect commands that launch GUI apps and should run detached."""
        cmd = command.strip().split()[0] if command.strip() else ""
        # Python scripts that use pygame/tkinter/etc.
        if cmd in ("python", "python3"):
            gui_keywords = ("pygame", "tkinter", "kivy", "PyQt", "PySide",
                            "wxPython", "turtle", "arcade")
            # Check if the script file imports GUI libraries
            parts = command.strip().split()
            if len(parts) >= 2 and parts[1].endswith(".py"):
                try:
                    with open(os.path.join(os.getcwd(), parts[1])) as f:
                        head = f.read(2000)
                    if any(kw in head for kw in gui_keywords):
                        return True
                except (OSError, FileNotFoundError):
                    pass
        # macOS open command, electron, etc.
        if cmd in ("open", "electron", "flutter"):
            return True
        return False

    async def _run_background(self, command: str) -> str:
        """Launch a process detached — for GUI apps and servers."""
        import subprocess
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=os.getcwd(),
                stdin=None,
                stdout=None,
                stderr=None,
                start_new_session=True,
            )
            return (
                f"Launched in background (PID {process.pid}). "
                f"The app window should appear on your screen."
            )
        except Exception as e:
            return f"Error launching: {e}"
