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
            },
            "required": ["command"],
        }

    async def execute(self, command: str, timeout: int = 120, **kw) -> str:
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
