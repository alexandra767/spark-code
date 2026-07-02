"""Bash execution tool."""

import asyncio
import os
import re
import signal

from .base import Tool

# Raise the asyncio StreamReader buffer limit well above the 64 KB default so a
# single very long output line doesn't blow up readline().
_STREAM_LIMIT = 10 * 1024 * 1024  # 10 MB

# Cap the returned output. Truncate the MIDDLE (keep head + tail) so the exit
# code / tail of a failing command is never lost.
_MAX_OUTPUT_CHARS = 100_000


def _truncate_middle(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate the middle of *text*, preserving head and tail."""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n\n... [truncated {omitted} chars in the middle] ...\n\n"
        + text[-tail:]
    )


SIDE_EFFECT_PATTERNS = [
    (r'\bpip\s+install\b', "Installs packages into the active Python environment"),
    (r'\bnpm\s+install\b', "Modifies node_modules and package-lock.json"),
    (r'\brm\s', "Deletes files or directories"),
    (r'\bgit\s+(push|reset|checkout)\b', "Modifies git state"),
    (r'\bbrew\s+install\b', "Installs system-level packages"),
    (r'\bcurl\b.*\|\s*(bash|sh)\b', "Pipes remote script to shell"),
    (r'\bsudo\b', "Runs with elevated privileges"),
    (r'\bdocker\s+(rm|rmi|stop|kill)\b', "Modifies Docker containers/images"),
]


def detect_side_effects(command: str) -> list[str]:
    """Check a command for known side-effect patterns. Returns list of warnings."""
    warnings = []
    for pattern, description in SIDE_EFFECT_PATTERNS:
        if re.search(pattern, command):
            warnings.append(description)
    return warnings


class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command and return its output (stdout + stderr). Use for git, npm, pip, tests, and other system commands."

    @property
    def supports_streaming(self) -> bool:
        return True

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
                # Own process group so a timeout can kill the whole tree, not
                # just /bin/sh (which would orphan children).
                start_new_session=True,
                limit=_STREAM_LIMIT,
            )
        except Exception as e:
            return f"Error executing command: {e}"

        stdout_buf = bytearray()
        stderr_buf = bytearray()

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    self._drain(process.stdout, stdout_buf),
                    self._drain(process.stderr, stderr_buf),
                    process.wait(),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Kill the whole process group and return whatever we captured so far.
            await self._kill_process_group(process)
            return self._finalize_timeout(stdout_buf, stderr_buf, timeout)
        except Exception as e:
            await self._kill_process_group(process)
            return f"Error executing command: {e}"

        output = stdout_buf.decode("utf-8", errors="replace")
        stderr_text = stderr_buf.decode("utf-8", errors="replace")
        if stderr_text:
            if output:
                output += "\n"
            output += stderr_text

        return self._finalize(output, process.returncode)

    @staticmethod
    async def _drain(stream, buf: bytearray) -> None:
        """Read a stream into *buf* in chunks.

        Uses read(n) (not readline) so a single multi-megabyte line never trips
        the StreamReader limit.
        """
        if stream is None:
            return
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            buf.extend(chunk)

    @staticmethod
    async def _kill_process_group(process) -> None:
        """SIGTERM then SIGKILL the process's whole group; reap it."""
        try:
            pgid = os.getpgid(process.pid)
        except (ProcessLookupError, OSError):
            pgid = None

        def _send(sig):
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)
                else:
                    process.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass

        _send(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
            return
        except (asyncio.TimeoutError, ProcessLookupError):
            pass
        _send(signal.SIGKILL)
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

    def _finalize(self, output: str, exit_code) -> str:
        """Truncate (middle) then append the exit code so it's never cut off."""
        output = _truncate_middle(output)
        if exit_code is not None and exit_code != 0:
            output += f"\n\nExit code: {exit_code}"
        stripped = output.strip()
        return stripped if stripped else f"Command completed (exit code {exit_code})"

    def _finalize_timeout(self, stdout_buf: bytearray, stderr_buf: bytearray,
                          timeout: int) -> str:
        """Return partial output plus a timeout marker (never discard it)."""
        output = stdout_buf.decode("utf-8", errors="replace")
        stderr_text = stderr_buf.decode("utf-8", errors="replace")
        if stderr_text:
            if output:
                output += "\n"
            output += stderr_text
        output = _truncate_middle(output)
        marker = (f"[timed out after {timeout}s] "
                  f"Command exceeded the time limit and was killed.")
        stripped = output.strip()
        return (stripped + "\n\n" + marker) if stripped else marker

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

    async def execute_streaming(self, command: str, timeout: int = 120,
                                callback=None, background: bool = False,
                                **kw) -> str:
        """Execute command with line-by-line streaming output.

        callback(line: str) is called for each output line as it arrives.
        Still returns the full output for context.
        """
        # Honor background/GUI commands here too — streaming can't stream a
        # detached server, and blocking-until-timeout on `npm start` is wrong.
        if background or self._is_gui_command(command):
            return await self._run_background(command)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd(),
                # Own process group so a timeout kills the whole tree.
                start_new_session=True,
                # Raise the readline() buffer limit so long lines don't blow up.
                limit=_STREAM_LIMIT,
            )
        except Exception as e:
            return f"Error executing command: {e}"

        lines: list[str] = []

        async def read_lines():
            while True:
                try:
                    line = await process.stdout.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    # A single line exceeded even the raised limit. Fall back to
                    # a bounded raw read so we don't lose output or orphan the
                    # process (the old code let this except nuke everything).
                    try:
                        line = await process.stdout.read(_STREAM_LIMIT)
                    except Exception:
                        break
                    if not line:
                        break
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                lines.append(decoded)
                if callback:
                    try:
                        callback(decoded)
                    except Exception:
                        pass  # Don't crash on callback failure

        try:
            await asyncio.wait_for(read_lines(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._kill_process_group(process)
            partial = _truncate_middle("\n".join(lines))
            marker = (f"[timed out after {timeout}s] "
                      f"Command exceeded the time limit and was killed.")
            return (partial.strip() + "\n\n" + marker) if partial.strip() else marker
        except Exception as e:
            await self._kill_process_group(process)
            partial = _truncate_middle("\n".join(lines)).strip()
            err = f"Error executing command: {e}"
            return (partial + "\n\n" + err) if partial else err

        await process.wait()
        return self._finalize("\n".join(lines), process.returncode)

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
