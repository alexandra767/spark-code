"""Grep search tool — uses ripgrep if available, falls back to Python."""

import asyncio
import os
import re
import shutil
import signal

from .base import Tool


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


class GrepTool(Tool):
    name = "grep"
    description = "Search file contents for a regex pattern. Uses ripgrep for speed. Returns matching lines with file paths and line numbers."
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (default: current directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob to filter files (e.g., '*.py', '*.{ts,tsx}')",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive search (default: false)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, pattern: str, path: str = "", glob: str = "",
                      case_insensitive: bool = False, **kw) -> str:
        search_path = path or os.getcwd()
        search_path = os.path.expanduser(search_path)

        # Try ripgrep first
        if shutil.which("rg"):
            return await self._rg_search(pattern, search_path, glob, case_insensitive)
        else:
            # Sync os.walk over a big tree would block the event loop.
            return await asyncio.to_thread(
                self._python_search, pattern, search_path, glob, case_insensitive
            )

    async def _rg_search(self, pattern: str, path: str, file_glob: str,
                         case_insensitive: bool) -> str:
        cmd = ["rg", "--no-heading", "--line-number", "--max-count", "50",
               # Cap match line length so a minified/one-line JSON doesn't dump
               # megabytes; show a short preview of the omitted tail instead.
               "--max-columns", "500", "--max-columns-preview"]
        if case_insensitive:
            cmd.append("-i")
        if file_glob:
            cmd.extend(["--glob", file_glob])
        # -e guards a pattern that starts with '-' (otherwise parsed as a flag);
        # '--' ends option parsing so a path can't be misread as a flag either.
        cmd.extend(["-e", pattern, "--", path])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=30
                )
            except asyncio.TimeoutError:
                await _kill_process_group(process)
                return "Error: Search timed out"
        except Exception as e:
            return f"Error: {e}"

        rc = process.returncode
        output = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()

        # ripgrep exit codes: 0 = matches, 1 = no matches, 2 = error
        # (bad regex, unreadable path, etc.). Don't disguise an error as
        # "no matches".
        if rc == 2 or (rc not in (0, 1) and err):
            return f"Error: ripgrep failed: {err or 'unknown error'}"
        if not output:
            msg = f"No matches found for: {pattern}"
            if err:
                msg += f"\n(note: {err})"
            return msg
        return output

    def _python_search(self, pattern: str, path: str, file_glob: str,
                       case_insensitive: bool) -> str:
        """Fallback Python-based search."""
        import fnmatch
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex: {e}"

        results = []
        skipped_files = []
        max_results = 50

        for root, dirs, files in os.walk(path):
            # Skip hidden dirs and common ignore dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                       {"node_modules", "__pycache__", "venv", ".git", "dist", "build"}]
            for fname in files:
                if file_glob and not fnmatch.fnmatch(fname, file_glob):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"{fpath}:{i}:{line.rstrip()}")
                                if len(results) >= max_results:
                                    output = "\n".join(results) + f"\n\n... (truncated at {max_results} matches)"
                                    if skipped_files:
                                        output += f"\n({len(skipped_files)} file(s) skipped due to errors)"
                                    return output
                except (OSError, UnicodeDecodeError):
                    skipped_files.append(fpath)
                    continue

        if not results:
            msg = f"No matches found for: {pattern}"
            if skipped_files:
                msg += f"\n({len(skipped_files)} file(s) skipped due to errors)"
            return msg
        output = "\n".join(results)
        if skipped_files:
            output += f"\n\n({len(skipped_files)} file(s) skipped due to errors)"
        return output
