"""Grep search tool — uses ripgrep if available, falls back to Python."""

import asyncio
import os
import re
import shutil
from .base import Tool


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
            return self._python_search(pattern, search_path, glob, case_insensitive)

    async def _rg_search(self, pattern: str, path: str, file_glob: str,
                         case_insensitive: bool) -> str:
        cmd = ["rg", "--no-heading", "--line-number", "--max-count", "50"]
        if case_insensitive:
            cmd.append("-i")
        if file_glob:
            cmd.extend(["--glob", file_glob])
        cmd.extend([pattern, path])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace").strip()
            if not output:
                return f"No matches found for: {pattern}"
            return output
        except asyncio.TimeoutError:
            return "Error: Search timed out"
        except Exception as e:
            return f"Error: {e}"

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
                    with open(fpath, "r", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"{fpath}:{i}:{line.rstrip()}")
                                if len(results) >= max_results:
                                    return "\n".join(results) + f"\n\n... (truncated at {max_results} matches)"
                except (OSError, UnicodeDecodeError):
                    continue

        if not results:
            return f"No matches found for: {pattern}"
        return "\n".join(results)
