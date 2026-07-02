"""Glob file search tool."""

import asyncio
import glob as globlib
import os

from .base import Tool


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern (e.g., '**/*.py', 'src/**/*.ts'). Returns matching file paths."
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files (e.g., '**/*.py')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, pattern: str, path: str = "", **kw) -> str:
        search_dir = path or os.getcwd()
        search_dir = os.path.expanduser(search_dir)

        full_pattern = os.path.join(search_dir, pattern)

        # glob.glob is sync-blocking on large trees — offload it so it doesn't
        # stall the event loop.
        def _scan():
            matches = sorted(globlib.glob(full_pattern, recursive=True))
            return [m for m in matches if os.path.isfile(m)]

        files = await asyncio.to_thread(_scan)

        if not files:
            return f"No files found matching: {pattern}"

        # Limit output
        total = len(files)
        shown = files[:100]
        result = "\n".join(shown)
        if total > 100:
            result += f"\n\n... and {total - 100} more files"
        else:
            result += f"\n\n{total} file(s) found"

        return result
