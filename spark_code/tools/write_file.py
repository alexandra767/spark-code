"""Write file tool."""

import os

from .base import Tool, _backup_for_undo, _validate_path


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create a new file or overwrite an existing file with the given content."

    def __init__(self, write_roots: list[str] | None = None,
                 cwd: str | None = None):
        # Phase 5 Task 1: extra allowlisted write roots beyond cwd + temp,
        # resolved once at build_tools() time from permissions.write_roots
        # (see config.resolve_write_roots). Empty by default — identical to
        # pre-Phase-5 sandboxing.
        self.write_roots = write_roots or []
        # Phase 5 Task 8: an instance-level default root for write
        # validation, used when no per-call ``cwd`` argument is given. This
        # NARROWS the sandbox (unlike write_roots, which only widens it) —
        # dispatch.py's isolated implementer dispatch sets this to a git
        # worktree path so writes are validated against the worktree instead
        # of the ambient process cwd, without ever touching os.getcwd()
        # (which would race with other concurrently-running agents/workers
        # sharing the same process). None (default) preserves the exact
        # pre-Task-8 behavior: validation falls back to os.getcwd().
        self.cwd = cwd

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(self, file_path: str, content: str,
                      cwd: str | None = None, **kw) -> str:
        try:
            path = _validate_path(file_path, cwd=cwd or self.cwd, for_write=True,
                                  extra_roots=self.write_roots)
        except ValueError as e:
            return f"Error: {e}"

        # Create parent directories if needed
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        # Backup for /undo
        _backup_for_undo(path)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Successfully wrote {lines} lines to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
