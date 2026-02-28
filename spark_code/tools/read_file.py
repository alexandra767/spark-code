"""Read file tool."""

import os
from .base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file. Returns the file with line numbers."
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based). Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Optional.",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, file_path: str, offset: int = 1, limit: int = 0, **kw) -> str:
        path = os.path.expanduser(file_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        if os.path.isdir(path):
            return f"Error: {path} is a directory, not a file. Use list_dir instead."

        try:
            with open(path, "r", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading file: {e}"

        # Apply offset (1-based)
        start = max(0, offset - 1)
        if limit > 0:
            end = start + limit
        else:
            end = len(lines)

        selected = lines[start:end]

        # Format with line numbers
        result = []
        for i, line in enumerate(selected, start=start + 1):
            result.append(f"{i:>6}\t{line.rstrip()}")

        if not result:
            return f"File is empty: {path}"

        total = len(lines)
        header = f"File: {path} ({total} lines)"
        if start > 0 or end < total:
            header += f" [showing lines {start + 1}-{min(end, total)}]"

        return header + "\n" + "\n".join(result)
