"""Read file tool."""

import os

from .base import MAX_READ_SIZE, Tool, _is_binary, _validate_path


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
        try:
            path = _validate_path(file_path)
        except ValueError as e:
            return f"Error: {e}"

        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        if os.path.isdir(path):
            return f"Error: {path} is a directory, not a file. Use list_dir instead."

        # Check file size
        try:
            size = os.path.getsize(path)
            if size > MAX_READ_SIZE:
                return f"Error: File too large ({size / 1024 / 1024:.1f} MB). Maximum is {MAX_READ_SIZE / 1024 / 1024:.0f} MB."
        except OSError as e:
            return f"Error checking file size: {e}"

        # Check for binary files
        if _is_binary(path):
            return f"Warning: {path} appears to be a binary file. Showing first 512 bytes as hex is not supported. Use bash tool with 'xxd' or 'file' command instead."

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
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
