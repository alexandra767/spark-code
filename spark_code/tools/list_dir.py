"""List directory tool."""

import os

from .base import Tool


class ListDirTool(Tool):
    name = "list_dir"
    description = "List the contents of a directory, showing files and subdirectories."
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                },
            },
        }

    async def execute(self, path: str = "", **kw) -> str:
        dir_path = os.path.expanduser(path) if path else os.getcwd()

        if not os.path.exists(dir_path):
            return f"Error: Directory not found: {dir_path}"
        if not os.path.isdir(dir_path):
            return f"Error: {dir_path} is a file, not a directory."

        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            return f"Error: Permission denied: {dir_path}"

        if not entries:
            return f"Directory is empty: {dir_path}"

        result = []
        dirs = []
        files = []

        for entry in entries:
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full):
                dirs.append(f"  {entry}/")
            else:
                size = os.path.getsize(full)
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                files.append(f"  {entry}  ({size_str})")

        result = [f"Directory: {dir_path}"]
        if dirs:
            result.append("\nDirectories:")
            result.extend(dirs)
        if files:
            result.append("\nFiles:")
            result.extend(files)

        return "\n".join(result)
