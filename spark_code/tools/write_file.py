"""Write file tool."""

import os
from .base import Tool


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create a new file or overwrite an existing file with the given content."

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

    async def execute(self, file_path: str, content: str, **kw) -> str:
        path = os.path.expanduser(file_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        # Create parent directories if needed
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        try:
            with open(path, "w") as f:
                f.write(content)
            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Successfully wrote {lines} lines to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
