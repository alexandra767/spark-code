"""Edit file tool — find and replace."""

import os
from .base import Tool


class EditFileTool(Tool):
    name = "edit_file"
    description = "Edit a file by replacing an exact string match with new content. The old_string must match exactly (including whitespace/indentation)."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def execute(self, file_path: str, old_string: str, new_string: str,
                      replace_all: bool = False, **kw) -> str:
        path = os.path.expanduser(file_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        if not os.path.exists(path):
            return f"Error: File not found: {path}"

        try:
            with open(path, "r") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

        if old_string not in content:
            return f"Error: old_string not found in {path}. Make sure it matches exactly (including whitespace)."

        count = content.count(old_string)
        if count > 1 and not replace_all:
            return f"Error: old_string found {count} times in {path}. Set replace_all=true to replace all, or provide more context to make it unique."

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        try:
            with open(path, "w") as f:
                f.write(new_content)
            return f"Successfully replaced {replacements} occurrence(s) in {path}"
        except Exception as e:
            return f"Error writing file: {e}"
