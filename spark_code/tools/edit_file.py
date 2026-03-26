"""Edit file tool — find and replace."""

import difflib
import os

from .base import Tool, _backup_for_undo, _validate_path


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
        try:
            path = _validate_path(file_path)
        except ValueError as e:
            return f"Error: {e}"

        if not os.path.exists(path):
            return f"Error: File not found: {path}"

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

        if old_string not in content:
            hint = self._find_closest_match(content, old_string)
            base_msg = f"Error: old_string not found in {path}."
            if hint:
                return f"{base_msg}\n\n{hint}\n\nHint: Check whitespace, indentation, and exact string content."
            return f"{base_msg} Make sure it matches exactly (including whitespace). Try reading the file first."

        count = content.count(old_string)
        if count > 1 and not replace_all:
            return f"Error: old_string found {count} times in {path}. Set replace_all=true to replace all, or provide more context to make it unique."

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements = 1

        # Backup for /undo
        _backup_for_undo(path)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"Successfully replaced {replacements} occurrence(s) in {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def _find_closest_match(content: str, old_string: str) -> str:
        """Find the most similar block in the file to old_string."""
        old_lines = old_string.splitlines()
        file_lines = content.splitlines()
        window_size = len(old_lines)

        if window_size == 0 or len(file_lines) == 0:
            return ""

        best_ratio = 0.0
        best_start = 0

        for i in range(max(1, len(file_lines) - window_size + 1)):
            window = file_lines[i:i + window_size]
            window_text = "\n".join(window)
            ratio = difflib.SequenceMatcher(None, old_string, window_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio < 0.4:
            return ""

        best_window = file_lines[best_start:best_start + window_size]
        match_text = "\n".join(f"    {line}" for line in best_window)
        start_line = best_start + 1
        end_line = best_start + window_size
        return f"Closest match (lines {start_line}-{end_line}, {best_ratio:.0%} similar):\n{match_text}"
