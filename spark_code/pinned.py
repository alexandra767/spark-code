"""Pinned files — keep key files always in context."""

import os


class PinnedFiles:
    """Manages files that stay in context across turns."""

    def __init__(self):
        self._files: dict[str, str] = {}  # path -> content

    def pin(self, file_path: str) -> tuple[bool, str]:
        """Pin a file. Returns (success, message)."""
        path = os.path.expanduser(file_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.isfile(path):
            return False, f"File not found: {file_path}"
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 50000:
                return False, f"File too large to pin ({len(content)} chars)"
            self._files[path] = content
            return True, f"Pinned: {file_path}"
        except OSError as e:
            return False, f"Error reading {file_path}: {e}"

    def unpin(self, file_path: str) -> tuple[bool, str]:
        """Unpin a file."""
        path = os.path.expanduser(file_path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if path in self._files:
            del self._files[path]
            return True, f"Unpinned: {file_path}"
        return False, f"Not pinned: {file_path}"

    def refresh(self):
        """Re-read all pinned files (in case they changed)."""
        for path in list(self._files):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    self._files[path] = f.read()
            except OSError:
                del self._files[path]

    def get_context(self) -> str:
        """Get pinned files as context string for system prompt."""
        if not self._files:
            return ""
        parts = []
        home = os.path.expanduser("~")
        for path, content in self._files.items():
            display = "~" + path[len(home):] if path.startswith(home) else path
            parts.append(f"### {display}\n```\n{content}\n```")
        return "# Pinned Files (always in context)\n\n" + "\n\n".join(parts)

    def list(self) -> list[str]:
        return list(self._files.keys())

    @property
    def count(self) -> int:
        return len(self._files)
