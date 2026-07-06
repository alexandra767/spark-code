"""Pinned files — keep key files always in context."""

import os
import re


class PinnedFiles:
    """Manages files that stay in context across turns."""

    # Delimiters marking the single pinned-files region inside the system
    # prompt. The region is rewritten (never appended) each turn so edited or
    # unpinned files don't accumulate stale duplicate blocks.
    PIN_START = "<!--spark:pinned-->"
    PIN_END = "<!--/spark:pinned-->"

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
        """Get pinned files as a delimited context block for the system prompt.

        The block is wrapped in ``PIN_START``/``PIN_END`` so it can be located
        and replaced as a single region (see ``apply_to_prompt``). Returns an
        empty string when nothing is pinned.
        """
        if not self._files:
            return ""
        parts = []
        home = os.path.expanduser("~")
        for path, content in self._files.items():
            display = "~" + path[len(home):] if path.startswith(home) else path
            parts.append(f"### {display}\n```\n{content}\n```")
        body = "# Pinned Files (always in context)\n\n" + "\n\n".join(parts)
        return f"{self.PIN_START}\n{body}\n{self.PIN_END}"

    def apply_to_prompt(self, base_prompt: str) -> str:
        """Return ``base_prompt`` with the pinned region rewritten to the current
        pinned content (or removed entirely when nothing is pinned).

        Idempotent: any existing pinned region is stripped first, so calling this
        every turn never accumulates duplicate or stale blocks — the defect this
        replaces (appending a fresh copy on each content change / unpin).
        """
        stripped = re.sub(
            re.escape(self.PIN_START) + r".*?" + re.escape(self.PIN_END),
            "", base_prompt, flags=re.DOTALL).rstrip()
        block = self.get_context()
        if block:
            return stripped + "\n\n" + block
        return stripped

    def list(self) -> list[str]:
        return list(self._files.keys())

    @property
    def count(self) -> int:
        return len(self._files)
