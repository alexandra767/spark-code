"""Persistent memory system across sessions."""

import os
from pathlib import Path

# Claude Code stores per-project memory under ~/.claude/projects/<encoded-cwd>/memory/.
# The default for this user is ~/.claude/projects/-Users-alexandratitus/memory/.
# We surface MEMORY.md (the index) only — detail files stay lazy via the read_file tool.
DEFAULT_CLAUDE_MEMORY_PATH = "~/.claude/projects/-Users-alexandratitus/memory"


class Memory:
    """Manages persistent memory files that get injected into context."""

    def __init__(self, global_path: str = "~/.spark/memory",
                 project_path: str = ".spark/memory",
                 claude_memory_path: str | None = DEFAULT_CLAUDE_MEMORY_PATH):
        self.global_path = Path(os.path.expanduser(global_path))
        self.project_path = Path(project_path)
        self.claude_memory_path = (
            Path(os.path.expanduser(claude_memory_path)) if claude_memory_path else None
        )

    def ensure_dirs(self):
        """Create memory directories if they don't exist."""
        self.global_path.mkdir(parents=True, exist_ok=True)
        if self.project_path.parent.exists():
            self.project_path.mkdir(parents=True, exist_ok=True)

    def load_global(self) -> str:
        """Load global MEMORY.md content."""
        mem_file = self.global_path / "MEMORY.md"
        if mem_file.exists():
            return mem_file.read_text(encoding="utf-8")
        return ""

    def load_project(self) -> str:
        """Load project-level memory."""
        mem_file = self.project_path / "MEMORY.md"
        if mem_file.exists():
            return mem_file.read_text(encoding="utf-8")
        return ""

    def load_claude(self) -> str:
        """Load Claude Code's MEMORY.md index (detail files stay lazy)."""
        if not self.claude_memory_path:
            return ""
        mem_file = self.claude_memory_path / "MEMORY.md"
        if mem_file.exists():
            return mem_file.read_text(encoding="utf-8")
        return ""

    def load_all(self) -> str:
        """Load all memory and return as context string."""
        parts = []

        global_mem = self.load_global()
        if global_mem:
            parts.append(f"## Global Memory\n{global_mem}")

        project_mem = self.load_project()
        if project_mem:
            parts.append(f"## Project Memory\n{project_mem}")

        claude_mem = self.load_claude()
        if claude_mem:
            # Tell the model these notes mirror Claude Code's index — detail files
            # live alongside MEMORY.md at the same path and can be opened with read_file.
            header = (
                f"## Claude Code Memory (index from {self.claude_memory_path})\n"
                "_Detail files referenced as `[name](file.md)` live in the same directory — "
                "use read_file to open them on demand._\n"
            )
            parts.append(header + claude_mem)

        if not parts:
            return ""

        return "# Persistent Memory\n\n" + "\n\n".join(parts)

    def save_global(self, content: str):
        """Save to global MEMORY.md."""
        self.ensure_dirs()
        (self.global_path / "MEMORY.md").write_text(content, encoding="utf-8")

    def save_project(self, content: str):
        """Save to project MEMORY.md."""
        self.ensure_dirs()
        (self.project_path / "MEMORY.md").write_text(content, encoding="utf-8")

    def append_global(self, entry: str):
        """Append an entry to global memory."""
        self.ensure_dirs()
        mem_file = self.global_path / "MEMORY.md"
        existing = mem_file.read_text(encoding="utf-8") if mem_file.exists() else ""
        mem_file.write_text(existing.rstrip() + "\n\n" + entry + "\n", encoding="utf-8")

    def append_project(self, entry: str):
        """Append an entry to project memory."""
        self.ensure_dirs()
        mem_file = self.project_path / "MEMORY.md"
        existing = mem_file.read_text(encoding="utf-8") if mem_file.exists() else ""
        mem_file.write_text(existing.rstrip() + "\n\n" + entry + "\n", encoding="utf-8")
