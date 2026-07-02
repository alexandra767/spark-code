"""Persistent memory system across sessions."""

import os
from pathlib import Path

# Claude Code stores per-project memory under ~/.claude/projects/<encoded-cwd>/memory/,
# where <encoded-cwd> is the absolute cwd with "/" replaced by "-".
# We surface MEMORY.md (the index) only — detail files stay lazy via the read_file tool.
_CLAUDE_PROJECTS_ROOT = "~/.claude/projects"
# Legacy fallback (this user's home-encoded project) used only when the
# cwd-derived path does not exist.
DEFAULT_CLAUDE_MEMORY_PATH = "~/.claude/projects/-Users-alexandratitus/memory"

# Sentinel so we can tell "caller passed nothing" (auto-derive) apart from
# "caller explicitly passed None" (disable) and "caller passed a path".
_UNSET = object()


def encode_cwd(cwd: str) -> str:
    """Encode an absolute path the way Claude Code names its project dirs."""
    return cwd.replace("/", "-")


def resolve_claude_memory_path(cwd: str | None = None) -> Path:
    """Derive the Claude Code memory dir for the given cwd (default: current).

    Uses ~/.claude/projects/<encoded-cwd>/memory. If that doesn't exist, falls
    back to the legacy home-encoded path if IT exists; otherwise returns the
    (non-existent) derived path — load_claude() handles a missing dir gracefully.
    """
    cwd = os.path.abspath(cwd or os.getcwd())
    derived = Path(os.path.expanduser(_CLAUDE_PROJECTS_ROOT)) / encode_cwd(cwd) / "memory"
    if derived.exists():
        return derived
    fallback = Path(os.path.expanduser(DEFAULT_CLAUDE_MEMORY_PATH))
    if fallback.exists():
        return fallback
    return derived


class Memory:
    """Manages persistent memory files that get injected into context."""

    def __init__(self, global_path: str = "~/.spark/memory",
                 project_path: str = ".spark/memory",
                 claude_memory_path=_UNSET,
                 cwd: str | None = None):
        self.global_path = Path(os.path.expanduser(global_path))
        self.project_path = Path(project_path)
        if claude_memory_path is _UNSET:
            # Derive from the current working directory (correct per-project),
            # instead of hardcoding one machine-specific path for every project.
            self.claude_memory_path = resolve_claude_memory_path(cwd)
        elif claude_memory_path is None:
            self.claude_memory_path = None
        else:
            self.claude_memory_path = Path(os.path.expanduser(claude_memory_path))

    def ensure_dirs(self):
        """Create memory directories if they don't exist.

        Creates the FULL project path (parents included). The old guard only
        created the project dir when its parent already existed, so
        /memory add in a project without a .spark/ dir raised FileNotFoundError
        on the subsequent write.
        """
        self.global_path.mkdir(parents=True, exist_ok=True)
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
