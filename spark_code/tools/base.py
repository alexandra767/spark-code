"""Base tool interface for Spark Code."""

import os
from abc import ABC, abstractmethod

# Maximum file size for reading (50 MB)
MAX_READ_SIZE = 50 * 1024 * 1024


def _validate_path(file_path: str, cwd: str | None = None) -> str:
    """Validate and resolve a file path. Rejects symlinks outside cwd and path traversal.

    Returns the resolved absolute path.
    Raises ValueError if the path is invalid or outside the allowed directory.
    """
    path = os.path.expanduser(file_path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    # Resolve symlinks to get the real path
    real_path = os.path.realpath(path)

    # Check for path traversal outside cwd (if cwd is provided)
    if cwd:
        real_cwd = os.path.realpath(cwd)
        if not real_path.startswith(real_cwd + os.sep) and real_path != real_cwd:
            # Allow home directory paths too (for reading configs, etc.)
            home = os.path.expanduser("~")
            if not real_path.startswith(home + os.sep) and real_path != home:
                raise ValueError(
                    f"Path '{file_path}' resolves outside allowed directories"
                )

    return path


def _backup_for_undo(path: str):
    """Save a backup of the file for /undo support (1-deep)."""
    import json
    import time
    undo_dir = os.path.expanduser("~/.spark/.undo")
    os.makedirs(undo_dir, exist_ok=True)

    # Clear old backups (keep only 1)
    try:
        for f in os.listdir(undo_dir):
            os.remove(os.path.join(undo_dir, f))
    except OSError:
        pass

    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        meta = {"path": path, "content": content}
        meta_path = os.path.join(undo_dir, f"undo_{int(time.time())}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except OSError:
        pass


def _is_binary(path: str) -> bool:
    """Check if a file is likely binary by reading first 512 bytes."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(512)
        return b"\x00" in chunk
    except OSError:
        return False


class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calling."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description for the model."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for parameters."""

    @property
    def requires_permission(self) -> bool:
        """Whether this tool needs user approval."""
        return True

    @property
    def is_read_only(self) -> bool:
        """Whether this tool only reads (doesn't modify anything)."""
        return False

    @property
    def supports_streaming(self) -> bool:
        """Whether this tool supports streaming output line-by-line."""
        return False

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool and return result as string."""

    def to_schema(self) -> dict:
        """Convert to tool schema for the model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())
