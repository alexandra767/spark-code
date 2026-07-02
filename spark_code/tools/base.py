"""Base tool interface for Spark Code."""

import os
import tempfile
from abc import ABC, abstractmethod

# Maximum file size for reading (50 MB)
MAX_READ_SIZE = 50 * 1024 * 1024


def _validate_path(file_path: str, cwd: str | None = None,
                   for_write: bool = False) -> str:
    """Validate and resolve a file path.

    Returns the resolved absolute path.

    Reads (``for_write=False``) are permissive: a coding assistant legitimately
    needs to read configs, dependencies and files outside the project tree, so
    only path expansion/normalisation happens.

    Writes/edits (``for_write=True``) are sandboxed: the *resolved* path (after
    following symlinks — so a symlink inside the project can't be used to escape)
    must live inside the project root (``cwd``, defaulting to ``os.getcwd()``) or
    the system temp directory (scratch space / build artifacts / tests). Anything
    else — path traversal (``../../etc/passwd``), an absolute path elsewhere, or a
    symlink pointing outside — raises ``ValueError``.
    """
    path = os.path.expanduser(file_path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    # Resolve symlinks to get the real path (defeats symlink-escape).
    real_path = os.path.realpath(path)

    if for_write:
        root = os.path.realpath(cwd or os.getcwd())
        allowed_roots = [root]
        # Allow the common temp dirs as legitimate scratch space. On macOS
        # gettempdir() is /var/folders/... but /tmp (-> /private/tmp) is also a
        # widely-used scratch location, so allow both.
        for tmp_candidate in (tempfile.gettempdir(), "/tmp"):
            try:
                real_tmp = os.path.realpath(tmp_candidate)
                if real_tmp not in allowed_roots:
                    allowed_roots.append(real_tmp)
            except Exception:
                pass

        def _within(target: str, base: str) -> bool:
            return target == base or target.startswith(base + os.sep)

        if not any(_within(real_path, r) for r in allowed_roots):
            raise ValueError(
                f"Refusing to write outside the project directory: '{file_path}' "
                f"resolves to '{real_path}', which is outside '{root}'."
            )

    return path


MAX_UNDO_DEPTH = 20


def _backup_for_undo(path: str):
    """Save a backup of the file for /undo support (multi-file, depth-limited stack)."""
    import json
    import time
    undo_dir = os.path.expanduser("~/.spark/.undo")
    os.makedirs(undo_dir, exist_ok=True)

    # Prune old backups beyond MAX_UNDO_DEPTH
    try:
        entries = sorted(os.listdir(undo_dir), reverse=True)
        for old in entries[MAX_UNDO_DEPTH - 1:]:
            os.remove(os.path.join(undo_dir, old))
    except OSError:
        pass

    if not os.path.exists(path):
        return

    # Skip undo backup for binary / non-UTF-8 files: the /undo restore writes the
    # stored content back as UTF-8 text, so backing up a binary file with
    # errors="replace" would silently corrupt it on restore.
    if _is_binary(path):
        return

    try:
        with open(path, "rb") as f:
            raw = f.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return  # non-UTF-8: skip undo rather than corrupt on restore
        meta = {"path": path, "content": content}
        # Use monotonic counter + timestamp for proper ordering
        meta_path = os.path.join(undo_dir, f"undo_{int(time.time() * 1000)}.json")
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
