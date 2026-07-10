"""Base tool interface for Spark Code."""

import os
import tempfile
from abc import ABC, abstractmethod

# Maximum file size for reading (50 MB)
MAX_READ_SIZE = 50 * 1024 * 1024


def _temp_dir_candidates() -> tuple[str, ...]:
    """Temp directories allowed as legitimate write scratch space.

    On macOS ``gettempdir()`` is ``/var/folders/...`` but ``/tmp`` (->
    ``/private/tmp``) is also a widely-used scratch location, so both are
    allowed. Factored into a single function so it's one patch point in tests:
    the hardcoded ``"/tmp"`` literal can't otherwise be neutralized, and on
    Linux — where pytest's ``tmp_path`` nests under ``/tmp`` — write-sandbox
    tests would silently pass via this allowance instead of the path they mean
    to exercise.
    """
    return (tempfile.gettempdir(), "/tmp")


def protected_credential_realpaths(keys_path_fn) -> set[str]:
    """Realpaths of files that may hold LITERAL secrets and must never be
    returned by any content-reading tool (read_file, grep, …).

    Covers the ``/setkey`` store plus the two ``~/.spark`` config files that can
    carry inline (non-``${ENV}``) credentials — ``config.yaml`` (inline
    ``api_key:``) and ``mcp.yaml`` (literal ``Authorization``/token headers).
    ``keys_path_fn`` is injected (each reader does ``from ..keys import
    keys_path``) so a test can redirect it at the reader's bound reference,
    matching the existing read_file monkeypatch pattern. Realpath resolution
    defeats symlink / relative / traversal aliases the same way the write
    sandbox and the MCP image sentinel do.
    """
    spark_dir = os.path.expanduser("~/.spark")
    candidates = [
        keys_path_fn(),
        os.path.join(spark_dir, "config.yaml"),
        os.path.join(spark_dir, "mcp.yaml"),
    ]
    out: set[str] = set()
    for c in candidates:
        try:
            out.add(os.path.realpath(c))
        except OSError:
            pass
    return out


def is_protected_credential_file(path: str, keys_path_fn) -> bool:
    """True if ``path`` resolves to one of the protected credential files.

    Single choke point every content-reader routes through, so a future reader
    tool can't silently reopen the exfil hole the way ``grep`` did (it had no
    keys guard while ``read_file`` did, and ``grep`` is always-allowed).
    """
    try:
        return os.path.realpath(path) in protected_credential_realpaths(keys_path_fn)
    except OSError:
        return False


def _validate_path(file_path: str, cwd: str | None = None,
                   for_write: bool = False,
                   extra_roots: list[str] | None = None) -> str:
    """Validate and resolve a file path.

    Returns the resolved absolute path.

    Reads (``for_write=False``) are permissive: a coding assistant legitimately
    needs to read configs, dependencies and files outside the project tree, so
    only path expansion/normalisation happens.

    Writes/edits (``for_write=True``) are sandboxed: the *resolved* path (after
    following symlinks — so a symlink inside the project can't be used to escape)
    must live inside the project root (``cwd``, defaulting to ``os.getcwd()``),
    the system temp directory (scratch space / build artifacts / tests), or one
    of ``extra_roots`` (Phase 5 Task 1: ``permissions.write_roots`` — additional
    project trees, e.g. a sibling repo, the caller has explicitly allowlisted
    for writes). Anything else — path traversal (``../../etc/passwd``), an
    absolute path elsewhere, or a symlink pointing outside — raises
    ``ValueError``.
    """
    path = os.path.expanduser(file_path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    # Resolve symlinks to get the real path (defeats symlink-escape).
    real_path = os.path.realpath(path)

    if for_write:
        root = os.path.realpath(cwd or os.getcwd())
        allowed_roots = [root]
        # Allow the common temp dirs as legitimate scratch space (single patch
        # point — see _temp_dir_candidates).
        for tmp_candidate in _temp_dir_candidates():
            try:
                real_tmp = os.path.realpath(tmp_candidate)
                if real_tmp not in allowed_roots:
                    allowed_roots.append(real_tmp)
            except Exception:
                pass
        # Phase 5 Task 1: additional configured write roots. Realpath'd like
        # everything else in allowed_roots — a symlink can't be used to smuggle
        # in a root the caller didn't actually intend to allowlist. A root that
        # doesn't resolve (or was never valid) just fails the prefix check
        # below rather than raising here.
        for extra in (extra_roots or []):
            try:
                real_extra = os.path.realpath(extra)
                if real_extra not in allowed_roots:
                    allowed_roots.append(real_extra)
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
