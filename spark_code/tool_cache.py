"""Tool result caching — avoids redundant reads/globs/greps within a session."""

import hashlib
import time
from typing import Any


class ToolCache:
    """In-memory cache for read-only tool results.

    Invalidated when write_file or edit_file modifies a cached path.
    """

    def __init__(self, ttl: float = 60.0, max_entries: int = 200):
        self.ttl = ttl
        self.max_entries = max_entries
        self._cache: dict[str, tuple[float, str]] = {}  # key -> (timestamp, result)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(tool_name: str, args: dict[str, Any]) -> str:
        import json
        raw = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Return cached result or None."""
        key = self._key(tool_name, args)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        ts, result = entry
        if time.monotonic() - ts > self.ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return result

    def put(self, tool_name: str, args: dict[str, Any], result: str):
        """Store a result."""
        if len(self._cache) >= self.max_entries:
            # Evict oldest
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        key = self._key(tool_name, args)
        self._cache[key] = (time.monotonic(), result)

    def invalidate_path(self, path: str):
        """Invalidate all cache entries that reference a path.

        Checks both the cached result text and reconstructs keys for
        common tools that use the path.
        """
        to_remove = []
        # Direct key invalidation for common patterns
        for tool_name in ("read_file", "glob", "grep", "list_dir"):
            for args_dict in [
                {"file_path": path},
                {"path": path},
            ]:
                key = self._key(tool_name, args_dict)
                if key in self._cache:
                    to_remove.append(key)

        # Also scan results for the path string
        for key, (_, result) in self._cache.items():
            if path in result and key not in to_remove:
                to_remove.append(key)

        for key in to_remove:
            self._cache.pop(key, None)

    def invalidate_all(self):
        """Clear the entire cache."""
        self._cache.clear()

    @property
    def stats(self) -> dict:
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / max(1, self._hits + self._misses) * 100:.0f}%",
        }

    # Which tools are cacheable
    CACHEABLE_TOOLS = {"read_file", "glob", "grep", "list_dir"}
    # Which tools invalidate cache
    INVALIDATING_TOOLS = {"write_file", "edit_file"}
