"""Session statistics tracking."""

import time
from collections import defaultdict


class SessionStats:
    """Tracks session-level statistics for /stats display."""

    def __init__(self):
        self.start_time = time.monotonic()
        self.tool_calls: dict[str, int] = defaultdict(int)
        self.files_read: set[str] = set()
        self.files_written: set[str] = set()
        self.files_edited: set[str] = set()
        self.commands_run: int = 0

    def record_tool_call(self, tool_name: str, args: dict):
        """Record a tool execution."""
        self.tool_calls[tool_name] += 1

        if tool_name == "read_file":
            path = args.get("file_path", "")
            if path:
                self.files_read.add(path)
        elif tool_name == "write_file":
            path = args.get("file_path", "")
            if path:
                self.files_written.add(path)
        elif tool_name == "edit_file":
            path = args.get("file_path", "")
            if path:
                self.files_edited.add(path)
        elif tool_name == "bash":
            self.commands_run += 1

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.monotonic() - self.start_time

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    def format_duration(self) -> str:
        """Format elapsed time as human-readable string."""
        elapsed = self.elapsed
        if elapsed < 60:
            return f"{elapsed:.0f}s"
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if minutes < 60:
            return f"{minutes}m {seconds}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m"
