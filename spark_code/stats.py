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
        self.last_tokens_per_sec: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._cost_input_rate: float = 0.0
        self._cost_output_rate: float = 0.0
        self.files_created: set[str] = set()

    def record_generation_speed(self, tokens: int, elapsed: float):
        """Record the speed of the last generation."""
        if elapsed > 0:
            self.last_tokens_per_sec = tokens / elapsed
        else:
            self.last_tokens_per_sec = 0.0

    def format_speed(self) -> str:
        """Format speed for display. Empty string if no data."""
        if self.last_tokens_per_sec > 0:
            return f"{self.last_tokens_per_sec:.1f} tok/s"
        return ""

    def set_cost_rates(self, input_rate: float = 0.0, output_rate: float = 0.0):
        """Set cost per million tokens (input and output)."""
        self._cost_input_rate = input_rate
        self._cost_output_rate = output_rate

    def record_token_usage(self, input_tokens: int = 0, output_tokens: int = 0):
        """Accumulate token counts."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def session_cost(self) -> float:
        """Calculate session cost in dollars."""
        input_cost = (self.input_tokens * self._cost_input_rate) / 1_000_000
        output_cost = (self.output_tokens * self._cost_output_rate) / 1_000_000
        return input_cost + output_cost

    def format_cost(self) -> str:
        """Format cost for display. Empty string if zero."""
        cost = self.session_cost
        if cost > 0:
            if cost < 0.01:
                return f"${cost:.4f}"
            return f"${cost:.2f}"
        return ""

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

    def record_file_created(self, path: str):
        """Record a newly created file (not an edit of existing)."""
        self.files_created.add(path)

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
