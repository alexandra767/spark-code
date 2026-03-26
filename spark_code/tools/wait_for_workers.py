"""Tool that waits for background workers to complete."""

import asyncio

from spark_code.tools.base import Tool


class WaitForWorkersTool(Tool):
    """Wait for background workers to finish and return their results."""

    name = "wait_for_workers"
    description = (
        "Wait for background workers to complete and return their results. "
        "Optionally specify worker names; defaults to waiting for all running workers."
    )
    is_read_only = True
    requires_permission = False

    def __init__(self, team):
        self._team = team

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific worker names to wait for. Empty = all.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Max seconds to wait. Default: 300.",
                },
            },
        }

    async def execute(self, names=None, timeout=300, **kwargs) -> str:
        if not self._team:
            return "No team manager available."

        workers = list(self._team.workers.values())
        if not workers:
            return "No running workers to wait for."

        if names:
            targets = [w for w in workers if w.name in names]
            if not targets:
                return f"No workers found with names: {', '.join(names)}"
        else:
            targets = [w for w in workers if w.status == "running"]
            if not targets:
                return "No running workers to wait for."

        elapsed = 0.0
        poll_interval = 2.0
        while elapsed < timeout:
            still_running = [w for w in targets if w.status == "running"]
            if not still_running:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        lines = []
        for w in targets:
            status = w.status
            result = w.result[:500] if w.result else "(no output)"
            lines.append(f"- {w.name} [{status}]: {result}")

        still_running = [w for w in targets if w.status == "running"]
        if still_running:
            names_str = ", ".join(w.name for w in still_running)
            lines.append(f"\nTimeout reached. Still running: {names_str}")

        return "\n".join(lines)
