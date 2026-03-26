"""Spawn worker tool — lets the lead agent create background workers."""

from .base import Tool


class SpawnWorkerTool(Tool):
    """Tool that lets the lead agent spawn background worker agents."""

    name = "spawn_worker"
    description = (
        "Spawn ONE background worker agent for ONE subtask. "
        "Call this tool multiple times (once per worker) to spawn multiple workers. "
        "Each call creates one worker. Max 2 concurrent workers for local models, 6 for cloud. "
        "Do NOT combine multiple tasks into one call."
    )
    is_read_only = False
    requires_permission = False

    def __init__(self):
        self._team_manager = None

    def set_team_manager(self, team_manager):
        """Bind to the team manager. Called during CLI setup."""
        self._team_manager = team_manager

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The task for the worker to complete. Be specific — "
                        "include file paths, what to create, and any context needed."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Optional name for the worker (e.g. 'backend', 'tests'). "
                        "Defaults to 'worker-N'."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str = "", name: str = "", **kwargs) -> str:
        # Handle fallback when JSON parsing wrapped args in {"raw": "..."}
        if not task and "raw" in kwargs:
            import json
            raw = kwargs["raw"]
            # Try parsing as JSON (or first JSON object if concatenated)
            try:
                parsed = json.loads(raw)
                task = parsed.get("task", "")
                name = parsed.get("name", name)
            except (json.JSONDecodeError, TypeError):
                # Try extracting first JSON object from concatenated string
                if raw.startswith("{"):
                    depth = 0
                    for i, ch in enumerate(raw):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                try:
                                    parsed = json.loads(raw[:i + 1])
                                    task = parsed.get("task", "")
                                    name = parsed.get("name", name)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                                break
                if not task:
                    task = raw  # Use raw string as the task itself

        if not task:
            return "Error: 'task' is required."
        if not self._team_manager:
            return "Error: Team system not available."

        worker = await self._team_manager.spawn(task, name=name)
        if not worker:
            return "Error: Could not spawn worker (max workers reached)."

        return (
            f"Worker '{worker.name}' (#{worker.id}) spawned and running.\n"
            f"Task: {task}\n"
            f"The worker is running in the background. "
            f"Continue with other work or spawn more workers."
        )
