"""Shared task list — JSON-backed store for team coordination."""

import json
import os
import time
import uuid


class Task:
    """A single task in the shared task list."""

    def __init__(self, description: str, task_id: str | None = None):
        self.id = task_id or uuid.uuid4().hex[:8]
        self.description = description
        self.status: str = "pending"  # pending | in_progress | completed | failed
        self.assigned_to: str | None = None  # worker id
        self.result: str | None = None
        self.created_at: float = time.time()
        self.completed_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "result": self.result,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        task = cls(data["description"], task_id=data["id"])
        task.status = data.get("status", "pending")
        task.assigned_to = data.get("assigned_to")
        task.result = data.get("result")
        task.created_at = data.get("created_at", time.time())
        task.completed_at = data.get("completed_at")
        return task


class TaskStore:
    """JSON-backed shared task list at ~/.spark/tasks.json."""

    def __init__(self, path: str = "~/.spark/tasks.json"):
        self.path = os.path.expanduser(path)
        self.tasks: dict[str, Task] = {}
        self.load()

    def create(self, description: str, assigned_to: str | None = None) -> Task:
        """Create a new task."""
        task = Task(description)
        if assigned_to:
            task.assigned_to = assigned_to
            task.status = "in_progress"
        self.tasks[task.id] = task
        self.save()
        return task

    def update(self, task_id: str, **kwargs) -> Task | None:
        """Update a task's fields."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        if kwargs.get("status") in ("completed", "failed"):
            task.completed_at = time.time()
        self.save()
        return task

    def get(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def list(self, status: str | None = None) -> list[Task]:
        """List tasks, optionally filtered by status."""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at)

    def clear(self):
        """Remove all tasks."""
        self.tasks.clear()
        self.save()

    def save(self):
        """Persist tasks to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {tid: t.to_dict() for tid, t in self.tasks.items()}
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load tasks from disk."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.tasks = {tid: Task.from_dict(td) for tid, td in data.items()}
        except (json.JSONDecodeError, KeyError):
            self.tasks = {}
