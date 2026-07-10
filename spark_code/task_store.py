"""Shared task list — JSON-backed store for team coordination."""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid

logger = logging.getLogger(__name__)

# Retention / pruning knobs — keep the shared file from growing unbounded
# (it had 84 stale tasks and was never pruned).
MAX_TASKS = 200
TERMINAL_STATUSES = frozenset({"completed", "failed", "stale"})
COMPLETED_RETENTION_SECONDS = 7 * 24 * 3600  # drop terminal tasks older than 7 days


class Task:
    """A single task in the shared task list."""

    def __init__(self, description: str, task_id: str | None = None):
        self.id = task_id or uuid.uuid4().hex[:8]
        self.description = description
        self.status: str = "pending"  # pending | in_progress | completed | failed | stale
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
        self._lock = threading.Lock()  # in-process guard around writes
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

    def _prune(self):
        """Drop old terminal tasks and cap the total task count."""
        now = time.time()
        # 1. Drop terminal tasks older than the retention window.
        survivors: dict[str, Task] = {}
        for tid, t in self.tasks.items():
            if t.status in TERMINAL_STATUSES:
                ref = t.completed_at or t.created_at or now
                if (now - ref) > COMPLETED_RETENTION_SECONDS:
                    continue  # too old — prune
            survivors[tid] = t
        self.tasks = survivors

        # 2. Hard cap: if still over MAX_TASKS, drop the oldest terminal tasks.
        if len(self.tasks) > MAX_TASKS:
            terminal = sorted(
                (t for t in self.tasks.values() if t.status in TERMINAL_STATUSES),
                key=lambda t: t.completed_at or t.created_at or 0,
            )
            to_remove = len(self.tasks) - MAX_TASKS
            for t in terminal[:to_remove]:
                self.tasks.pop(t.id, None)

    def save(self):
        """Persist tasks to disk atomically (temp file + os.replace).

        Also prunes stale/old tasks first. Writing to a temp file and renaming
        means a crash mid-write can never truncate the live file (which the old
        code did, causing 'recovery' to reset everything to empty).

        KNOWN LIMITATION (cross-session lost update): the atomic replace
        prevents file CORRUPTION but not lost updates. This store is a global
        ~/.spark/tasks.json shared across CLI sessions; each session does
        load()->mutate->save(all), so two concurrent `spark` processes will
        clobber each other's tasks (last writer wins), and each load()'s
        orphan reconciliation marks the OTHER session's live in_progress tasks
        stale. In-process (a single session's asyncio workers) is safe via
        self._lock. A real fix would need per-record merge or file locking;
        not worth it at current single-user scale.
        """
        with self._lock:
            self._prune()
            dir_name = os.path.dirname(self.path) or "."
            os.makedirs(dir_name, exist_ok=True)
            data = {tid: t.to_dict() for tid, t in self.tasks.items()}

            fd, tmp_path = tempfile.mkstemp(
                dir=dir_name, prefix=".tasks-", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.path)  # atomic on POSIX
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def load(self):
        """Load tasks from disk, then reconcile orphaned in-progress tasks."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.tasks = {tid: Task.from_dict(td) for tid, td in data.items()}
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupted tasks file %s: %s — backing up and resetting", self.path, e)
            backup_path = self.path + ".corrupt"
            try:
                shutil.copy2(self.path, backup_path)
                logger.info("Backed up corrupted file to %s", backup_path)
            except OSError:
                pass
            self.tasks = {}
            return

        # Startup reconciliation: any task still 'in_progress' on load is
        # orphaned (its session/worker died) — mark it 'stale' so it doesn't
        # linger as forever-running, and persist the change.
        self._reconcile_orphans()

    def _reconcile_orphans(self):
        """Mark orphaned in_progress tasks as stale and persist if changed."""
        now = time.time()
        changed = False
        for t in self.tasks.values():
            if t.status == "in_progress":
                t.status = "stale"
                if t.completed_at is None:
                    t.completed_at = now
                changed = True
        if changed:
            self.save()
