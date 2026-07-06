"""Tests for spark_code.task_store — Task and TaskStore."""

import json
import os
import time

from spark_code.task_store import TaskStore


class TestTaskStoreCreate:
    def test_create_adds_task_and_saves_to_disk(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path)
        task = store.create("Build the feature", assigned_to="agent-1")
        assert task.description == "Build the feature"
        # Verify persisted to disk
        with open(path) as f:
            data = json.load(f)
        # data is {task_id: {task_dict}} format
        assert any(v["description"] == "Build the feature" for v in data.values())


class TestTaskStoreUpdate:
    def test_update_changes_status_and_sets_completed_at(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path)
        task = store.create("Do something", assigned_to="agent-1")
        store.update(task.id, status="completed")
        updated = store.get(task.id)
        assert updated.status == "completed"
        assert updated.completed_at is not None


class TestTaskStoreList:
    def test_list_filters_by_status(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path)
        store.create("Task A", assigned_to="agent-1")
        task_b = store.create("Task B", assigned_to="agent-2")
        store.update(task_b.id, status="completed")
        in_progress = store.list(status="in_progress")
        completed = store.list(status="completed")
        assert all(t.status == "in_progress" for t in in_progress)
        assert all(t.status == "completed" for t in completed)
        assert len(in_progress) + len(completed) == 2


class TestTaskStoreClear:
    def test_clear_removes_all_tasks(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path)
        store.create("Task 1", assigned_to="agent-1")
        store.create("Task 2", assigned_to="agent-2")
        store.clear()
        assert len(store.list()) == 0


class TestTaskStoreCorruptedFile:
    def test_corrupted_file_is_backed_up_and_reset(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        # Write invalid JSON to the file
        with open(path, "w") as f:
            f.write("{{{invalid json garbage!!!")
        # Loading should handle corruption gracefully
        store = TaskStore(path)
        # Store should be empty after recovering from corruption
        assert len(store.list()) == 0
        # A backup file should exist
        backup_path = tmp_path / "tasks.json.corrupt"
        assert backup_path.exists()
        # The store should still be functional
        task = store.create("Recovery task", assigned_to="agent-1")
        assert task.description == "Recovery task"


# ---------------------------------------------------------------------------
# Regression tests — audit 2026-07-02 (bug 17)
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_no_temp_files_left_and_valid_json(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path)
        store.create("A", assigned_to="agent-1")
        store.create("B", assigned_to="agent-2")
        # File is valid JSON.
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        # No leftover temp files in the directory.
        leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
        assert leftovers == []


class TestPruning:
    def test_old_completed_tasks_are_pruned(self, tmp_path):
        from spark_code.task_store import COMPLETED_RETENTION_SECONDS
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path)

        fresh = store.create("fresh", assigned_to="a")
        store.update(fresh.id, status="completed")

        old = store.create("old", assigned_to="a")
        store.update(old.id, status="completed")
        # Backdate the old task well past the retention window.
        store.tasks[old.id].completed_at = time.time() - COMPLETED_RETENTION_SECONDS - 100
        store.save()  # prune runs inside save

        assert store.get(fresh.id) is not None
        assert store.get(old.id) is None

    def test_hard_cap_drops_oldest_terminal(self, tmp_path, monkeypatch):
        import spark_code.task_store as ts
        monkeypatch.setattr(ts, "MAX_TASKS", 5)
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path)
        # Create 8 completed tasks (all terminal).
        for i in range(8):
            t = store.create(f"t{i}", assigned_to="a")
            store.update(t.id, status="completed")
        store.save()
        assert len(store.list()) <= 5


class TestOrphanReconciliation:
    def test_in_progress_marked_stale_on_reload(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path)
        t = store.create("running task", assigned_to="worker-1")
        assert store.get(t.id).status == "in_progress"

        # Simulate a new session/process reopening the same file.
        store2 = TaskStore(path)
        reloaded = store2.get(t.id)
        assert reloaded is not None
        assert reloaded.status == "stale"
        assert reloaded.completed_at is not None
