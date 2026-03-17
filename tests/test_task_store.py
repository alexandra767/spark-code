"""Tests for spark_code.task_store — Task and TaskStore."""

import json
import pytest

from spark_code.task_store import Task, TaskStore


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
