"""Tests for spark_code.memory.Memory."""

import pytest
from pathlib import Path

from spark_code.memory import Memory


class TestLoadGlobal:
    def test_load_global_returns_empty_for_missing_file(self, tmp_path):
        mem = Memory(
            global_path=tmp_path / "global",
            project_path=tmp_path / "project",
        )
        result = mem.load_global()
        assert result == "" or result is None


class TestSaveLoadRoundtrip:
    def test_save_global_then_load_global(self, tmp_path):
        global_dir = tmp_path / "global"
        mem = Memory(
            global_path=global_dir,
            project_path=tmp_path / "project",
        )
        mem.save_global("# Global Memory\nSome notes here.")
        loaded = mem.load_global()
        assert "Global Memory" in loaded
        assert "Some notes here." in loaded


class TestAppendGlobal:
    def test_append_global_adds_to_existing(self, tmp_path):
        global_dir = tmp_path / "global"
        mem = Memory(
            global_path=global_dir,
            project_path=tmp_path / "project",
        )
        mem.save_global("# Existing content\n")
        mem.append_global("New entry added.")
        loaded = mem.load_global()
        assert "Existing content" in loaded
        assert "New entry added." in loaded


class TestLoadAll:
    def test_load_all_combines_global_and_project(self, tmp_path):
        global_dir = tmp_path / "global"
        project_dir = tmp_path / "project"
        mem = Memory(
            global_path=global_dir,
            project_path=project_dir,
        )
        mem.save_global("Global stuff.")
        mem.save_project("Project stuff.")
        combined = mem.load_all()
        assert "Global stuff." in combined
        assert "Project stuff." in combined


class TestEnsureDirs:
    def test_ensure_dirs_creates_directories(self, tmp_path):
        global_dir = tmp_path / "deep" / "nested" / "global"
        project_dir = tmp_path / "deep" / "nested" / "project"
        mem = Memory(
            global_path=global_dir,
            project_path=project_dir,
        )
        mem.ensure_dirs()
        assert global_dir.exists()
        assert project_dir.exists()
