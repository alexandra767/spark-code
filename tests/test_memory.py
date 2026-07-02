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


class TestEnsureDirsCleanProject:
    def test_append_project_in_clean_dir_without_spark_parent(self, tmp_path):
        """Bug 11: /memory add must work in a project with no .spark/ dir.

        The old ensure_dirs only created the project dir if its PARENT existed,
        so append_project raised FileNotFoundError on the first write.
        """
        clean = tmp_path / "brand_new_project"
        clean.mkdir()
        project_dir = clean / ".spark" / "memory"  # parent (.spark) does NOT exist
        assert not project_dir.parent.exists()

        mem = Memory(
            global_path=tmp_path / "g",
            project_path=project_dir,
        )
        # Must not raise.
        mem.append_project("First note.")
        assert project_dir.exists()
        assert "First note." in mem.load_project()


class TestClaudeMemoryPathDerivation:
    def test_resolve_claude_memory_path_from_cwd(self, tmp_path, monkeypatch):
        """Bug 12: the Claude memory path is derived from the cwd encoding."""
        from spark_code.memory import resolve_claude_memory_path, encode_cwd

        # Build a fake ~/.claude/projects/<encoded-cwd>/memory that exists.
        fake_home = tmp_path / "home"
        cwd = "/Users/someone/my-project"
        encoded = encode_cwd(cwd)
        target = fake_home / ".claude" / "projects" / encoded / "memory"
        target.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(fake_home))

        resolved = resolve_claude_memory_path(cwd)
        assert resolved == target

    def test_resolve_falls_back_when_derived_missing(self, tmp_path, monkeypatch):
        """When neither derived nor legacy path exists, returns derived path."""
        from spark_code.memory import resolve_claude_memory_path, encode_cwd

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        cwd = "/tmp/nonexistent-project-xyz"
        resolved = resolve_claude_memory_path(cwd)
        # Derived path (non-existent) is returned; load_claude handles missing.
        assert encode_cwd(cwd) in str(resolved)

    def test_different_projects_get_different_paths(self, tmp_path, monkeypatch):
        from spark_code.memory import resolve_claude_memory_path
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        p1 = resolve_claude_memory_path("/Users/x/proj-a")
        p2 = resolve_claude_memory_path("/Users/x/proj-b")
        assert p1 != p2
