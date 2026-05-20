"""Tests for the Claude Code memory bridge in memory.py."""

from spark_code.memory import Memory


def test_load_claude_returns_empty_when_disabled(tmp_path):
    mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                 claude_memory_path=None)
    assert mem.load_claude() == ""


def test_load_claude_returns_empty_when_missing(tmp_path):
    mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                 claude_memory_path=str(tmp_path / "claude"))
    assert mem.load_claude() == ""


def test_load_claude_reads_memory_index(tmp_path):
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    (claude_dir / "MEMORY.md").write_text("# Memory Index\n- Item one\n", encoding="utf-8")

    mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                 claude_memory_path=str(claude_dir))

    out = mem.load_claude()
    assert "Memory Index" in out
    assert "Item one" in out


def test_load_all_includes_claude_section(tmp_path):
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    (claude_dir / "MEMORY.md").write_text("CLAUDE MEMORY CONTENT", encoding="utf-8")

    mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                 claude_memory_path=str(claude_dir))

    out = mem.load_all()
    assert "Claude Code Memory" in out
    assert "CLAUDE MEMORY CONTENT" in out
    assert "use read_file" in out  # hint to model about detail-file lookup


def test_load_all_combines_all_three_sources(tmp_path):
    global_dir = tmp_path / "g"
    project_dir = tmp_path / "p"
    claude_dir = tmp_path / "claude"
    global_dir.mkdir()
    project_dir.mkdir()
    claude_dir.mkdir()
    (global_dir / "MEMORY.md").write_text("GLOBAL", encoding="utf-8")
    (project_dir / "MEMORY.md").write_text("PROJECT", encoding="utf-8")
    (claude_dir / "MEMORY.md").write_text("CLAUDE", encoding="utf-8")

    mem = Memory(global_path=str(global_dir), project_path=str(project_dir),
                 claude_memory_path=str(claude_dir))

    out = mem.load_all()
    assert "GLOBAL" in out and "PROJECT" in out and "CLAUDE" in out


def test_load_all_empty_when_nothing_present(tmp_path):
    mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                 claude_memory_path=str(tmp_path / "claude"))
    assert mem.load_all() == ""
