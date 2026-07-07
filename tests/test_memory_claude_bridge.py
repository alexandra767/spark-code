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


class TestBridgeBudget:
    """P2 Task 9: cap the injected bridge content — MEMORY.md can grow to tens
    of KB and, injected whole, single-handedly busts the fixed system-prompt
    token budget."""

    def test_under_budget_content_passes_through_unchanged(self, tmp_path):
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()
        content = "# Memory Index\n- Item one\n- Item two\n"
        (claude_dir / "MEMORY.md").write_text(content, encoding="utf-8")

        mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                     claude_memory_path=str(claude_dir), bridge_budget_chars=4000)

        assert mem.load_claude() == content

    def test_over_budget_content_truncates_at_line_boundary_with_marker(self, tmp_path):
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()
        # Each line is 10 chars incl. newline; 500 lines = 5000 chars, over a
        # 100-char budget.
        lines = [f"line-{i:04d}\n" for i in range(500)]
        content = "".join(lines)
        (claude_dir / "MEMORY.md").write_text(content, encoding="utf-8")

        mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                     claude_memory_path=str(claude_dir), bridge_budget_chars=100)

        out = mem.load_claude()
        assert len(out) < len(content)
        # Truncated at a full line boundary — no partial "line-00xx" fragment
        # right before the marker.
        body = out.split("[memory index truncated")[0]
        assert body == "" or body.endswith("\n")
        for line in body.splitlines():
            assert line in content
        assert "[memory index truncated" in out
        assert "use read_file" in out
        assert str(claude_dir) in out

    def test_budget_zero_disables_bridge_entirely(self, tmp_path):
        claude_dir = tmp_path / "claude"
        claude_dir.mkdir()
        (claude_dir / "MEMORY.md").write_text("# Memory Index\n- Item one\n", encoding="utf-8")

        mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                     claude_memory_path=str(claude_dir), bridge_budget_chars=0)

        assert mem.load_claude() == ""
        assert mem.load_all() == ""

    def test_default_budget_is_4000_chars(self, tmp_path):
        mem = Memory(global_path=str(tmp_path / "g"), project_path=str(tmp_path / "p"),
                     claude_memory_path=None)
        assert mem.bridge_budget_chars == 4000
