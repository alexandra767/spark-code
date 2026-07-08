import os

from spark_code.instructions import Instructions, load_instructions


def _write(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_loads_project_claude_md(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "proj/CLAUDE.md", "use tabs")
    got = load_instructions(str(tmp_path / "proj"))
    assert isinstance(got, Instructions)
    assert "use tabs" in got.text
    assert "CLAUDE.md" in got.sources


def test_claude_md_walks_up_first_hit_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "root/CLAUDE.md", "root rules")
    _write(tmp_path, "root/pkg/CLAUDE.md", "pkg rules")
    (tmp_path / "root" / "pkg" / "deep").mkdir(parents=True, exist_ok=True)
    got = load_instructions(str(tmp_path / "root" / "pkg" / "deep"))
    assert "pkg rules" in got.text
    assert "root rules" not in got.text


def test_precedence_spark_md_first(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "proj/SPARK.md", "spark says")
    _write(tmp_path, "proj/CLAUDE.md", "claude says")
    got = load_instructions(str(tmp_path / "proj"))
    assert got.text.index("spark says") < got.text.index("claude says")
    assert got.sources[0] == "SPARK.md"


def test_global_claude_md_loaded_last(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "home/.claude/CLAUDE.md", "global rules")
    _write(tmp_path, "proj/CLAUDE.md", "project rules")
    got = load_instructions(str(tmp_path / "proj"))
    assert got.text.index("project rules") < got.text.index("global rules")
    assert got.sources[-1] == "~/.claude/CLAUDE.md"


def test_per_file_budget_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "proj/CLAUDE.md", "x" * 20000)
    got = load_instructions(str(tmp_path / "proj"))
    assert len(got.text) < 10000
    assert "[... truncated]" in got.text


def test_no_files_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    (tmp_path / "empty").mkdir()
    got = load_instructions(str(tmp_path / "empty"))
    assert got.text == ""
    assert got.sources == []


def test_global_spark_md_loads_from_any_cwd(tmp_path, monkeypatch):
    # ~/.spark/SPARK.md is Spark-Code's global context — loads regardless of cwd,
    # and does NOT require a ~/.claude/CLAUDE.md.
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "home/.spark/SPARK.md", "DGX SPARK KNOWLEDGE")
    (tmp_path / "work").mkdir()
    got = load_instructions(str(tmp_path / "work"))
    assert "DGX SPARK KNOWLEDGE" in got.text
    assert "~/.spark/SPARK.md" in got.sources
