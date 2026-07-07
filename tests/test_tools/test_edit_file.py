"""Tests for EditFileTool."""

import os

import pytest

from spark_code.tools import base as base_module
from spark_code.tools.edit_file import EditFileTool


@pytest.fixture
def tool():
    return EditFileTool()


def _neutralize_temp_allowance(monkeypatch):
    """pytest's tmp_path lives INSIDE the system temp dir, which
    _validate_path allows unconditionally as scratch space — neutralize it so
    a test exercises the cwd/root gate for real, not the temp allowance.
    See test_write_file.py's identical helper for the full rationale."""
    monkeypatch.setattr(base_module, "_temp_dir_candidates", lambda: ())


@pytest.fixture
def editable_file(tmp_dir):
    path = os.path.join(tmp_dir, "editable.py")
    with open(path, "w") as f:
        f.write("hello world\nfoo bar\ngoodbye world\n")
    return path


async def test_basic_find_and_replace(tool, editable_file):
    result = await tool.execute(
        file_path=editable_file, old_string="foo bar", new_string="baz qux"
    )
    assert "Successfully replaced" in result
    with open(editable_file) as f:
        content = f.read()
    assert "baz qux" in content
    assert "foo bar" not in content


async def test_replace_all(tool, tmp_dir):
    path = os.path.join(tmp_dir, "dupes.txt")
    with open(path, "w") as f:
        f.write("aaa bbb aaa ccc aaa\n")
    result = await tool.execute(
        file_path=path, old_string="aaa", new_string="zzz", replace_all=True
    )
    assert "Successfully replaced" in result
    assert "3 occurrence" in result
    with open(path) as f:
        assert f.read() == "zzz bbb zzz ccc zzz\n"


async def test_old_string_not_found(tool, editable_file):
    result = await tool.execute(
        file_path=editable_file, old_string="nonexistent", new_string="replacement"
    )
    assert "Error" in result
    assert "not found" in result


async def test_multiple_occurrences_without_replace_all(tool, tmp_dir):
    path = os.path.join(tmp_dir, "multi.txt")
    with open(path, "w") as f:
        f.write("word word word\n")
    result = await tool.execute(
        file_path=path, old_string="word", new_string="replaced"
    )
    assert "Error" in result
    assert "3 times" in result


async def test_file_not_found(tool, tmp_dir):
    path = os.path.join(tmp_dir, "missing.txt")
    result = await tool.execute(
        file_path=path, old_string="a", new_string="b"
    )
    assert "Error" in result
    assert "File not found" in result


async def test_preserves_other_content(tool, editable_file):
    await tool.execute(
        file_path=editable_file, old_string="foo bar", new_string="replaced"
    )
    with open(editable_file) as f:
        content = f.read()
    assert "hello world" in content
    assert "goodbye world" in content


async def test_empty_new_string_deletes(tool, editable_file):
    result = await tool.execute(
        file_path=editable_file, old_string="foo bar\n", new_string=""
    )
    assert "Successfully replaced" in result
    with open(editable_file) as f:
        content = f.read()
    assert "foo bar" not in content
    assert "hello world" in content


async def test_path_validation_valid_path(tool, editable_file):
    result = await tool.execute(
        file_path=editable_file, old_string="foo bar", new_string="ok"
    )
    assert "Error" not in result


async def test_edit_line_with_trailing_whitespace(tool, tmp_dir):
    # Regression: read_file no longer strips trailing whitespace, so a
    # byte-exact edit of a line ending in spaces must succeed.
    path = os.path.join(tmp_dir, "ws.py")
    with open(path, "w") as f:
        f.write("x = 1   \ny = 2\n")
    result = await tool.execute(
        file_path=path, old_string="x = 1   ", new_string="x = 42"
    )
    assert "Successfully replaced" in result
    with open(path) as f:
        assert f.read() == "x = 42\ny = 2\n"


async def test_edit_file_with_invalid_utf8_bytes(tool, tmp_dir):
    # edit_file reads with errors="replace" (like read_file), so a stray byte
    # doesn't make the edit fail with a codec error.
    path = os.path.join(tmp_dir, "latin.txt")
    with open(path, "wb") as f:
        f.write(b"header\ncaf\xe9 here\nfooter\n")  # 0xe9 is invalid UTF-8
    result = await tool.execute(
        file_path=path, old_string="header", new_string="HEADER"
    )
    assert "Successfully replaced" in result


# --- Finding 1 (Phase 5 whole-branch review, headline): sandbox escape via a
# model-supplied ``cwd`` — see test_write_file.py for the full rationale;
# edit_file has the identical ``cwd=cwd or self.cwd`` bug.


async def test_worktree_scoped_tool_refuses_model_cwd_escape_to_real_repo(tmp_path, monkeypatch):
    _neutralize_temp_allowance(monkeypatch)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    real_repo = tmp_path / "real_repo"
    real_repo.mkdir()
    real_file = real_repo / "existing.txt"
    real_file.write_text("secret\n")
    tool = EditFileTool(cwd=str(worktree))

    result = await tool.execute(
        file_path=str(real_file), old_string="secret", new_string="pwned",
        cwd=str(real_repo),
    )

    assert "Error" in result
    assert "outside the project" in result.lower()
    assert real_file.read_text() == "secret\n"  # untouched


async def test_worktree_scoped_tool_still_edits_inside_worktree(tmp_path, monkeypatch):
    _neutralize_temp_allowance(monkeypatch)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    target = worktree / "f.txt"
    target.write_text("hello\n")
    tool = EditFileTool(cwd=str(worktree))

    result = await tool.execute(
        file_path=str(target), old_string="hello", new_string="bye",
        cwd=str(tmp_path / "elsewhere"),
    )

    assert "Successfully replaced" in result
    assert target.read_text() == "bye\n"
