"""Tests for EditFileTool."""

import os

import pytest

from spark_code.tools.edit_file import EditFileTool


@pytest.fixture
def tool():
    return EditFileTool()


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
