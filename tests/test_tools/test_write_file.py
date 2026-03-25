"""Tests for WriteFileTool."""

import os

import pytest

from spark_code.tools.write_file import WriteFileTool


@pytest.fixture
def tool():
    return WriteFileTool()


async def test_create_new_file(tool, tmp_dir):
    path = os.path.join(tmp_dir, "new_file.txt")
    result = await tool.execute(file_path=path, content="hello world\n")
    assert "Successfully wrote" in result
    assert os.path.exists(path)
    with open(path) as f:
        assert f.read() == "hello world\n"


async def test_overwrite_existing_file(tool, tmp_file):
    result = await tool.execute(file_path=tmp_file, content="replaced\n")
    assert "Successfully wrote" in result
    with open(tmp_file) as f:
        assert f.read() == "replaced\n"


async def test_creates_parent_directories(tool, tmp_dir):
    path = os.path.join(tmp_dir, "sub", "deep", "file.txt")
    result = await tool.execute(file_path=path, content="nested\n")
    assert "Successfully wrote" in result
    assert os.path.exists(path)


async def test_reports_correct_line_count(tool, tmp_dir):
    path = os.path.join(tmp_dir, "lines.txt")
    content = "line 1\nline 2\nline 3\n"
    result = await tool.execute(file_path=path, content=content)
    assert "3 lines" in result


async def test_handles_empty_content(tool, tmp_dir):
    path = os.path.join(tmp_dir, "empty.txt")
    result = await tool.execute(file_path=path, content="")
    assert "Successfully wrote" in result
    assert "0 lines" in result
    with open(path) as f:
        assert f.read() == ""


async def test_writes_utf8_content(tool, tmp_dir):
    path = os.path.join(tmp_dir, "utf8.txt")
    result = await tool.execute(file_path=path, content="caf\u00e9\n")
    assert "Successfully wrote" in result
    with open(path, encoding="utf-8") as f:
        assert f.read() == "caf\u00e9\n"


async def test_path_validation_valid_path(tool, tmp_dir):
    path = os.path.join(tmp_dir, "valid.txt")
    result = await tool.execute(file_path=path, content="ok\n")
    assert "Error" not in result


async def test_error_on_unwritable_location(tool):
    # On macOS, /System is read-only — makedirs raises PermissionError
    path = "/System/nonexistent/deep/path/file.txt"
    with pytest.raises(PermissionError):
        await tool.execute(file_path=path, content="fail\n")
