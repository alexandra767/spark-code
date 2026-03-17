"""Tests for ReadFileTool."""

import os
from unittest.mock import patch

import pytest

from spark_code.tools.read_file import ReadFileTool


@pytest.fixture
def tool():
    return ReadFileTool()


async def test_read_existing_file_returns_content_with_line_numbers(tool, tmp_file):
    result = await tool.execute(file_path=tmp_file)
    assert "File:" in result
    assert "(5 lines)" in result
    assert "1\tline 1" in result
    assert "5\tline 5" in result


async def test_file_not_found_returns_error(tool, tmp_dir):
    result = await tool.execute(file_path=os.path.join(tmp_dir, "nonexistent.py"))
    assert "Error" in result
    assert "File not found" in result


async def test_directory_returns_error(tool, tmp_dir):
    result = await tool.execute(file_path=tmp_dir)
    assert "Error" in result
    assert "is a directory" in result


async def test_offset_parameter(tool, tmp_file):
    result = await tool.execute(file_path=tmp_file, offset=3)
    assert "showing lines 3-5" in result
    assert "3\tline 3" in result
    assert "1\tline 1" not in result


async def test_limit_parameter(tool, tmp_file):
    result = await tool.execute(file_path=tmp_file, limit=2)
    assert "showing lines 1-2" in result
    assert "1\tline 1" in result
    assert "2\tline 2" in result
    assert "3\tline 3" not in result


async def test_offset_and_limit_together(tool, tmp_file):
    result = await tool.execute(file_path=tmp_file, offset=2, limit=2)
    assert "showing lines 2-3" in result
    assert "2\tline 2" in result
    assert "3\tline 3" in result
    assert "1\tline 1" not in result
    assert "4\tline 4" not in result


async def test_empty_file_returns_empty_message(tool, tmp_dir):
    empty_path = os.path.join(tmp_dir, "empty.py")
    with open(empty_path, "w") as f:
        pass  # write nothing
    result = await tool.execute(file_path=empty_path)
    assert "File is empty" in result


async def test_binary_file_detection(tool, tmp_dir):
    bin_path = os.path.join(tmp_dir, "binary.dat")
    with open(bin_path, "wb") as f:
        f.write(b"\x00\x01\x02\xff" * 100)
    result = await tool.execute(file_path=bin_path)
    assert "binary" in result.lower()


async def test_large_file_size_check(tool, tmp_file):
    huge_size = 100 * 1024 * 1024  # 100 MB
    with patch("os.path.getsize", return_value=huge_size):
        result = await tool.execute(file_path=tmp_file)
    assert "Error" in result
    assert "too large" in result


async def test_path_validation_with_valid_path(tool, tmp_file):
    result = await tool.execute(file_path=tmp_file)
    assert "Error" not in result
    assert "File:" in result
