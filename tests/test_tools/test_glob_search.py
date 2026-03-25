"""Tests for GlobTool."""

import os

import pytest

from spark_code.tools.glob_search import GlobTool


@pytest.fixture
def tool():
    return GlobTool()


@pytest.fixture
def populated_dir(tmp_dir):
    """Create a directory tree with various files for glob testing."""
    # Top-level files
    for name in ["app.py", "utils.py", "readme.txt"]:
        with open(os.path.join(tmp_dir, name), "w") as f:
            f.write(f"# {name}\n")
    # Subdirectory with files
    sub = os.path.join(tmp_dir, "sub")
    os.makedirs(sub)
    for name in ["helper.py", "data.json"]:
        with open(os.path.join(sub, name), "w") as f:
            f.write(f"# {name}\n")
    return tmp_dir


async def test_find_py_files(tool, populated_dir):
    result = await tool.execute(pattern="*.py", path=populated_dir)
    assert "app.py" in result
    assert "utils.py" in result
    # Non-recursive: should not find sub/helper.py
    assert "helper.py" not in result


async def test_recursive_pattern(tool, populated_dir):
    result = await tool.execute(pattern="**/*.py", path=populated_dir)
    assert "app.py" in result
    assert "helper.py" in result


async def test_no_results(tool, tmp_dir):
    result = await tool.execute(pattern="*.xyz", path=tmp_dir)
    assert "No files found" in result


async def test_path_parameter_narrows_search(tool, populated_dir):
    sub_path = os.path.join(populated_dir, "sub")
    result = await tool.execute(pattern="*.py", path=sub_path)
    assert "helper.py" in result
    assert "app.py" not in result


async def test_returns_file_count(tool, populated_dir):
    result = await tool.execute(pattern="**/*.py", path=populated_dir)
    assert "file(s) found" in result
    assert "3 file(s) found" in result
