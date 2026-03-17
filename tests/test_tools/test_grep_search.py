"""Tests for GrepTool."""

import os

import pytest

from spark_code.tools.grep_search import GrepTool


@pytest.fixture
def tool():
    return GrepTool()


@pytest.fixture
def searchable_dir(tmp_dir):
    """Create files with known content for grep testing."""
    with open(os.path.join(tmp_dir, "hello.py"), "w") as f:
        f.write("def greet():\n    print('Hello World')\n    return True\n")
    with open(os.path.join(tmp_dir, "utils.py"), "w") as f:
        f.write("import os\ndef helper():\n    pass\n")
    with open(os.path.join(tmp_dir, "data.txt"), "w") as f:
        f.write("Hello from text\nno match here\nHELLO uppercase\n")
    # Create a binary file that should be skipped
    with open(os.path.join(tmp_dir, "binary.dat"), "wb") as f:
        f.write(b"\x00\x01\x02Hello hidden\xff\xfe")
    return tmp_dir


async def test_find_pattern_in_files(tool, searchable_dir):
    result = await tool.execute(pattern="greet", path=searchable_dir)
    assert "hello.py" in result
    assert "greet" in result


async def test_case_insensitive_search(tool, searchable_dir):
    result = await tool.execute(
        pattern="hello", path=searchable_dir, case_insensitive=True
    )
    assert "Hello" in result or "hello" in result
    # Should match in both hello.py and data.txt
    assert "hello.py" in result
    assert "data.txt" in result


async def test_no_matches(tool, searchable_dir):
    result = await tool.execute(pattern="zzz_nonexistent_zzz", path=searchable_dir)
    assert "No matches found" in result


async def test_glob_filter(tool, searchable_dir):
    result = await tool.execute(pattern="Hello", path=searchable_dir, glob="*.py")
    assert "hello.py" in result
    assert "data.txt" not in result


async def test_regex_pattern(tool, searchable_dir):
    result = await tool.execute(pattern="def \\w+\\(\\)", path=searchable_dir)
    assert "greet" in result
    assert "helper" in result


async def test_skipped_binary_files(tool, searchable_dir):
    # Search for a pattern that exists in the binary file — should not crash
    # and the binary file should either be skipped or handled gracefully
    result = await tool.execute(pattern="Hello", path=searchable_dir)
    # Main results should still come through
    assert "hello.py" in result or "data.txt" in result
