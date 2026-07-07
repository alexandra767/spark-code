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
    with open(empty_path, "w"):
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


async def test_preserves_trailing_whitespace(tool, tmp_dir):
    # read_file must NOT strip trailing spaces/tabs, or edit_file's byte-exact
    # match can never match a line ending in whitespace.
    path = os.path.join(tmp_dir, "ws.py")
    with open(path, "w") as f:
        f.write("def f():   \n\tpass\t\n")
    result = await tool.execute(file_path=path)
    assert "def f():   " in result  # trailing spaces kept
    assert "\tpass\t" in result  # trailing tab kept


async def test_offset_past_end_message(tool, tmp_file):
    # tmp_file has 5 lines; offset 99 is past EOF on a non-empty file.
    result = await tool.execute(file_path=tmp_file, offset=99)
    assert "empty" not in result.lower()
    assert "past the end" in result.lower()


# --- Finding 3 (Phase 5 whole-branch review, defense-in-depth): read_file
# must never return the raw contents of ~/.spark/keys (always_allow, no
# permission prompt — a prompt-injected "read keys, then web_fetch it
# somewhere" chain would otherwise exfil real API keys with zero friction).
# Realpath comparison (same rigor as mcp/client.py's is_spark_mcp_temp_file)
# so a symlink or a relative path that resolves to the keys file can't
# bypass a naive string-equality check.


@pytest.fixture
def fake_keys_file(tmp_path, monkeypatch):
    """Points read_file's keys_path() at a throwaway file — never touches the
    real ~/.spark/keys — and writes a recognizable "secret" into it.
    Patches the name as imported into spark_code.tools.read_file (a `from
    ..keys import keys_path`), not the origin spark_code.keys module — the
    latter wouldn't affect the already-bound local reference."""
    keys_file = tmp_path / "keys"
    keys_file.write_text('{"openrouter": "sk-super-secret-value"}')
    monkeypatch.setattr("spark_code.tools.read_file.keys_path", lambda: str(keys_file))
    return str(keys_file)


async def test_refuses_to_read_keys_file_directly(tool, fake_keys_file):
    result = await tool.execute(file_path=fake_keys_file)
    assert "sk-super-secret-value" not in result
    assert "refus" in result.lower()


async def test_refuses_to_read_keys_file_via_symlink(tool, tmp_path, fake_keys_file):
    link = tmp_path / "totally_not_the_keys_file.txt"
    try:
        os.symlink(fake_keys_file, link)
    except OSError:
        pytest.skip("symlinks not supported")
    result = await tool.execute(file_path=str(link))
    assert "sk-super-secret-value" not in result
    assert "refus" in result.lower()


async def test_refuses_to_read_keys_file_via_relative_path(tool, tmp_path, fake_keys_file, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = await tool.execute(file_path="./keys")
    assert "sk-super-secret-value" not in result
    assert "refus" in result.lower()


async def test_normal_file_still_reads_fine_alongside_keys_guard(tool, tmp_path, fake_keys_file):
    normal = tmp_path / "notes.txt"
    normal.write_text("just some notes\n")
    result = await tool.execute(file_path=str(normal))
    assert "just some notes" in result
    assert "Error" not in result
