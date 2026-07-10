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


async def test_invalid_regex_surfaces_error(tool, searchable_dir):
    # A bad regex must be reported as an error, NOT disguised as "no matches".
    result = await tool.execute(pattern="(unclosed", path=searchable_dir)
    assert "No matches found" not in result
    assert "error" in result.lower() or "invalid" in result.lower()


async def test_pattern_starting_with_dash(tool, tmp_dir):
    # A pattern that begins with '-' must be searched literally, not parsed as a
    # ripgrep flag.
    path = os.path.join(tmp_dir, "flags.txt")
    with open(path, "w") as f:
        f.write("normal line\n--dash-prefixed-token here\n")
    result = await tool.execute(pattern="--dash-prefixed-token", path=tmp_dir)
    # Must not error out as an unknown flag, and must find the line.
    assert "unknown" not in result.lower()
    assert "--dash-prefixed-token" in result


# --- SECURITY: grep must never dump the Spark credential files. grep is a
# content-reader on always_allow (no permission prompt in any mode); read_file
# guarded ~/.spark/keys but grep did not, leaving the "read keys then
# web_fetch out" exfil chain open through a sibling tool. Realpath comparison
# so a symlink/relative/traversal alias can't bypass it.

@pytest.fixture
def fake_keys_file(tmp_path, monkeypatch):
    keys_file = tmp_path / "keys"
    keys_file.write_text('{"openrouter": "sk-super-secret-value"}')
    monkeypatch.setattr("spark_code.tools.grep_search.keys_path", lambda: str(keys_file))
    return str(keys_file)


async def test_grep_refuses_keys_file_directly(tool, fake_keys_file):
    result = await tool.execute(pattern=".", path=fake_keys_file)
    assert "sk-super-secret-value" not in result
    assert "refus" in result.lower()


async def test_grep_refuses_keys_file_via_symlink(tool, tmp_path, fake_keys_file):
    link = tmp_path / "not_keys.txt"
    try:
        os.symlink(fake_keys_file, link)
    except OSError:
        pytest.skip("symlinks not supported")
    result = await tool.execute(pattern=".", path=str(link))
    assert "sk-super-secret-value" not in result
    assert "refus" in result.lower()


async def test_grep_excludes_keys_file_in_directory_walk(tool, tmp_path, monkeypatch):
    # keys file lives inside a directory that is grepped wholesale.
    keys_file = tmp_path / "keys"
    keys_file.write_text('{"openrouter": "sk-super-secret-value"}')
    (tmp_path / "app.py").write_text("print('sk-super-secret-value is not here')\n")
    monkeypatch.setattr("spark_code.tools.grep_search.keys_path", lambda: str(keys_file))
    result = await tool.execute(pattern="sk-super-secret-value", path=str(tmp_path))
    # The literal in app.py is fine to match; the keys file's own line must not appear.
    assert f"{keys_file}:" not in result


async def test_grep_python_fallback_excludes_keys_file(tool, tmp_path, monkeypatch):
    keys_file = tmp_path / "keys"
    keys_file.write_text('{"openrouter": "sk-super-secret-value"}')
    monkeypatch.setattr("spark_code.tools.grep_search.keys_path", lambda: str(keys_file))
    # Force the Python fallback path (no ripgrep).
    monkeypatch.setattr("spark_code.tools.grep_search.shutil.which", lambda _: None)
    result = await tool.execute(pattern="secret", path=str(tmp_path))
    assert "sk-super-secret-value" not in result
