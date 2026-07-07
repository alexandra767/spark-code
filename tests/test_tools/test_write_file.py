"""Tests for WriteFileTool."""

import os

import pytest

from spark_code.tools import base as base_module
from spark_code.tools.base import _validate_path
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


async def test_write_outside_project_is_refused(tool):
    # Writes are now sandboxed to the project (cwd) + system temp. A path
    # elsewhere (e.g. /System) is refused *before* any OS write is attempted.
    path = "/System/nonexistent/deep/path/file.txt"
    result = await tool.execute(file_path=path, content="fail\n")
    assert "Error" in result
    assert "outside the project" in result.lower()
    assert not os.path.exists(path)


async def test_write_rejects_path_traversal_outside_cwd(tool, tmp_dir):
    # A traversal that escapes the given project root (cwd) is rejected.
    escape = os.path.join(tmp_dir, "sub", "..", "..", "..", "etc_escape.txt")
    result = await tool.execute(file_path=escape, content="x", cwd=os.path.join(tmp_dir, "sub"))
    assert "Error" in result
    assert "outside the project" in result.lower()


async def test_write_rejects_symlink_escape(tool, tmp_dir):
    # A symlink inside the project pointing outside must not be a write escape.
    project = os.path.join(tmp_dir, "project")
    os.makedirs(project)
    outside = os.path.join(os.path.expanduser("~"), ".spark_symlink_escape_target")
    link = os.path.join(project, "link.txt")
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("symlinks not supported")
    try:
        result = await tool.execute(file_path=link, content="pwned", cwd=project)
        assert "Error" in result
        assert "outside the project" in result.lower()
        assert not os.path.exists(outside)
    finally:
        if os.path.exists(outside):
            os.remove(outside)


async def test_write_within_explicit_cwd_allowed(tool, tmp_dir):
    project = os.path.join(tmp_dir, "proj")
    os.makedirs(project)
    path = os.path.join(project, "ok.txt")
    result = await tool.execute(file_path=path, content="ok\n", cwd=project)
    assert "Error" not in result
    assert os.path.exists(path)


def test_write_root_allows_sibling_dir(tmp_path, monkeypatch):
    # pytest's tmp_path lives *inside* the system temp dir, which
    # _validate_path already allows unconditionally as scratch space (see its
    # docstring) — independent of extra_roots. Point the temp-dir check at a
    # sentinel elsewhere so this test actually exercises the extra_roots gate
    # instead of incidentally passing via the pre-existing temp allowlist.
    monkeypatch.setattr(base_module.tempfile, "gettempdir", lambda: "/nonexistent-tempdir-for-test")
    proj = tmp_path / "proj"
    proj.mkdir()
    sibling = tmp_path / "other"
    sibling.mkdir()
    target = str(sibling / "f.txt")
    # without extra_roots → refused
    try:
        _validate_path(target, cwd=str(proj), for_write=True)
        assert False, "should have refused"
    except ValueError:
        pass
    # with the sibling as a write_root → allowed
    resolved = _validate_path(target, cwd=str(proj), for_write=True,
                              extra_roots=[str(sibling)])
    assert resolved.endswith("f.txt")


def test_write_root_still_refuses_outside_all_roots(tmp_path, monkeypatch):
    # See test_write_root_allows_sibling_dir above for why this is needed.
    monkeypatch.setattr(base_module.tempfile, "gettempdir", lambda: "/nonexistent-tempdir-for-test")
    proj = tmp_path / "proj"
    proj.mkdir()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        _validate_path(str(outside / "x"), cwd=str(proj), for_write=True,
                       extra_roots=[str(allowed)])
        assert False
    except ValueError:
        pass


def test_write_root_nonexistent_skipped(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    # a non-existent root must not crash; write to cwd still works
    resolved = _validate_path(str(proj / "a.txt"), cwd=str(proj), for_write=True,
                              extra_roots=["/does/not/exist/anywhere"])
    assert resolved.endswith("a.txt")
