"""Tests for BashTool."""

import os

import pytest

from spark_code.tools.bash import BashTool


@pytest.fixture
def tool():
    return BashTool()


async def test_simple_echo_command(tool):
    result = await tool.execute(command="echo hello world")
    assert "hello world" in result


async def test_command_with_nonzero_exit_code(tool):
    result = await tool.execute(command="exit 42")
    assert "Exit code: 42" in result


async def test_stderr_captured(tool):
    result = await tool.execute(command="echo error_msg >&2")
    assert "error_msg" in result


async def test_timeout_handling(tool):
    result = await tool.execute(command="sleep 10", timeout=1)
    assert "timed out" in result


async def test_background_mode_returns_pid(tool):
    result = await tool.execute(command="sleep 0.1", background=True)
    assert "PID" in result
    assert "background" in result.lower()


async def test_gui_detection_open_command(tool):
    assert tool._is_gui_command("open .") is True


async def test_gui_detection_python_pygame(tool, tmp_dir):
    script = os.path.join(tmp_dir, "game.py")
    with open(script, "w") as f:
        f.write("import pygame\npygame.init()\n")
    # _is_gui_command reads the script relative to cwd
    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)
        assert tool._is_gui_command("python game.py") is True
    finally:
        os.chdir(orig_cwd)


async def test_empty_command_output(tool):
    result = await tool.execute(command="true")
    # 'true' produces no output; should show exit code 0 message
    assert "exit code 0" in result.lower() or result.strip() == ""


async def test_timeout_returns_partial_output(tool):
    # Output produced before the timeout must be returned, not discarded.
    result = await tool.execute(
        command="echo partial_marker_123; sleep 10", timeout=1
    )
    assert "partial_marker_123" in result
    assert "timed out" in result.lower()


async def test_exit_code_preserved_after_truncation(tool):
    # Large failing output: the middle is truncated but the exit code (appended
    # at the tail) must survive.
    from spark_code.tools.bash import _MAX_OUTPUT_CHARS
    result = await tool.execute(
        command="yes 0123456789012345678901234567890123456789 | head -n 6000; exit 7"
    )
    assert "Exit code: 7" in result
    assert "truncated" in result.lower()
    assert len(result) < _MAX_OUTPUT_CHARS + 2000


async def test_streaming_handles_long_single_line(tool):
    # A single line far larger than the old 64KB asyncio default must not
    # nuke the command output.
    result = await tool.execute_streaming(
        command="head -c 200000 /dev/zero | tr '\\0' A"
    )
    assert "Error executing command" not in result
    assert "A" * 1000 in result


async def test_streaming_honors_background(tool):
    # background=true must not block-until-timeout in the streaming path.
    result = await tool.execute_streaming(command="sleep 0.1", background=True)
    assert "PID" in result
    assert "background" in result.lower()


async def test_timeout_kills_child_process_tree(tool):
    # The child `sleep` (grandchild of the shell) must be killed on timeout;
    # the call must return promptly rather than hang for the full sleep.
    import time
    start = time.monotonic()
    result = await tool.execute(command="sleep 30", timeout=1)
    elapsed = time.monotonic() - start
    assert "timed out" in result.lower()
    assert elapsed < 10  # would be ~30s if the tree weren't killed


# --- cwd scoping (Phase 5 Task 8) -------------------------------------------
# BashTool(cwd=...) runs its child process in that directory via the
# subprocess cwd= kwarg — per-child, never os.chdir()'ing the parent — so an
# isolated implementer sub-agent's bash lands in its worktree, not the user's
# real tree.


async def test_cwd_override_runs_child_in_that_dir(tmp_dir):
    scoped = BashTool(cwd=tmp_dir)
    result = await scoped.execute(command="pwd")
    assert os.path.realpath(result.strip()) == os.path.realpath(tmp_dir)


async def test_cwd_override_writes_land_in_that_dir(tmp_dir):
    scoped = BashTool(cwd=tmp_dir)
    await scoped.execute(command="echo hi > scoped_file.txt")
    # The file lands in the override dir, NOT in the process cwd.
    assert os.path.isfile(os.path.join(tmp_dir, "scoped_file.txt"))
    assert not os.path.isfile(os.path.join(os.getcwd(), "scoped_file.txt"))


async def test_cwd_override_streaming_runs_child_in_that_dir(tmp_dir):
    scoped = BashTool(cwd=tmp_dir)
    result = await scoped.execute_streaming(command="pwd")
    assert os.path.realpath(result.strip()) == os.path.realpath(tmp_dir)


async def test_cwd_override_background_runs_child_in_that_dir(tmp_dir):
    scoped = BashTool(cwd=tmp_dir)
    # Background write, then briefly wait and confirm it landed in the override.
    await scoped.execute(command="echo bg > bg_file.txt; sleep 0", background=True)
    import asyncio
    for _ in range(50):
        if os.path.isfile(os.path.join(tmp_dir, "bg_file.txt")):
            break
        await asyncio.sleep(0.02)
    assert os.path.isfile(os.path.join(tmp_dir, "bg_file.txt"))
    assert not os.path.isfile(os.path.join(os.getcwd(), "bg_file.txt"))


async def test_default_bash_uses_process_cwd(tmp_dir):
    # BashTool() with no override is unchanged: runs in the process cwd.
    default = BashTool()
    result = await default.execute(command="pwd")
    assert os.path.realpath(result.strip()) == os.path.realpath(os.getcwd())
