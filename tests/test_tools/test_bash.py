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
