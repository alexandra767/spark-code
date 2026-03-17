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
