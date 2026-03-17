"""Tests for Round 2 features — SPARK.md, git banner, /stats, /diff, /memory,
model switch, connection check, inline diff, bash streaming."""

import os
import shutil
import subprocess
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from spark_code.cli import _get_git_info, handle_slash_command, load_spark_md
from spark_code.context import Context
from spark_code.memory import Memory
from spark_code.model import ModelClient
from spark_code.permissions import PermissionManager
from spark_code.skills.base import SkillRegistry
from spark_code.stats import SessionStats
from spark_code.tools.bash import BashTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="spark_r2_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def console():
    return Console(force_terminal=True, width=120)


@pytest.fixture
def context():
    return Context()


@pytest.fixture
def config():
    return {
        "model": {
            "endpoint": "http://localhost:11434",
            "name": "test-model",
            "temperature": 0.7,
            "max_tokens": 4096,
            "context_window": 32768,
            "api_key": "",
            "provider": "ollama",
        },
        "permissions": {
            "mode": "auto",
            "always_allow": ["read_file", "glob", "grep", "list_dir"],
        },
    }


@pytest.fixture
def skills():
    return SkillRegistry()


@pytest.fixture
def model():
    m = MagicMock(spec=ModelClient)
    m.total_input_tokens = 100
    m.total_output_tokens = 200
    return m


@pytest.fixture
def permissions():
    return PermissionManager(mode="auto")


@pytest.fixture
def memory(tmp_dir):
    return Memory(
        global_path=os.path.join(tmp_dir, "global_mem"),
        project_path=os.path.join(tmp_dir, "project_mem"),
    )


@pytest.fixture
def stats():
    return SessionStats()


# ---------------------------------------------------------------------------
# SPARK.md loading
# ---------------------------------------------------------------------------

def test_load_spark_md_not_found(tmp_dir):
    orig = os.getcwd()
    os.chdir(tmp_dir)
    try:
        assert load_spark_md() == ""
    finally:
        os.chdir(orig)


def test_load_spark_md_root(tmp_dir):
    orig = os.getcwd()
    os.chdir(tmp_dir)
    with open(os.path.join(tmp_dir, "SPARK.md"), "w") as f:
        f.write("# My Instructions\nDo this and that.")
    try:
        result = load_spark_md()
        assert "My Instructions" in result
    finally:
        os.chdir(orig)


def test_load_spark_md_dotdir(tmp_dir):
    orig = os.getcwd()
    os.chdir(tmp_dir)
    spark_dir = os.path.join(tmp_dir, ".spark")
    os.makedirs(spark_dir)
    with open(os.path.join(spark_dir, "SPARK.md"), "w") as f:
        f.write("# Dot Instructions")
    try:
        result = load_spark_md()
        assert "Dot Instructions" in result
    finally:
        os.chdir(orig)


def test_load_spark_md_prefers_root(tmp_dir):
    """SPARK.md at root takes priority over .spark/SPARK.md."""
    orig = os.getcwd()
    os.chdir(tmp_dir)
    with open(os.path.join(tmp_dir, "SPARK.md"), "w") as f:
        f.write("ROOT")
    spark_dir = os.path.join(tmp_dir, ".spark")
    os.makedirs(spark_dir)
    with open(os.path.join(spark_dir, "SPARK.md"), "w") as f:
        f.write("DOT")
    try:
        assert "ROOT" in load_spark_md()
    finally:
        os.chdir(orig)


# ---------------------------------------------------------------------------
# Git info
# ---------------------------------------------------------------------------

def test_git_info_in_git_repo(tmp_dir):
    orig = os.getcwd()
    os.chdir(tmp_dir)
    subprocess.run(["git", "init"], capture_output=True, cwd=tmp_dir)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                    capture_output=True, cwd=tmp_dir)
    try:
        info = _get_git_info()
        assert info  # Should have branch name
        assert "✓" in info or "*" in info
    finally:
        os.chdir(orig)


def test_git_info_not_git(tmp_dir):
    orig = os.getcwd()
    os.chdir(tmp_dir)
    try:
        info = _get_git_info()
        assert info == ""
    finally:
        os.chdir(orig)


# ---------------------------------------------------------------------------
# /stats command
# ---------------------------------------------------------------------------

def test_stats_command(console, context, config, skills, model, permissions, stats):
    stats.record_tool_call("read_file", {"file_path": "/a"})
    stats.record_tool_call("bash", {"command": "echo hi"})
    result = handle_slash_command(
        "/stats", context, console, config, skills, model,
        permissions=permissions, stats=stats,
    )
    assert result is None  # Handled


def test_stats_command_no_stats(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/stats", context, console, config, skills, model,
        permissions=permissions, stats=None,
    )
    assert result is None


# ---------------------------------------------------------------------------
# /diff command
# ---------------------------------------------------------------------------

def test_diff_command_no_git(console, context, config, skills, model, permissions, tmp_dir):
    orig = os.getcwd()
    os.chdir(tmp_dir)
    try:
        result = handle_slash_command(
            "/diff", context, console, config, skills, model,
            permissions=permissions,
        )
        assert result is None
    finally:
        os.chdir(orig)


def test_diff_command_in_git(console, context, config, skills, model, permissions, tmp_dir):
    orig = os.getcwd()
    os.chdir(tmp_dir)
    subprocess.run(["git", "init"], capture_output=True, cwd=tmp_dir)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                    capture_output=True, cwd=tmp_dir)
    try:
        result = handle_slash_command(
            "/diff", context, console, config, skills, model,
            permissions=permissions,
        )
        assert result is None
    finally:
        os.chdir(orig)


# ---------------------------------------------------------------------------
# /memory command
# ---------------------------------------------------------------------------

def test_memory_show_empty(console, context, config, skills, model, permissions, memory):
    result = handle_slash_command(
        "/memory", context, console, config, skills, model,
        permissions=permissions, memory=memory,
    )
    assert result is None


def test_memory_add(console, context, config, skills, model, permissions, memory):
    result = handle_slash_command(
        "/memory add Remember this detail", context, console, config, skills, model,
        permissions=permissions, memory=memory,
    )
    assert result is None
    # Check it was saved
    project_mem = memory.load_project()
    assert "Remember this detail" in project_mem


def test_memory_edit(console, context, config, skills, model, permissions, memory):
    result = handle_slash_command(
        "/memory edit", context, console, config, skills, model,
        permissions=permissions, memory=memory,
    )
    # Should return a prompt for the agent
    assert result is not None
    assert "memory" in result.lower()


def test_memory_not_available(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/memory", context, console, config, skills, model,
        permissions=permissions, memory=None,
    )
    assert result is None


# ---------------------------------------------------------------------------
# /model switch
# ---------------------------------------------------------------------------

def test_model_info_no_args(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/model", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None


def test_model_list(console, context, config, skills, model, permissions):
    config["providers"] = {
        "ollama": {"model": "qwen2.5:72b"},
        "gemini": {"model": "gemini-2.0-flash"},
    }
    config["model"]["provider"] = "ollama"
    result = handle_slash_command(
        "/model list", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None


def test_model_switch_signal(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/model gemini", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result == "__MODEL_SWITCH__gemini"


# ---------------------------------------------------------------------------
# Model ping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_ping_success():
    """Test ping returns success tuple on 200."""
    client = ModelClient(
        endpoint="http://localhost:11434",
        model="test",
        provider="ollama",
    )
    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        ok, msg = await client.ping()
        assert ok is True
        assert "Connected" in msg
    await client.close()


@pytest.mark.asyncio
async def test_model_ping_connection_error():
    """Test ping returns failure on connection error."""
    import httpx
    client = ModelClient(
        endpoint="http://localhost:99999",
        model="test",
        provider="ollama",
    )
    with patch.object(client._client, "get", new_callable=AsyncMock,
                       side_effect=httpx.ConnectError("fail")):
        ok, msg = await client.ping()
        assert ok is False
        assert "Cannot connect" in msg
    await client.close()


# ---------------------------------------------------------------------------
# Bash streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bash_streaming_basic():
    tool = BashTool()
    lines_received = []
    result = await tool.execute_streaming(
        command="echo 'line1' && echo 'line2'",
        callback=lambda line: lines_received.append(line),
    )
    assert "line1" in result
    assert "line2" in result
    assert "line1" in lines_received
    assert "line2" in lines_received


@pytest.mark.asyncio
async def test_bash_streaming_timeout():
    tool = BashTool()
    result = await tool.execute_streaming(
        command="sleep 10",
        timeout=1,
    )
    assert "timed out" in result.lower()


@pytest.mark.asyncio
async def test_bash_streaming_no_callback():
    """Streaming works without a callback too."""
    tool = BashTool()
    result = await tool.execute_streaming(command="echo hello")
    assert "hello" in result


# ---------------------------------------------------------------------------
# Tool supports_streaming property
# ---------------------------------------------------------------------------

def test_bash_tool_supports_streaming():
    tool = BashTool()
    assert tool.supports_streaming is True


def test_default_tool_no_streaming():
    """Base Tool.supports_streaming defaults to False."""
    from spark_code.tools.read_file import ReadFileTool
    tool = ReadFileTool()
    assert tool.supports_streaming is False


# ---------------------------------------------------------------------------
# Inline diff preview
# ---------------------------------------------------------------------------

def test_inline_diff_renders(console, tmp_dir):
    from spark_code.ui.diff import render_inline_diff
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
    # Should not raise
    render_inline_diff(console, path, "line 2", "line TWO")


def test_inline_diff_fallback_missing_file(console, tmp_dir):
    from spark_code.ui.diff import render_inline_diff
    # File doesn't exist — should fall back to simple diff
    render_inline_diff(console, "/nonexistent/file.py", "old", "new")


def test_inline_diff_fallback_string_not_found(console, tmp_dir):
    from spark_code.ui.diff import render_inline_diff
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("hello world\n")
    # old_string not in file — should fall back
    render_inline_diff(console, path, "not here", "new stuff")
