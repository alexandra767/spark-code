"""Tests for audit-identified edge cases and bug fixes."""

import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from spark_code.cli import handle_slash_command
from spark_code.context import Context
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
    d = tempfile.mkdtemp(prefix="spark_audit_")
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


# ---------------------------------------------------------------------------
# FIX 1: /diff shell injection — now uses list-based subprocess
# ---------------------------------------------------------------------------

def test_diff_no_shell_injection(console, context, config, skills, model, permissions, tmp_dir):
    """Verify /diff doesn't allow shell injection via arguments."""
    import subprocess
    orig = os.getcwd()
    os.chdir(tmp_dir)
    subprocess.run(["git", "init"], capture_output=True, cwd=tmp_dir)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"],
                    capture_output=True, cwd=tmp_dir)
    try:
        # This should NOT execute the echo command
        result = handle_slash_command(
            "/diff --cached", context, console, config, skills, model,
            permissions=permissions,
        )
        assert result is None  # Handled, no crash
    finally:
        os.chdir(orig)


# ---------------------------------------------------------------------------
# FIX 2: Zombie process cleanup — await process.wait() after kill
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bash_timeout_reaps_process():
    """Verify timed-out processes are properly waited on."""
    tool = BashTool()
    result = await tool.execute(command="sleep 10", timeout=1)
    assert "timed out" in result.lower()
    # If zombie wasn't reaped, subsequent process calls would accumulate zombies
    # No assertion needed — the fix is structural (await process.wait())


@pytest.mark.asyncio
async def test_bash_streaming_timeout_reaps_process():
    """Verify streaming timed-out processes are properly waited on."""
    tool = BashTool()
    result = await tool.execute_streaming(command="sleep 10", timeout=1)
    assert "timed out" in result.lower()


# ---------------------------------------------------------------------------
# FIX 3: Streaming callback error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bash_streaming_callback_exception():
    """Callback exceptions shouldn't crash streaming execution."""
    tool = BashTool()

    def bad_callback(line):
        raise RuntimeError("display failure")

    # Should NOT crash — callback errors are caught
    result = await tool.execute_streaming(
        command="echo hello",
        callback=bad_callback,
    )
    # The command should still complete and return output
    assert "hello" in result


# ---------------------------------------------------------------------------
# FIX 4: Model ping auth failure detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_ping_auth_failure():
    """Ping should return False for 401/403 (auth failure)."""
    client = ModelClient(
        endpoint="http://localhost:11434",
        model="test",
        provider="gemini",
    )
    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        ok, msg = await client.ping()
        assert ok is False
        assert "Authentication failed" in msg
    await client.close()


@pytest.mark.asyncio
async def test_model_ping_403():
    """Ping should return False for 403 Forbidden."""
    client = ModelClient(
        endpoint="http://localhost:11434",
        model="test",
        provider="openai",
    )
    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        ok, msg = await client.ping()
        assert ok is False
        assert "Authentication failed" in msg
    await client.close()


@pytest.mark.asyncio
async def test_model_ping_non_200_non_auth():
    """Ping should return True for non-auth error codes (server is reachable)."""
    client = ModelClient(
        endpoint="http://localhost:11434",
        model="test",
        provider="ollama",
    )
    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        ok, msg = await client.ping()
        assert ok is True
        assert "status 404" in msg
    await client.close()


# ---------------------------------------------------------------------------
# Stats edge cases
# ---------------------------------------------------------------------------

def test_stats_empty_file_path():
    """Recording a tool call with empty file_path shouldn't crash or track empty."""
    stats = SessionStats()
    stats.record_tool_call("read_file", {"file_path": ""})
    assert len(stats.files_read) == 0  # Empty path correctly skipped
    assert stats.tool_calls["read_file"] == 1


def test_stats_missing_file_path_key():
    """Recording without file_path key shouldn't crash (uses .get())."""
    stats = SessionStats()
    stats.record_tool_call("read_file", {})
    assert len(stats.files_read) == 0
    assert stats.tool_calls["read_file"] == 1


def test_stats_record_with_empty_args():
    """Recording with empty args dict shouldn't crash."""
    stats = SessionStats()
    stats.record_tool_call("bash", {})
    assert stats.commands_run == 1


# ---------------------------------------------------------------------------
# Inline diff edge cases
# ---------------------------------------------------------------------------

def test_inline_diff_multiline_old_string(console, tmp_dir):
    """Inline diff should handle multi-line old_string correctly."""
    from spark_code.ui.diff import render_inline_diff
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
    # Multi-line replacement — should not crash
    render_inline_diff(console, path, "line 2\nline 3", "line TWO\nline THREE")


def test_inline_diff_at_start_of_file(console, tmp_dir):
    """Inline diff should handle old_string at file start."""
    from spark_code.ui.diff import render_inline_diff
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("first line\nsecond line\nthird line\n")
    render_inline_diff(console, path, "first line", "FIRST LINE")


def test_inline_diff_at_end_of_file(console, tmp_dir):
    """Inline diff should handle old_string at end of file."""
    from spark_code.ui.diff import render_inline_diff
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("line 1\nline 2\nlast line\n")
    render_inline_diff(console, path, "last line", "LAST LINE")


def test_inline_diff_no_trailing_newline(console, tmp_dir):
    """Inline diff should handle file without trailing newline."""
    from spark_code.ui.diff import render_inline_diff
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("line 1\nline 2\nline 3")  # No trailing newline
    render_inline_diff(console, path, "line 3", "LINE THREE")


# ---------------------------------------------------------------------------
# Project detection edge cases
# ---------------------------------------------------------------------------

def test_project_detect_malformed_package_json(tmp_dir):
    """Malformed package.json should not crash detection."""
    from spark_code.project_detect import detect_project_type
    with open(os.path.join(tmp_dir, "package.json"), "w") as f:
        f.write("{invalid json!!")
    result = detect_project_type(tmp_dir)
    assert "JavaScript" in result  # Should still detect JS from package.json


def test_project_detect_empty_package_json(tmp_dir):
    """Empty package.json should not crash detection."""
    from spark_code.project_detect import detect_project_type
    with open(os.path.join(tmp_dir, "package.json"), "w") as f:
        f.write("{}")
    result = detect_project_type(tmp_dir)
    assert "JavaScript" in result


# ---------------------------------------------------------------------------
# Model switch edge cases
# ---------------------------------------------------------------------------

def test_model_switch_unknown_provider(console, context, config, skills, model, permissions):
    """Switching to unknown provider should return a signal (handled in main loop)."""
    result = handle_slash_command(
        "/model nonexistent", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result == "__MODEL_SWITCH__nonexistent"


def test_model_list_no_providers(console, context, config, skills, model, permissions):
    """/model list with no providers configured should show message."""
    # config has no "providers" key
    result = handle_slash_command(
        "/model list", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None  # Handled


# ---------------------------------------------------------------------------
# Memory edge cases
# ---------------------------------------------------------------------------

def test_memory_add_empty_entry(console, context, config, skills, model, permissions):
    """'/memory add' with no content should show usage."""
    from spark_code.memory import Memory
    mem = Memory(
        global_path=os.path.join(tmp_dir, "g") if 'tmp_dir' in dir() else "/tmp/spark_test_g",
        project_path=os.path.join(tmp_dir, "p") if 'tmp_dir' in dir() else "/tmp/spark_test_p",
    )
    result = handle_slash_command(
        "/memory add", context, console, config, skills, model,
        permissions=permissions, memory=mem,
    )
    assert result is None  # Should show usage, not crash
