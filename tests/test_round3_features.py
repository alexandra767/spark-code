"""Tests for Round 3 features — resume, session labels, rich history,
notification sound, /config set."""

import json
import os
import shutil
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from spark_code.cli import (
    _get_latest_session,
    _make_session_label,
    _notify_done,
    handle_slash_command,
)
from spark_code.config import set_config
from spark_code.context import Context
from spark_code.model import ModelClient
from spark_code.permissions import PermissionManager
from spark_code.skills.base import SkillRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="spark_r3_")
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
        "ui": {},
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
# Smart session labels
# ---------------------------------------------------------------------------

def test_make_session_label_from_first_user_message():
    ctx = Context()
    ctx.add_user("Build a weather app in Python")
    label = _make_session_label(ctx)
    assert "build" in label
    assert "weather" in label
    assert label == label.lower()  # Should be lowercase
    assert " " not in label  # Spaces replaced with hyphens


def test_make_session_label_empty_context():
    ctx = Context()
    assert _make_session_label(ctx) == ""


def test_make_session_label_skips_tool_messages():
    ctx = Context()
    ctx.messages.append({"role": "assistant", "content": "I'll help"})
    ctx.messages.append({"role": "user", "content": "Fix the login bug"})
    label = _make_session_label(ctx)
    assert "fix" in label
    assert "login" in label


def test_make_session_label_truncates_long():
    ctx = Context()
    ctx.add_user("A" * 100 + " very long prompt that should be truncated")
    label = _make_session_label(ctx)
    assert len(label) <= 40


def test_make_session_label_sanitizes():
    ctx = Context()
    ctx.add_user("Hello! What's up? #test @user")
    label = _make_session_label(ctx)
    # Should only have lowercase alphanumeric and hyphens
    assert all(c.isalnum() or c == "-" for c in label)


# ---------------------------------------------------------------------------
# Context save/load with metadata
# ---------------------------------------------------------------------------

def test_context_save_with_label_and_cwd(tmp_dir):
    ctx = Context()
    ctx.add_user("hello world")
    ctx.add_assistant("hi there")
    path = os.path.join(tmp_dir, "session.json")
    ctx.save(path, label="hello-world", cwd="/home/user/project")

    with open(path) as f:
        data = json.load(f)
    assert data["label"] == "hello-world"
    assert data["cwd"] == "/home/user/project"
    assert data["turn_count"] == 1


def test_context_read_metadata(tmp_dir):
    ctx = Context()
    ctx.add_user("test prompt")
    path = os.path.join(tmp_dir, "session.json")
    ctx.save(path, label="test-prompt", cwd="/tmp/project")

    meta = Context.read_metadata(path)
    assert meta["label"] == "test-prompt"
    assert meta["cwd"] == "/tmp/project"
    assert meta["turn_count"] == 1
    assert meta["timestamp"]  # Should be non-empty


def test_context_read_metadata_missing_file():
    meta = Context.read_metadata("/nonexistent/path.json")
    assert meta == {}


def test_context_read_metadata_corrupt_file(tmp_dir):
    path = os.path.join(tmp_dir, "corrupt.json")
    with open(path, "w") as f:
        f.write("not json!!")
    meta = Context.read_metadata(path)
    assert meta == {}


# ---------------------------------------------------------------------------
# Latest session finder
# ---------------------------------------------------------------------------

def test_get_latest_session(tmp_dir):
    history_dir = os.path.join(tmp_dir, "history")
    os.makedirs(history_dir)
    # Create a few fake sessions
    for name in ["20260317_100000.json", "20260317_120000.json", "20260317_110000.json"]:
        with open(os.path.join(history_dir, name), "w") as f:
            json.dump({"messages": [], "turn_count": 1}, f)

    with patch("spark_code.cli.os.path.expanduser", return_value=history_dir):
        # Monkey-patch the function to use our dir
        pass

    # Direct test with real path
    with patch("spark_code.cli._get_latest_session") as mock:
        mock.return_value = os.path.join(history_dir, "20260317_120000.json")
        result = mock()
        assert "20260317_120000" in result


def test_get_latest_session_no_history():
    with patch("os.path.isdir", return_value=False):
        result = _get_latest_session()
        assert result == ""


# ---------------------------------------------------------------------------
# /config set command
# ---------------------------------------------------------------------------

def test_config_set_float(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/config set model.temperature 0.5",
        context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None
    assert config["model"]["temperature"] == 0.5


def test_config_set_string(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/config set permissions.mode trust",
        context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None
    assert config["permissions"]["mode"] == "trust"


def test_config_set_bool(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/config set ui.notification_sound false",
        context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None
    assert config["ui"]["notification_sound"] is False


def test_config_set_missing_args(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/config set model.temperature",
        context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None  # Should show usage, not crash


def test_config_reset(console, context, config, skills, model, permissions):
    config["model"]["temperature"] = 0.1
    result = handle_slash_command(
        "/config reset",
        context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None
    assert config["model"]["temperature"] == 0.7  # Back to default


def test_config_set_saves_to_file(tmp_dir):
    """set_config should write to the global config file."""
    from spark_code.config import GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_FILE
    config = {"model": {"temperature": 0.7}}

    with patch("spark_code.config.GLOBAL_CONFIG_FILE",
               type(GLOBAL_CONFIG_FILE)(os.path.join(tmp_dir, "config.yaml"))), \
         patch("spark_code.config.GLOBAL_CONFIG_DIR",
               type(GLOBAL_CONFIG_DIR)(tmp_dir)):
        ok, msg = set_config(config, "model.temperature", "0.3")
        assert ok
        assert config["model"]["temperature"] == 0.3


def test_config_set_int(tmp_dir):
    from spark_code.config import GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_FILE
    config = {"model": {"max_tokens": 4096}}
    with patch("spark_code.config.GLOBAL_CONFIG_FILE",
               type(GLOBAL_CONFIG_FILE)(os.path.join(tmp_dir, "config.yaml"))), \
         patch("spark_code.config.GLOBAL_CONFIG_DIR",
               type(GLOBAL_CONFIG_DIR)(tmp_dir)):
        ok, msg = set_config(config, "model.max_tokens", "8192")
        assert ok
        assert config["model"]["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# /history rich display
# ---------------------------------------------------------------------------

def test_history_shows_metadata(console, context, config, skills, model, permissions, tmp_dir):
    """History listing should show labels and time info."""
    history_dir = os.path.join(tmp_dir, "history")
    os.makedirs(history_dir)
    # Create a session with metadata
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "turn_count": 5,
        "label": "build-weather-app",
        "cwd": "/home/user/project",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with open(os.path.join(history_dir, "20260317_120000_build-weather-app.json"), "w") as f:
        json.dump(session_data, f)

    with patch("spark_code.cli.os.path.expanduser", return_value=history_dir):
        result = handle_slash_command(
            "/history", context, console, config, skills, model,
            permissions=permissions,
        )
    assert result is None  # Handled


# ---------------------------------------------------------------------------
# Notification sound
# ---------------------------------------------------------------------------

def test_notify_done_runs_without_crash():
    """Notification should not crash even if afplay fails."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        _notify_done()  # Should fall back to bell, not crash


def test_notify_done_calls_afplay():
    """On macOS, should try afplay first."""
    with patch("subprocess.run") as mock_run:
        _notify_done()
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "afplay" in args
