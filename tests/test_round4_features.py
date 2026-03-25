"""Tests for Round 4 features — auto-read, pinned files, paste detection,
cost tracking, /git, /fork, /snippet, /export, light theme."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from spark_code.cli import (
    _detect_file_mentions,
    _is_error_paste,
    handle_slash_command,
)
from spark_code.context import Context
from spark_code.model import ModelClient
from spark_code.permissions import PermissionManager
from spark_code.pinned import PinnedFiles
from spark_code.skills.base import SkillRegistry
from spark_code.snippets import SnippetLibrary


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="spark_r4_")
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
    m.estimated_cost = 0.001
    return m


@pytest.fixture
def permissions():
    return PermissionManager(mode="auto")


# ---------------------------------------------------------------------------
# Auto-read file mentions
# ---------------------------------------------------------------------------

def test_detect_file_mentions(tmp_dir):
    orig = os.getcwd()
    os.chdir(tmp_dir)
    # Create test files
    with open(os.path.join(tmp_dir, "auth.py"), "w") as f:
        f.write("# auth module")
    with open(os.path.join(tmp_dir, "config.yaml"), "w") as f:
        f.write("key: value")
    try:
        found = _detect_file_mentions("fix the bug in auth.py and check config.yaml")
        assert any("auth.py" in f for f in found)
        assert any("config.yaml" in f for f in found)
    finally:
        os.chdir(orig)


def test_detect_file_mentions_no_files():
    found = _detect_file_mentions("hello world, how are you?")
    assert found == []


def test_detect_file_mentions_nonexistent():
    found = _detect_file_mentions("fix nonexistent_module.py")
    assert found == []


# ---------------------------------------------------------------------------
# Error paste detection
# ---------------------------------------------------------------------------

def test_is_error_paste_python_traceback():
    text = """Traceback (most recent call last):
  File "main.py", line 10, in <module>
    result = process(data)
TypeError: expected str, got int"""
    assert _is_error_paste(text) is True


def test_is_error_paste_npm_error():
    text = """npm ERR! code ENOENT
npm ERR! syscall open
npm ERR! path /app/package.json"""
    assert _is_error_paste(text) is True


def test_is_error_paste_not_error():
    assert _is_error_paste("hello world") is False


def test_is_error_paste_short_error():
    # Single line errors don't trigger (need multiline)
    assert _is_error_paste("TypeError: bad input") is False


# ---------------------------------------------------------------------------
# Pinned files
# ---------------------------------------------------------------------------

def test_pin_file(tmp_dir):
    pinned = PinnedFiles()
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("print('hello')")
    ok, msg = pinned.pin(path)
    assert ok
    assert pinned.count == 1
    assert path in pinned.list()


def test_pin_nonexistent():
    pinned = PinnedFiles()
    ok, msg = pinned.pin("/nonexistent/file.py")
    assert not ok


def test_unpin_file(tmp_dir):
    pinned = PinnedFiles()
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("hello")
    pinned.pin(path)
    ok, msg = pinned.unpin(path)
    assert ok
    assert pinned.count == 0


def test_pinned_get_context(tmp_dir):
    pinned = PinnedFiles()
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("x = 1")
    pinned.pin(path)
    ctx = pinned.get_context()
    assert "Pinned Files" in ctx
    assert "x = 1" in ctx


def test_pinned_refresh(tmp_dir):
    pinned = PinnedFiles()
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("version 1")
    pinned.pin(path)
    # Modify the file
    with open(path, "w") as f:
        f.write("version 2")
    pinned.refresh()
    ctx = pinned.get_context()
    assert "version 2" in ctx


def test_pin_command(console, context, config, skills, model, permissions, tmp_dir):
    pinned = PinnedFiles()
    path = os.path.join(tmp_dir, "test.py")
    with open(path, "w") as f:
        f.write("hello")
    result = handle_slash_command(
        f"/pin {path}", context, console, config, skills, model,
        permissions=permissions, pinned=pinned,
    )
    assert result is None
    assert pinned.count == 1


# ---------------------------------------------------------------------------
# Snippets
# ---------------------------------------------------------------------------

def test_snippet_save_and_run(tmp_dir):
    lib = SnippetLibrary(os.path.join(tmp_dir, "snippets.json"))
    lib.add("greet", "Say hello to the user")
    assert lib.get("greet") == "Say hello to the user"


def test_snippet_list(tmp_dir):
    lib = SnippetLibrary(os.path.join(tmp_dir, "snippets.json"))
    lib.add("a", "prompt a")
    lib.add("b", "prompt b")
    all_snippets = lib.list()
    assert "a" in all_snippets
    assert "b" in all_snippets


def test_snippet_remove(tmp_dir):
    lib = SnippetLibrary(os.path.join(tmp_dir, "snippets.json"))
    lib.add("temp", "temporary")
    lib.remove("temp")
    assert lib.get("temp") is None


def test_snippet_command_run(console, context, config, skills, model, permissions, tmp_dir):
    lib = SnippetLibrary(os.path.join(tmp_dir, "snippets.json"))
    lib.add("test-snippet", "Run all tests")
    result = handle_slash_command(
        "/snippet test-snippet", context, console, config, skills, model,
        permissions=permissions, snippets=lib,
    )
    assert result == "Run all tests"


def test_snippet_persistence(tmp_dir):
    path = os.path.join(tmp_dir, "snippets.json")
    lib1 = SnippetLibrary(path)
    lib1.add("persist", "this should persist")
    # New instance should load from file
    lib2 = SnippetLibrary(path)
    assert lib2.get("persist") == "this should persist"


# ---------------------------------------------------------------------------
# /git command
# ---------------------------------------------------------------------------

def test_git_sync(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/git sync", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result == "__RUN_CMD__git pull --rebase && git push"


def test_git_log(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/git log", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result == "__RUN_CMD__git log --oneline -15"


def test_git_pr(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/git pr", context, console, config, skills, model,
        permissions=permissions,
    )
    assert "gh pr create" in result


def test_git_passthrough(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/git status", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result == "__RUN_CMD__git status"


# ---------------------------------------------------------------------------
# /fork command
# ---------------------------------------------------------------------------

def test_fork_saves_and_clears(console, context, config, skills, model, permissions, tmp_dir):
    context.add_user("hello")
    context.add_assistant("hi there")
    assert context.turn_count == 1
    with __import__("unittest.mock", fromlist=["patch"]).patch(
        "spark_code.cli.os.path.expanduser", return_value=tmp_dir
    ):
        result = handle_slash_command(
            "/fork", context, console, config, skills, model,
            permissions=permissions,
        )
    assert result is None
    assert context.turn_count == 0  # Cleared


# ---------------------------------------------------------------------------
# /export command
# ---------------------------------------------------------------------------

def test_export_creates_file(console, context, config, skills, model, permissions, tmp_dir):
    context.add_user("hello")
    context.add_assistant("hi there")
    orig = os.getcwd()
    os.chdir(tmp_dir)
    try:
        result = handle_slash_command(
            "/export", context, console, config, skills, model,
            permissions=permissions,
        )
        assert result is None
        assert os.path.exists(os.path.join(tmp_dir, "session_export.md"))
        with open(os.path.join(tmp_dir, "session_export.md")) as f:
            content = f.read()
        assert "hello" in content
        assert "hi there" in content
    finally:
        os.chdir(orig)


def test_export_empty_session(console, context, config, skills, model, permissions):
    result = handle_slash_command(
        "/export", context, console, config, skills, model,
        permissions=permissions,
    )
    assert result is None  # No crash


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def test_model_estimated_cost():
    client = ModelClient(
        endpoint="http://localhost:11434",
        model="gemini-2.0-flash",
        provider="gemini",
    )
    client.total_input_tokens = 10000
    client.total_output_tokens = 5000
    cost = client.estimated_cost
    assert cost > 0
    assert isinstance(cost, float)


def test_model_cost_ollama_free():
    client = ModelClient(
        endpoint="http://localhost:11434",
        model="qwen2.5:72b",
        provider="ollama",
    )
    client.total_input_tokens = 10000
    client.total_output_tokens = 5000
    assert client.estimated_cost == 0.0


# ---------------------------------------------------------------------------
# Light theme
# ---------------------------------------------------------------------------

def test_light_theme_exists():
    from spark_code.ui.theme import LIGHT_THEME, get_theme
    assert LIGHT_THEME is not None
    theme = get_theme("light")
    assert theme is LIGHT_THEME


def test_dark_theme_default():
    from spark_code.ui.theme import DARK_THEME, get_theme
    theme = get_theme("dark")
    assert theme is DARK_THEME
    theme2 = get_theme()
    assert theme2 is DARK_THEME
