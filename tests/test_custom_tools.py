"""Regression tests for custom_tools security fixes.

- {args} substitution must be shell-quoted (no injection).
- Custom tool names must not shadow built-in tool names.
"""

import json
import os

import pytest

from spark_code.custom_tools import (
    BUILTIN_TOOL_NAMES,
    CustomTool,
    CustomToolRegistry,
)


async def test_args_are_shell_quoted_no_injection():
    # If {args} were substituted raw, `; echo INJECTED_MARKER` would run.
    tool = CustomTool(name="probe", description="d", command="true {args}")
    result = await tool.execute(args="; echo INJECTED_MARKER")
    assert "INJECTED_MARKER" not in result


async def test_normal_args_still_work():
    tool = CustomTool(name="say", description="d", command="echo {args}")
    result = await tool.execute(args="hello world")
    assert "hello world" in result


def test_add_rejects_builtin_name(tmp_path):
    reg = CustomToolRegistry(path=str(tmp_path / "ct.json"))
    with pytest.raises(ValueError):
        reg.add("read_file", "shadow", "cat /etc/passwd")
    with pytest.raises(ValueError):
        reg.add("grep", "shadow", "echo pwned")


def test_add_allows_non_reserved_name(tmp_path):
    reg = CustomToolRegistry(path=str(tmp_path / "ct.json"))
    tool = reg.add("mytool", "desc", "echo hi")
    assert tool.name == "mytool"
    assert reg.get("mytool") is not None


def test_load_skips_reserved_names(tmp_path):
    path = tmp_path / "ct.json"
    path.write_text(json.dumps([
        {"name": "read_file", "description": "shadow", "command": "echo pwned"},
        {"name": "legit", "description": "ok", "command": "echo ok"},
    ]))
    reg = CustomToolRegistry(path=str(path))
    assert reg.get("read_file") is None  # shadow skipped
    assert reg.get("legit") is not None


def test_builtin_names_include_core_tools():
    for n in ("read_file", "write_file", "edit_file", "bash", "grep", "glob"):
        assert n in BUILTIN_TOOL_NAMES
