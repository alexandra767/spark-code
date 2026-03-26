"""Tests for the wait_for_workers tool."""

import asyncio
import io

import pytest
from rich.console import Console

from spark_code.tools.wait_for_workers import WaitForWorkersTool
from spark_code.team import TeamManager, Worker


class TestWaitForWorkersTool:
    def test_tool_schema(self):
        tool = WaitForWorkersTool(team=None)
        schema = tool.to_schema()
        assert schema["name"] == "wait_for_workers"
        assert "names" in schema["parameters"]["properties"]
        assert "timeout" in schema["parameters"]["properties"]

    def test_tool_is_read_only(self):
        tool = WaitForWorkersTool(team=None)
        assert tool.is_read_only is True

    def test_returns_immediately_when_no_workers(self):
        console = Console(file=io.StringIO(), force_terminal=True)
        from spark_code.tools.base import ToolRegistry
        from spark_code.task_store import TaskStore
        team = TeamManager(model=None, tools=ToolRegistry(),
                          console=console, task_store=TaskStore())
        tool = WaitForWorkersTool(team=team)
        result = asyncio.run(tool.execute(names=[], timeout=5))
        assert "no running workers" in result.lower()

    def test_returns_completed_worker_results(self):
        console = Console(file=io.StringIO(), force_terminal=True)
        from spark_code.tools.base import ToolRegistry
        from spark_code.task_store import TaskStore
        team = TeamManager(model=None, tools=ToolRegistry(),
                          console=console, task_store=TaskStore())
        w = Worker(id="1", name="worker-test", prompt="test", status="completed",
                   result="All tests passed.")
        team.workers["1"] = w
        tool = WaitForWorkersTool(team=team)
        result = asyncio.run(tool.execute(names=["worker-test"], timeout=5))
        assert "worker-test" in result
        assert "completed" in result.lower()
