"""Tests for the TeamManager — background worker agents with messaging."""

import asyncio
import io

import pytest
from rich.console import Console

from spark_code.team import TeamManager, Worker, Message, MAX_WORKERS
from spark_code.tools.base import ToolRegistry
from spark_code.task_store import TaskStore


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class QuickMockModel:
    """A model that immediately returns 'Done.' — workers complete fast."""

    total_input_tokens = 0
    total_output_tokens = 0

    async def chat(self, **kwargs):
        yield {"type": "text", "content": "Done."}
        yield {"type": "done", "usage": {}}

    async def close(self):
        pass


class SlowMockModel:
    """A model that sleeps before responding — useful for testing stop."""

    total_input_tokens = 0
    total_output_tokens = 0

    async def chat(self, **kwargs):
        await asyncio.sleep(10)
        yield {"type": "text", "content": "Finally done."}
        yield {"type": "done", "usage": {}}

    async def close(self):
        pass


def _make_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True)


def _make_team(tmp_path, model=None):
    """Build a TeamManager with a temp TaskStore and optional model."""
    store = TaskStore(path=str(tmp_path / "tasks.json"))
    return TeamManager(
        model=model or QuickMockModel(),
        tools=ToolRegistry(),
        console=_make_console(),
        task_store=store,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_max_workers_limit(tmp_path):
    """Spawning more than MAX_WORKERS should return None for the excess."""
    team = _make_team(tmp_path, model=SlowMockModel())

    workers = []
    for i in range(MAX_WORKERS + 1):
        w = await team.spawn(f"task {i}", name=f"w-{i}")
        workers.append(w)

    # First MAX_WORKERS should succeed
    for i in range(MAX_WORKERS):
        assert workers[i] is not None

    # The extra one should be None
    assert workers[MAX_WORKERS] is None

    # Clean up
    await team.stop_all()


async def test_message_to_lead(tmp_path):
    """deliver_message to 'lead' should appear in lead_inbox."""
    team = _make_team(tmp_path)

    result = team.deliver_message("worker-1", "lead", "Hello lead!")

    assert "delivered" in result.lower()

    msgs = team.get_lead_messages()
    assert len(msgs) == 1
    assert msgs[0].from_name == "worker-1"
    assert msgs[0].to_name == "lead"
    assert msgs[0].content == "Hello lead!"


async def test_broadcast_message(tmp_path):
    """Broadcast should deliver to all running workers and the lead."""
    team = _make_team(tmp_path, model=SlowMockModel())

    w1 = await team.spawn("task 1", name="worker-1")
    w2 = await team.spawn("task 2", name="worker-2")
    await asyncio.sleep(0.1)  # let workers start

    result = team.deliver_message("worker-1", "broadcast", "Hey everyone!")

    assert "broadcast" in result.lower()

    # Lead should have the message
    lead_msgs = team.get_lead_messages()
    assert any(m.content == "Hey everyone!" for m in lead_msgs)

    # worker-2 should have the message (worker-1 is the sender, should not)
    assert len(w2.inbox) == 1
    assert w2.inbox[0].content == "Hey everyone!"

    # worker-1 (sender) should NOT have the message
    assert len(w1.inbox) == 0

    await team.stop_all()


async def test_message_to_unknown_worker(tmp_path):
    """Sending to a nonexistent worker should return an error string."""
    team = _make_team(tmp_path)

    result = team.deliver_message("worker-1", "ghost-worker", "Hello?")

    assert "error" in result.lower() or "not found" in result.lower()


async def test_get_lead_messages_clears_inbox(tmp_path):
    """After get_lead_messages, the lead inbox should be empty."""
    team = _make_team(tmp_path)

    team.deliver_message("worker-1", "lead", "msg 1")
    team.deliver_message("worker-2", "lead", "msg 2")

    msgs = team.get_lead_messages()
    assert len(msgs) == 2

    # Second call should return empty
    msgs2 = team.get_lead_messages()
    assert len(msgs2) == 0


async def test_worker_status_tracking(tmp_path):
    """Spawned worker should appear in status() with correct info."""
    team = _make_team(tmp_path)

    w = await team.spawn("do something", name="test-worker")
    assert w is not None

    await asyncio.sleep(0.5)  # let the worker complete

    statuses = team.status()
    assert len(statuses) == 1
    assert statuses[0]["name"] == "test-worker"
    assert statuses[0]["prompt"] == "do something"
    # Worker should have completed (QuickMockModel returns immediately)
    assert statuses[0]["status"] in ("running", "completed")


async def test_stop_worker(tmp_path):
    """stop() should return True for an existing worker, False for nonexistent."""
    team = _make_team(tmp_path, model=SlowMockModel())

    w = await team.spawn("long task", name="stoppable")
    assert w is not None

    await asyncio.sleep(0.1)  # let it start

    ok = await team.stop(w.id)
    assert ok is True

    # Stopping a nonexistent worker should return False
    ok2 = await team.stop("999")
    assert ok2 is False


async def test_stop_all(tmp_path):
    """stop_all should stop every running worker."""
    team = _make_team(tmp_path, model=SlowMockModel())

    await team.spawn("task A", name="a")
    await team.spawn("task B", name="b")
    await asyncio.sleep(0.1)

    assert team.active_count == 2

    await team.stop_all()
    await asyncio.sleep(0.2)  # let cancellation settle

    # All workers should now be failed/cancelled (not running)
    assert team.active_count == 0
