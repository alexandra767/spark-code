"""Tests for the TeamManager — background worker agents with messaging."""

import asyncio
import io

from rich.console import Console

from spark_code.task_store import TaskStore
from spark_code.team import MAX_WORKERS_LOCAL, TeamManager
from spark_code.tools.base import ToolRegistry

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
    """Spawning more than MAX_WORKERS_LOCAL should return None for the excess."""
    team = _make_team(tmp_path, model=SlowMockModel())

    workers = []
    for i in range(MAX_WORKERS_LOCAL + 1):
        w = await team.spawn(f"task {i}", name=f"w-{i}")
        workers.append(w)

    # First MAX_WORKERS_LOCAL should succeed
    for i in range(MAX_WORKERS_LOCAL):
        assert workers[i] is not None

    # The extra one should be None
    assert workers[MAX_WORKERS_LOCAL] is None

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


# ---------------------------------------------------------------------------
# Regression tests — audit 2026-07-02
# ---------------------------------------------------------------------------


class LongResultMockModel:
    """Returns a long (>500 char) text so we can test result truncation."""

    total_input_tokens = 0
    total_output_tokens = 0
    TEXT = "R" * 1000

    async def chat(self, **kwargs):
        yield {"type": "text", "content": self.TEXT}
        yield {"type": "done", "usage": {}}

    async def close(self):
        pass


async def test_duplicate_name_guard_rejects_running(tmp_path):
    """Bug 1: spawning a duplicate NAME while it's running must be rejected.

    The old guard used get_worker(name) which keys by numeric id and never
    matched a name, so this always silently created a second worker.
    """
    team = _make_team(tmp_path, model=SlowMockModel())

    w1 = await team.spawn("first", name="dupe")
    await asyncio.sleep(0.1)
    assert w1 is not None

    w2 = await team.spawn("second", name="dupe")
    assert w2 is None  # rejected as already running

    await team.stop_all()


async def test_queued_unnamed_workers_get_unique_names(tmp_path):
    """Bug 1: multiple queued UNNAMED workers must not collide on one name."""
    team = _make_team(tmp_path, model=SlowMockModel())

    # Fill all slots with named running workers.
    for i in range(MAX_WORKERS_LOCAL):
        await team.spawn(f"running {i}", name=f"run-{i}")
    await asyncio.sleep(0.1)

    # Now queue two UNNAMED workers — they should get distinct names.
    await team.spawn("queued A")
    await team.spawn("queued B")

    queued_names = [name for _, name in team._spawn_queue]
    assert len(queued_names) == 2
    assert queued_names[0] != queued_names[1]

    await team.stop_all()


async def test_stop_all_does_not_spawn_queued(tmp_path):
    """Bug 2: cancelling workers must NOT drain the queue and spawn new ones."""
    team = _make_team(tmp_path, model=SlowMockModel())

    for i in range(MAX_WORKERS_LOCAL):
        await team.spawn(f"running {i}", name=f"run-{i}")
    await asyncio.sleep(0.1)

    await team.spawn("queued", name="never-runs")
    assert len(team._spawn_queue) == 1

    await team.stop_all()
    await asyncio.sleep(0.3)  # give any (buggy) drain a chance to fire

    assert team.active_count == 0
    assert len(team._spawn_queue) == 0
    # The queued worker must never have started.
    assert team._find_worker_by_name("never-runs") is None


async def test_deliver_message_does_not_mutate_context(tmp_path):
    """Bug 3: deliver_message must enqueue only; drain_inbox injects later."""
    team = _make_team(tmp_path, model=SlowMockModel())

    w = await team.spawn("long task", name="msg-target")
    await asyncio.sleep(0.1)  # let the agent loop add the initial prompt

    before = len(w.agent.context.messages)
    result = team.deliver_message("lead", "msg-target", "hello there")
    assert "delivered" in result.lower()

    # Context must be untouched; the message sits in the inbox.
    assert len(w.agent.context.messages) == before
    assert len(w.inbox) == 1

    # Draining injects it as a user turn at a safe boundary.
    drained = w.drain_inbox()
    assert len(drained) == 1
    assert len(w.inbox) == 0
    assert len(w.agent.context.messages) == before + 1
    assert "hello there" in w.agent.context.messages[-1]["content"]

    await team.stop_all()


async def test_deliver_to_completed_worker_is_rejected(tmp_path):
    """Bug 3: no delivery to completed/dead workers."""
    team = _make_team(tmp_path)  # QuickMockModel — completes fast

    await team.spawn("quick", name="done-worker")
    await asyncio.sleep(0.4)  # let it complete
    assert team._find_worker_by_name("done-worker").status == "completed"

    result = team.deliver_message("lead", "done-worker", "you awake?")
    assert "error" in result.lower()
    assert "no longer running" in result.lower()


async def test_worker_permission_mode_defaults_to_auto_not_trust(tmp_path):
    """Bug 4: workers must not hardcode trust; default is the safer 'auto'."""
    team = _make_team(tmp_path, model=SlowMockModel())
    w = await team.spawn("task", name="perm")
    await asyncio.sleep(0.05)
    assert w.agent.permissions.mode == "auto"
    await team.stop_all()


async def test_worker_permission_mode_inherits_lead(tmp_path):
    """Bug 4: the lead's mode is passed through to workers."""
    store = TaskStore(path=str(tmp_path / "tasks.json"))
    team = TeamManager(model=SlowMockModel(), tools=ToolRegistry(),
                       console=_make_console(), task_store=store,
                       worker_permission_mode="trust")
    w = await team.spawn("task", name="perm2")
    await asyncio.sleep(0.05)
    assert w.agent.permissions.mode == "trust"
    await team.stop_all()


async def test_spawn_worker_excluded_from_worker_tools(tmp_path):
    """Bug 4: workers must not be able to recursively spawn sub-workers."""
    from spark_code.tools.spawn_worker import SpawnWorkerTool

    tools = ToolRegistry()
    tools.register(SpawnWorkerTool())
    store = TaskStore(path=str(tmp_path / "tasks.json"))
    team = TeamManager(model=SlowMockModel(), tools=tools,
                       console=_make_console(), task_store=store)

    w = await team.spawn("task", name="norecurse")
    await asyncio.sleep(0.05)

    assert w.agent.tools.get("spawn_worker") is None
    # But it still gets its own send_message tool.
    assert w.agent.tools.get("send_message") is not None

    await team.stop_all()


async def test_worker_result_persisted_full_not_truncated(tmp_path):
    """Bug 5: full worker result is durably retrievable by name."""
    team = _make_team(tmp_path, model=LongResultMockModel())

    await team.spawn("long output", name="big")
    await asyncio.sleep(0.5)  # let it complete

    result = team.get_worker_result("big")
    assert result == LongResultMockModel.TEXT
    assert len(result) == 1000  # not truncated to 500

    # Task store also holds the full result.
    tasks = team.task_store.list()
    assert any((t.result or "") == LongResultMockModel.TEXT for t in tasks)


def test_get_worker_result_none_for_unknown(tmp_path):
    """get_worker_result returns None for an unknown worker name."""
    team = _make_team(tmp_path)
    assert team.get_worker_result("nope") is None


async def test_worker_permissions_are_non_interactive(tmp_path):
    """Audit 2026-07-06 item 3: workers run on the SHARED event loop, so a
    permission prompt (blocking input()) would freeze the whole session. A worker
    in ask/auto mode must never reach _prompt_user — its PermissionManager is
    non-interactive and must never prompt.

    Review fix (2026-07-06): a non-interactive manager must NOT auto-approve
    everything (that made every worker equivalent to trust mode, regardless of
    the inherited mode). Instead it fails safe: allow exactly what the mode
    would have allowed WITHOUT a prompt, and deny anything that would have
    required one."""
    from unittest.mock import patch

    store = TaskStore(path=str(tmp_path / "tasks.json"))
    team = TeamManager(model=SlowMockModel(), tools=ToolRegistry(),
                       console=_make_console(), task_store=store,
                       worker_permission_mode="ask")
    w = await team.spawn("task", name="noprompt")
    await asyncio.sleep(0.05)

    perms = w.agent.permissions
    assert perms.mode == "ask"          # mode still inherited from the lead
    assert perms.interactive is False   # but it must not prompt

    # A write that would normally prompt must be DENIED (fail-safe), WITHOUT
    # touching Prompt.ask (patched to raise so any prompt attempt fails the test).
    with patch("spark_code.permissions.Prompt.ask",
               side_effect=AssertionError("worker must not prompt")):
        allowed = perms.check("write_file", is_read_only=False,
                              details={"file_path": "/tmp/x", "content": "y"})
    assert allowed is False
    assert perms.last_denial_reason is not None
    assert "non-interactive worker" in perms.last_denial_reason
    assert "mode=ask" in perms.last_denial_reason
    assert "write_file" in perms.last_denial_reason

    await team.stop_all()


async def test_worker_permissions_auto_mode_allows_read_denies_write(tmp_path):
    """A worker inheriting 'auto' mode may still read files without a prompt
    (that's what auto mode allows without asking), but a write — which auto
    mode would normally prompt for — must be denied, not silently approved."""
    from unittest.mock import patch

    store = TaskStore(path=str(tmp_path / "tasks.json"))
    team = TeamManager(model=SlowMockModel(), tools=ToolRegistry(),
                       console=_make_console(), task_store=store,
                       worker_permission_mode="auto")
    w = await team.spawn("task", name="autoworker")
    await asyncio.sleep(0.05)

    perms = w.agent.permissions
    assert perms.interactive is False

    with patch("spark_code.permissions.Prompt.ask",
               side_effect=AssertionError("worker must not prompt")):
        assert perms.check("read_file", is_read_only=True,
                           details={"file_path": "/tmp/x"}) is True
        allowed = perms.check("bash", is_read_only=False,
                              details={"command": "rm -rf /tmp/x"})
    assert allowed is False
    assert perms.last_denial_reason is not None

    await team.stop_all()


async def test_worker_permissions_trust_mode_allows_everything(tmp_path):
    """A worker inheriting 'trust' mode (or --yolo, which uses trust) is
    unaffected by the non-interactive fail-safe: trust still allows
    everything, since trust mode never would have prompted anyway."""
    from unittest.mock import patch

    store = TaskStore(path=str(tmp_path / "tasks.json"))
    team = TeamManager(model=SlowMockModel(), tools=ToolRegistry(),
                       console=_make_console(), task_store=store,
                       worker_permission_mode="trust")
    w = await team.spawn("task", name="trustworker")
    await asyncio.sleep(0.05)

    perms = w.agent.permissions
    assert perms.interactive is False

    with patch("spark_code.permissions.Prompt.ask",
               side_effect=AssertionError("worker must not prompt")):
        assert perms.check("write_file", is_read_only=False,
                           details={"file_path": "/tmp/x", "content": "y"}) is True
        assert perms.check("bash", is_read_only=False,
                           details={"command": "rm -rf /tmp/x"}) is True

    await team.stop_all()
