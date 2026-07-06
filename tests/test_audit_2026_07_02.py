"""Regression tests for the 2026-07-02 audit fixes.

Covers branches (honest merge + project scoping), watcher (no `time` import,
off-thread scan), plan_executor (stop-on-failure), and projectplan (async RAG
fetch + machine-agnostic service URL).
"""

import inspect
import io

from rich.console import Console

from spark_code.context import Context

# ---------------------------------------------------------------------------
# branches.py — bug 15
# ---------------------------------------------------------------------------


class TestBranchMergeHonest:
    def test_merge_actually_merges_messages(self, tmp_path):
        from spark_code.branches import BranchManager

        bm = BranchManager(branch_dir=str(tmp_path))
        src = Context()
        src.add_user("do the thing")
        src.add_assistant("thing done")
        bm.save_branch("feature", src)

        bm._current_branch = "main"
        target = Context()
        ok, msg = bm.merge_branch("feature", target)

        assert ok
        contents = [m["content"] for m in target.messages
                    if isinstance(m.get("content"), str)]
        # The ACTUAL content is present — not just a claim it was incorporated.
        assert any("do the thing" in c for c in contents)
        assert any("thing done" in c for c in contents)
        # And the message no longer makes the false "incorporated" claim.
        assert "incorporated" not in msg.lower()

    def test_merge_skips_structured_tool_messages(self, tmp_path):
        """Tool-call / tool-result turns are skipped to keep a valid sequence."""
        from spark_code.branches import BranchManager

        bm = BranchManager(branch_dir=str(tmp_path))
        src = Context()
        src.add_user("plain user")
        # A structured assistant tool_calls message (non-str content).
        src.messages.append({"role": "assistant", "content": [{"type": "tool_use"}]})
        src.add_assistant("plain assistant")
        bm.save_branch("feat", src)

        bm._current_branch = "main"
        target = Context()
        ok, msg = bm.merge_branch("feat", target)
        assert ok
        contents = [m["content"] for m in target.messages
                    if isinstance(m.get("content"), str)]
        assert any("plain user" in c for c in contents)
        assert any("plain assistant" in c for c in contents)
        # The structured tool message must NOT have been spliced in.
        assert all(isinstance(m.get("content"), str) for m in target.messages)


class TestBranchProjectScoping:
    def test_branches_are_scoped_per_project(self, tmp_path):
        from spark_code.branches import BranchManager

        base = str(tmp_path / "branches")
        bm_a = BranchManager(branch_dir=base, project_dir="/proj/a")
        bm_b = BranchManager(branch_dir=base, project_dir="/proj/b")

        ctx = Context()
        ctx.add_user("hi")
        bm_a.save_branch("only-in-a", ctx)

        assert "only-in-a" in [b["name"] for b in bm_a.list_branches()]
        assert "only-in-a" not in [b["name"] for b in bm_b.list_branches()]


# ---------------------------------------------------------------------------
# watcher.py — bug 16
# ---------------------------------------------------------------------------


class TestWatcher:
    def test_no_unused_time_import(self):
        import spark_code.watcher as w
        assert not hasattr(w, "time")

    def test_scan_finds_watched_files(self, tmp_path):
        from spark_code.watcher import FileWatcher
        console = Console(file=io.StringIO(), force_terminal=True)
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "ignore.txt").write_text("nope\n")
        fw = FileWatcher(command="true", console=console, directory=str(tmp_path))
        snap = fw._scan()
        assert any(p.endswith("a.py") for p in snap)
        assert not any(p.endswith("ignore.txt") for p in snap)

    async def test_watch_loop_scans_off_thread(self, tmp_path):
        """The loop must invoke _scan (wrapped in asyncio.to_thread)."""
        from unittest.mock import MagicMock

        from spark_code.watcher import FileWatcher

        console = Console(file=io.StringIO(), force_terminal=True)
        fw = FileWatcher(command="true", console=console, directory=str(tmp_path))
        fw._scan = MagicMock(return_value={})
        await fw.start()
        import asyncio
        await asyncio.sleep(0.1)  # initial scan happens immediately
        await fw.stop()
        assert fw._scan.call_count >= 1


# ---------------------------------------------------------------------------
# plan_executor.py — bug 8 (stop-on-failure)
# ---------------------------------------------------------------------------


class _FakeContext:
    def __init__(self):
        self.messages = []

    def add_user(self, c):
        self.messages.append(("user", c))

    def add_assistant(self, c):
        self.messages.append(("assistant", c))


class _FakeAgent:
    def __init__(self, fail_marker):
        self.context = _FakeContext()
        self.calls = []
        self._fail_marker = fail_marker

    async def run(self, task_desc):
        self.calls.append(task_desc)
        if self._fail_marker in task_desc:
            raise RuntimeError("boom")
        return "ok"


class _FakeTeam:
    max_workers = 2

    @property
    def active_count(self):
        return 0


_SEQUENTIAL_PLAN = """# Plan

## Steps

1. **Alpha step**
   - do a

2. **Bravo step**
   - do b

3. **Charlie step**
   - do c
"""


async def test_execute_plan_halts_on_sequential_failure():
    from spark_code.plan_executor import execute_plan

    console = Console(file=io.StringIO(), force_terminal=True)
    agent = _FakeAgent(fail_marker="Bravo step")
    team = _FakeTeam()

    await execute_plan(_SEQUENTIAL_PLAN, team, agent, console)

    # Alpha + Bravo ran; Bravo failed → Charlie must be skipped (halted).
    assert len(agent.calls) == 2
    assert not any("Charlie step" in c for c in agent.calls)


async def test_execute_plan_runs_all_when_no_failure():
    from spark_code.plan_executor import execute_plan

    console = Console(file=io.StringIO(), force_terminal=True)
    agent = _FakeAgent(fail_marker="__never__")
    team = _FakeTeam()

    await execute_plan(_SEQUENTIAL_PLAN, team, agent, console)
    assert len(agent.calls) == 3


# ---------------------------------------------------------------------------
# projectplan.py — bugs 9 & 10
# ---------------------------------------------------------------------------


class TestRagServiceUrl:
    def test_default_is_machine_agnostic(self, monkeypatch):
        monkeypatch.delenv("RAG_SERVICE_URL", raising=False)
        from spark_code.projectplan import get_rag_service_url
        url = get_rag_service_url()
        assert "spark-4a54" not in url  # no personal machine hostname
        assert "localhost" in url

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RAG_SERVICE_URL", "http://myhost:9000")
        from spark_code.projectplan import get_rag_service_url
        assert get_rag_service_url() == "http://myhost:9000"


class TestFetchRagContextAsync:
    def test_is_coroutine_function(self):
        from spark_code.projectplan import fetch_rag_context
        assert inspect.iscoroutinefunction(fetch_rag_context)

    async def test_empty_keywords_returns_empty_without_network(self):
        from spark_code.projectplan import fetch_rag_context
        result = await fetch_rag_context([], "Swift project")
        assert result == ""
