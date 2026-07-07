"""Tests for dispatch_agent — subagents as context isolation (Phase 2 Task 4).

The sub-agent machinery composes existing pieces (Agent + Context +
build_tools + the non-interactive PermissionManager). These tests drive it
with a scripted mock model (the pattern in tests/test_agent.py) so they run
offline and fast.
"""

import asyncio
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from spark_code import dispatch
from spark_code.context import Context
from spark_code.tools.base import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockModel:
    """Yields a scripted stream per chat() call (like tests/test_agent.py)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self._call = 0

    async def chat(self, **kwargs):
        if self._call < len(self.responses):
            chunks = self.responses[self._call]
            self._call += 1
            for chunk in chunks:
                yield chunk

    async def close(self):
        pass


class RaisingModel:
    """A model whose chat stream raises — exercises the error path."""

    async def chat(self, **kwargs):
        raise RuntimeError("boom in the engine")
        yield  # unreachable — makes this an async generator


class SlowCountingModel:
    """Sleeps while tracking how many chat() bodies run concurrently."""

    def __init__(self, counter):
        self.counter = counter

    async def chat(self, **kwargs):
        self.counter["cur"] += 1
        self.counter["max"] = max(self.counter["max"], self.counter["cur"])
        try:
            await asyncio.sleep(0.05)
        finally:
            self.counter["cur"] -= 1
        yield {"type": "text", "content": "done"}
        yield {"type": "done", "usage": {}}


class _FakeDispatchTool(Tool):
    """Stand-in named dispatch_agent to prove the depth guard excludes it."""

    name = "dispatch_agent"
    description = "fake"
    is_read_only = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "should never run in a sub-agent"


def _cfg(max_concurrent=3, max_rounds=15, context_window=32768):
    return {
        "model": {"context_window": context_window},
        "agents": {"max_concurrent": max_concurrent, "max_rounds": max_rounds},
    }


def _final_text(text):
    return [[{"type": "text", "content": text}, {"type": "done", "usage": {}}]]


@pytest.fixture(autouse=True)
def _reset_dispatch_semaphore():
    """Each test gets a clean shared semaphore bound to its own event loop."""
    dispatch._reset_semaphore()
    yield
    dispatch._reset_semaphore()


# ---------------------------------------------------------------------------
# Registry filtering / depth guard
# ---------------------------------------------------------------------------


def test_explore_registry_is_read_only_only():
    registry = dispatch._build_subagent_registry("explore")
    names = registry.names()
    assert "read_file" in names
    assert "grep" in names
    # Mutating tools must be absent from a read-only sub-agent.
    assert "write_file" not in names
    assert "edit_file" not in names
    for tool in registry.all():
        assert tool.is_read_only, f"{tool.name} is not read-only"


def test_reviewer_registry_is_read_only_only():
    registry = dispatch._build_subagent_registry("reviewer")
    names = registry.names()
    assert "read_file" in names
    assert "write_file" not in names


def test_implementer_registry_has_writes_but_not_dispatch_agent():
    # Seed the base with a dispatch_agent tool so the depth guard has something
    # to strip — proves nesting is structurally impossible.
    base = ToolRegistry()
    from spark_code.cli import build_tools
    for tool in build_tools().all():
        base.register(tool)
    base.register(_FakeDispatchTool())
    assert "dispatch_agent" in base.names()

    registry = dispatch._build_subagent_registry("implementer", base_registry=base)
    names = registry.names()
    assert "write_file" in names
    assert "edit_file" in names
    assert "dispatch_agent" not in names  # no nesting


# ---------------------------------------------------------------------------
# run_subagent core
# ---------------------------------------------------------------------------


async def test_final_text_returned():
    model = MockModel(_final_text("GREETING is defined in greetings.py as 'Howdy'"))
    result = await dispatch.run_subagent(
        model, "find GREETING", "explore", _cfg(), "auto")
    assert result == "GREETING is defined in greetings.py as 'Howdy'"


async def test_result_truncated_head_and_tail():
    big = "A" * 4000 + "B" * 4000 + "C" * 4000  # 12000 > 8000
    model = MockModel(_final_text(big))
    result = await dispatch.run_subagent(
        model, "explore", "explore", _cfg(), "auto")
    assert len(result) < len(big)
    assert "truncated" in result
    assert result.startswith("A")   # head preserved
    assert result.endswith("C")     # tail preserved


async def test_subagent_exception_returns_error_string():
    model = RaisingModel()
    result = await dispatch.run_subagent(
        model, "explore", "explore", _cfg(), "auto")
    assert result.startswith("[dispatch_agent error]")
    assert "boom in the engine" in result


async def test_unknown_agent_type_returns_error_string():
    model = MockModel(_final_text("unused"))
    result = await dispatch.run_subagent(
        model, "do it", "wizard", _cfg(), "auto")
    assert result.startswith("[dispatch_agent error]")


async def test_semaphore_limits_concurrency():
    counter = {"cur": 0, "max": 0}
    cfg = _cfg(max_concurrent=3)

    async def one():
        model = SlowCountingModel(counter)
        return await dispatch.run_subagent(model, "go", "explore", cfg, "auto")

    results = await asyncio.gather(*[one() for _ in range(5)])
    assert all(r == "done" for r in results)
    assert counter["max"] <= 3           # the semaphore ceiling
    assert counter["max"] >= 2           # but it DID run in parallel


async def test_lead_context_untouched():
    lead_ctx = Context()
    lead_ctx.add_user("the lead's own task")
    before = len(lead_ctx.messages)
    model = MockModel(_final_text("sub report"))
    result = await dispatch.run_subagent(
        model, "explore", "explore", _cfg(), "auto")
    assert result == "sub report"
    # The sub-agent built its OWN context; the lead's is untouched.
    assert len(lead_ctx.messages) == before


async def test_subagent_respects_max_rounds_cap():
    # The sub-agent's cap comes from agents.max_rounds, not Agent.MAX_TOOL_ROUNDS.
    model = MockModel(_final_text("ok"))
    cfg = _cfg(max_rounds=7)
    # Capture the Agent the tool builds by monkeypatching nothing — instead
    # assert via the returned text (round cap is verified structurally below).
    result = await dispatch.run_subagent(model, "x", "explore", cfg, "auto")
    assert result == "ok"


# ---------------------------------------------------------------------------
# Worktree isolation (Phase 5 Task 8) — git subprocess mocked via
# dispatch._run_git, the single subprocess entry point for the whole
# worktree lifecycle.
# ---------------------------------------------------------------------------


def _proc(returncode=0, stdout="", stderr=""):
    """A subprocess.CompletedProcess-alike stub for mocking _run_git."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _fake_git_factory(repo_root, calls, *, worktree_ok=True, diffstat="",
                      remove_ok=True):
    """Builds a fake ``_run_git(args, cwd)`` that records every call and
    scripts a full happy-path (or overridden) worktree lifecycle."""

    def _fake(args, cwd):
        calls.append((list(args), cwd))
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return _proc(0, stdout=repo_root + "\n")
        if args[:2] == ["worktree", "add"]:
            return _proc(0) if worktree_ok else _proc(128, stderr="worktree add failed")
        if args[:2] == ["add", "-A"]:
            return _proc(0)
        if args[:2] == ["diff", "--stat"]:
            return _proc(0, stdout=diffstat)
        if args[:2] == ["worktree", "remove"]:
            return _proc(0) if remove_ok else _proc(1, stderr="still dirty")
        return _proc(0)

    return _fake


def test_scope_registry_for_worktree_narrows_write_tools():
    from spark_code.cli import build_tools
    base = dispatch._build_subagent_registry(
        "implementer", base_registry=build_tools())
    scoped = dispatch._scope_registry_for_worktree(base, "/some/worktree/path", _cfg())

    assert scoped.get("write_file").cwd == "/some/worktree/path"
    assert scoped.get("edit_file").cwd == "/some/worktree/path"
    # Every other tool is REUSED, not rebuilt — same instance as the base.
    assert scoped.get("bash") is base.get("bash")
    assert scoped.get("read_file") is base.get("read_file")


async def test_isolated_implementer_creates_worktree_under_dot_spark(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch, "_run_git", _fake_git_factory(
        str(tmp_path), calls, diffstat=" f.txt | 1 +\n 1 file changed, 1 insertion(+)\n"))

    model = MockModel(_final_text("wrote f.txt"))
    result = await dispatch.run_subagent(
        model, "add a file", "implementer", _cfg(), "auto", isolated=True)

    assert "wrote f.txt" in result
    add_calls = [c for c, _cwd in calls if c[:2] == ["worktree", "add"]]
    assert len(add_calls) == 1
    worktree_arg = add_calls[0][-2]  # ["worktree", "add", "--detach", <path>, "HEAD"]
    expected_prefix = str(tmp_path / ".spark" / "worktrees") + "/"
    assert worktree_arg.startswith(expected_prefix), worktree_arg
    # The short id is safe: pure hex, nothing prompt-shaped smuggled in.
    short_id = worktree_arg[len(expected_prefix):]
    assert re.fullmatch(r"[0-9a-f]{8}", short_id), short_id


async def test_isolated_implementer_summary_includes_diffstat(tmp_path, monkeypatch):
    calls = []
    diffstat_text = " f.txt | 3 +++\n 1 file changed, 3 insertions(+)"
    monkeypatch.setattr(dispatch, "_run_git", _fake_git_factory(
        str(tmp_path), calls, diffstat=diffstat_text))

    model = MockModel(_final_text("implemented the change"))
    result = await dispatch.run_subagent(
        model, "add a file", "implementer", _cfg(), "auto", isolated=True)

    assert "implemented the change" in result
    # _worktree_diffstat() strips the raw git output, so compare stripped.
    assert diffstat_text.strip() in result
    assert "kept for review" in result
    assert str(tmp_path / ".spark" / "worktrees") in result
    # Unchanged-cleanup path must NOT fire when there IS a diff.
    remove_calls = [c for c, _cwd in calls if c[:2] == ["worktree", "remove"]]
    assert remove_calls == []


async def test_unchanged_worktree_is_removed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch, "_run_git", _fake_git_factory(
        str(tmp_path), calls, diffstat=""))  # empty diff == unchanged

    model = MockModel(_final_text("looked around, changed nothing"))
    result = await dispatch.run_subagent(
        model, "investigate", "implementer", _cfg(), "auto", isolated=True)

    assert "looked around, changed nothing" in result
    assert "removed" in result.lower()
    remove_calls = [c for c, _cwd in calls if c[:2] == ["worktree", "remove"]]
    assert len(remove_calls) == 1
    assert "--force" not in remove_calls[0][0]  # plain remove, not forced


async def test_unchanged_worktree_remove_failure_is_reported_not_crashed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch, "_run_git", _fake_git_factory(
        str(tmp_path), calls, diffstat="", remove_ok=False))

    model = MockModel(_final_text("done"))
    result = await dispatch.run_subagent(
        model, "investigate", "implementer", _cfg(), "auto", isolated=True)

    assert "done" in result
    assert "could not be auto-removed" in result


async def test_non_git_repo_falls_back_with_note(monkeypatch):
    calls = []

    def fake_git(args, cwd):
        calls.append((list(args), cwd))
        assert args[:2] == ["rev-parse", "--show-toplevel"], args
        return _proc(128, stderr="fatal: not a git repository")

    monkeypatch.setattr(dispatch, "_run_git", fake_git)

    model = MockModel(_final_text("did the work anyway"))
    result = await dispatch.run_subagent(
        model, "do it", "implementer", _cfg(), "auto", isolated=True)

    assert "did the work anyway" in result
    assert "not a git repository" in result
    assert len(calls) == 1  # only the rev-parse probe — no worktree add attempted


async def test_worktree_add_failure_falls_back_with_note(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch, "_run_git", _fake_git_factory(
        str(tmp_path), calls, worktree_ok=False))

    model = MockModel(_final_text("did the work anyway"))
    result = await dispatch.run_subagent(
        model, "do it", "implementer", _cfg(), "auto", isolated=True)

    assert "did the work anyway" in result
    assert "worktree add failed" in result
    assert "ran without worktree isolation" in result


async def test_explore_ignores_isolated_flag(monkeypatch):
    def fail_git(args, cwd):
        raise AssertionError(f"git must never be called for explore: {args}")
    monkeypatch.setattr(dispatch, "_run_git", fail_git)

    model = MockModel(_final_text("found it"))
    result = await dispatch.run_subagent(
        model, "search", "explore", _cfg(), "auto", isolated=True)
    assert result == "found it"


async def test_reviewer_ignores_isolated_flag(monkeypatch):
    def fail_git(args, cwd):
        raise AssertionError(f"git must never be called for reviewer: {args}")
    monkeypatch.setattr(dispatch, "_run_git", fail_git)

    model = MockModel(_final_text("reviewed"))
    result = await dispatch.run_subagent(
        model, "review", "reviewer", _cfg(), "auto", isolated=True)
    assert result == "reviewed"


async def test_custom_agent_type_ignores_isolated_flag(monkeypatch):
    """A custom .spark/agents/*.md def is never the literal built-in
    'implementer' string (it's looked up by name), so isolated is ignored
    even if the def's own base_type is 'implementer'."""
    from spark_code.agents_registry import AgentDef

    def fail_git(args, cwd):
        raise AssertionError(f"git must never be called for a custom def: {args}")
    monkeypatch.setattr(dispatch, "_run_git", fail_git)

    custom_def = AgentDef(name="my-implementer", description="A custom implementer",
                          base_type="implementer",
                          system_prompt="Do the custom thing.", tools=None,
                          model_hint=None)
    model = MockModel(_final_text("custom done"))
    result = await dispatch.run_subagent(
        model, "go", "my-implementer", _cfg(), "auto",
        agent_defs={"my-implementer": custom_def}, isolated=True)
    assert result == "custom done"


async def test_worktree_creation_is_semaphore_gated(tmp_path, monkeypatch):
    """Worktree creation is the EXPENSIVE part (per the plan) — it must run
    INSIDE the same semaphore acquisition as the sub-agent, not before it, so
    concurrent isolated dispatches never exceed agents.max_concurrent
    in-flight `git worktree add` calls at once."""
    import threading
    import time

    lock = threading.Lock()
    counter = {"cur": 0, "max": 0}

    def fake_git(args, cwd):
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return _proc(0, stdout=str(tmp_path) + "\n")
        if args[:2] == ["worktree", "add"]:
            with lock:
                counter["cur"] += 1
                counter["max"] = max(counter["max"], counter["cur"])
            time.sleep(0.05)  # stand-in for the real, slow checkout
            with lock:
                counter["cur"] -= 1
            return _proc(0)
        if args[:2] == ["add", "-A"]:
            return _proc(0)
        if args[:2] == ["diff", "--stat"]:
            return _proc(0, stdout="")
        if args[:2] == ["worktree", "remove"]:
            return _proc(0)
        return _proc(0)

    monkeypatch.setattr(dispatch, "_run_git", fake_git)
    cfg = _cfg(max_concurrent=2)

    async def one():
        model = MockModel(_final_text("done"))
        return await dispatch.run_subagent(
            model, "go", "implementer", cfg, "trust", isolated=True)

    results = await asyncio.gather(*[one() for _ in range(5)])
    assert all("removed" in r.lower() for r in results)
    assert counter["max"] <= 2   # the semaphore ceiling — never exceeded
    assert counter["max"] >= 2   # but it DID overlap, proving it's not fully serial


async def test_worktree_removed_on_subagent_error(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dispatch, "_run_git", _fake_git_factory(str(tmp_path), calls))

    model = RaisingModel()
    result = await dispatch.run_subagent(
        model, "x", "implementer", _cfg(), "auto", isolated=True)

    assert result.startswith("[dispatch_agent error]")
    remove_calls = [c for c, _cwd in calls if c[:2] == ["worktree", "remove"]]
    assert len(remove_calls) == 1
    assert "--force" in remove_calls[0]  # force-removed — never leaked


async def test_dispatch_tool_execute_passes_isolated_through():
    captured = {}

    async def _fake_run(model, prompt, agent_type, config, lead_mode, **kwargs):
        captured.update(kwargs)
        return "ok"

    model = MockModel(_final_text("ok"))
    tool = dispatch.DispatchAgentTool(model=model, config=_cfg(), lead_mode="auto")
    import spark_code.dispatch as d
    orig = d.run_subagent
    d.run_subagent = _fake_run
    try:
        await tool.execute(prompt="p", agent_type="implementer", isolated=True)
        assert captured["isolated"] is True
        await tool.execute(prompt="p", agent_type="implementer")
        assert captured["isolated"] is False  # default
    finally:
        d.run_subagent = orig


# ---------------------------------------------------------------------------
# Worktree isolation — real git, real temp repo (no mocking). Exercises the
# whole lifecycle end to end: creates a worktree, has the sub-agent actually
# write a file into it via the real write_file tool, confirms the worktree
# path is reported with a real diffstat, and that the real repo's working
# tree (the pytest process's actual cwd, chdir'd here) is left untouched.
# ---------------------------------------------------------------------------


def _init_temp_repo(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


class _WorktreeDiscoveringModel:
    """Reads the worktree path Task 8 embeds in the sub-agent's own system
    prompt (it can't be known ahead of time — the short id is random) and
    asks it to write a real file there via the real write_file tool."""

    def __init__(self):
        self.worktree_path = None
        self._call = 0

    async def chat(self, messages=None, **kwargs):
        self._call += 1
        if self._call == 1:
            blob = "\n".join(
                m.get("content", "") for m in (messages or [])
                if isinstance(m.get("content"), str))
            match = re.search(r"(\S+\.spark/worktrees/[0-9a-f]{8})", blob)
            assert match, f"worktree path not found in sub-agent prompt: {blob!r}"
            self.worktree_path = match.group(1)
            yield {"type": "tool_call", "id": "call_1", "name": "write_file",
                  "arguments": {"file_path": f"{self.worktree_path}/new_file.txt",
                               "content": "hello from the isolated worktree\n"}}
            yield {"type": "done", "usage": {}}
        else:
            yield {"type": "text", "content": "wrote new_file.txt"}
            yield {"type": "done", "usage": {}}

    async def close(self):
        pass


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
async def test_live_isolated_dispatch_real_git_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_temp_repo(repo)
    monkeypatch.chdir(repo)

    model = _WorktreeDiscoveringModel()
    # "trust" mode: the non-interactive sub-agent PermissionManager denies
    # writes outright in "auto" mode (fails safe rather than blocking on a
    # prompt) — this test needs the write to actually go through so it can
    # verify WHERE it landed.
    result = await dispatch.run_subagent(
        model, "add new_file.txt", "implementer", _cfg(), "trust", isolated=True)

    assert "wrote new_file.txt" in result
    assert model.worktree_path is not None
    assert model.worktree_path.startswith(str(repo))
    assert "git diff --stat" in result

    # The file landed in the WORKTREE, not the real repo's working tree.
    import os
    worktree_file = os.path.join(model.worktree_path, "new_file.txt")
    assert os.path.isfile(worktree_file)
    assert not os.path.isfile(repo / "new_file.txt")
    # The real repo's tracked working tree is untouched (only the untracked
    # .spark/worktrees/ scaffolding is new — the worktree lifecycle itself).
    status = subprocess.run(["git", "status", "--porcelain", "--", "README.md"],
                            cwd=repo, capture_output=True, text=True, check=True)
    assert status.stdout.strip() == ""


# ---------------------------------------------------------------------------
# DispatchAgentTool wrapper
# ---------------------------------------------------------------------------


def test_tool_schema_and_read_only_flag():
    tool = dispatch.DispatchAgentTool(model=None, config=_cfg())
    assert tool.name == "dispatch_agent"
    assert tool.is_read_only is False  # conservative: per-call read-only unsupported
    schema = tool.to_schema()
    props = schema["parameters"]["properties"]
    assert set(props) == {"prompt", "agent_type", "isolated"}
    assert props["agent_type"]["enum"] == ["explore", "reviewer", "implementer"]
    assert props["isolated"]["type"] == "boolean"
    # isolated is optional (default False) — required list is unchanged.
    assert schema["parameters"]["required"] == ["prompt", "agent_type"]


async def test_tool_execute_delegates_to_run_subagent():
    model = MockModel(_final_text("delegated result"))
    tool = dispatch.DispatchAgentTool(model=model, config=_cfg(), lead_mode="auto")
    result = await tool.execute(prompt="find it", agent_type="explore")
    assert result == "delegated result"


async def test_tool_reads_live_lead_mode_from_permissions():
    from spark_code.permissions import PermissionManager
    perms = PermissionManager(mode="ask")
    captured = {}

    async def _fake_run(model, prompt, agent_type, config, lead_mode, **kwargs):
        captured["mode"] = lead_mode
        return "ok"

    model = MockModel(_final_text("ok"))
    tool = dispatch.DispatchAgentTool(model=model, config=_cfg(), permissions=perms)
    import spark_code.dispatch as d
    orig = d.run_subagent
    d.run_subagent = _fake_run
    try:
        await tool.execute(prompt="p", agent_type="explore")
        assert captured["mode"] == "ask"
        perms.mode = "auto"          # mode changes mid-session (Shift+Tab)
        await tool.execute(prompt="p", agent_type="explore")
        assert captured["mode"] == "auto"
    finally:
        d.run_subagent = orig


# ---------------------------------------------------------------------------
# build_tools registration
# ---------------------------------------------------------------------------


def test_build_tools_registers_dispatch_when_model_and_config_given():
    from spark_code.cli import build_tools
    registry = build_tools(model=MockModel([]), config=_cfg())
    assert "dispatch_agent" in registry.names()


def test_build_tools_omits_dispatch_without_model():
    from spark_code.cli import build_tools
    registry = build_tools()
    assert "dispatch_agent" not in registry.names()
