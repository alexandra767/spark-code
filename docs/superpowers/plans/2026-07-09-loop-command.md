# /loop Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/loop` slash command that autonomously works in rounds — until a goal is verifiably done (until-done flavor) or repeatedly on a timer (timed flavor) — with a git checkpoint before every round, an audit log, and hard stop conditions.

**Architecture:** New module `spark_code/loop.py` holds all loop logic (arg parsing, subprocess helpers, `LoopEngine` with an injectable async runner so tests never touch a model). `cli.py` wires a `/loop` handler that returns a `__LOOP__` sentinel (the `/watch` pattern), and the REPL consumes it: swaps in agentic prompt + trust for the duration (the `/yolo` recipe, restored in `finally`), wraps the engine in an `EscWatcher`, and runs it in the foreground.

**Tech Stack:** Python 3.12+, asyncio, subprocess (git stash / check commands), pytest with pytest-asyncio (existing conventions), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-09-loop-command-design.md`

## Global Constraints

- No test may invoke a real model (existing suite convention) — `LoopEngine` takes `runner: Callable[[str], Awaitable[str]]`, tests inject fakes.
- Checkpoint labels must contain the substring `spark-checkpoint` so `/rollback` lists loop checkpoints (cli.py `_get_checkpoints` filters on that substring).
- Checkpoint recipe is stash-push-then-apply (cli.py:2459-2479): `git stash push -m <label>` then `git stash apply --index` so the working tree is unchanged and the stash entry remains.
- Interval bounds: minimum 60s, maximum 24h.
- Until-done round cap default: 20. Per-round timeout default: 15 minutes (900s). No-progress guard: 2 consecutive rounds with no file changes AND failing verify → STALLED.
- With `--check`, the check command's exit code decides done-ness — never the model's own claim.
- Permission mode and system prompt MUST be restored when the loop ends for any reason (`finally`).
- Rich console colors: use the Nord palette hex codes already used in cli.py/watcher.py (`#88c0d0` info, `#a3be8c` green, `#bf616a` red, `#ebcb8b` yellow, `#8899aa`/`#4c566a` dim).
- One active loop per session; loop logs go to `.spark/loops/` (git-ignored).

---

### Task 1: `loop.py` — interval + argument parsing and dataclasses

**Files:**
- Create: `spark_code/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Produces: `parse_interval(text: str) -> int` (seconds; raises `ValueError`), `parse_loop_args(args: str) -> LoopConfig` (raises `ValueError`), `LoopConfig` dataclass (`goal: str`, `check_cmd: str | None`, `interval_s: int | None`, `max_rounds: int = 20`, `round_timeout_s: int = 900`), `LoopState` dataclass (`round_n: int = 0`, `status: str = "idle"`, `last_verify: str = ""`, `next_run_at: float | None = None`, `stop_requested: bool = False`, `log_path: str = ""`). Tasks 3–4 rely on these exact names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_loop.py
"""Tests for the /loop engine (spark_code/loop.py)."""

import pytest

from spark_code.loop import LoopConfig, parse_interval, parse_loop_args


class TestParseInterval:
    def test_minutes(self):
        assert parse_interval("30m") == 1800

    def test_hours(self):
        assert parse_interval("2h") == 7200

    def test_seconds(self):
        assert parse_interval("90s") == 90

    def test_case_and_whitespace(self):
        assert parse_interval(" 30M ") == 1800

    def test_below_minimum_rejected(self):
        with pytest.raises(ValueError):
            parse_interval("30s")

    def test_above_maximum_rejected(self):
        with pytest.raises(ValueError):
            parse_interval("25h")

    def test_garbage_rejected(self):
        for bad in ("", "fast", "10", "5x", "m30"):
            with pytest.raises(ValueError):
                parse_interval(bad)


class TestParseLoopArgs:
    def test_until_done_plain(self):
        cfg = parse_loop_args("fix the failing tests")
        assert cfg.goal == "fix the failing tests"
        assert cfg.check_cmd is None
        assert cfg.interval_s is None

    def test_until_done_with_check(self):
        cfg = parse_loop_args('--check "pytest -q" fix the failing tests')
        assert cfg.check_cmd == "pytest -q"
        assert cfg.goal == "fix the failing tests"
        assert cfg.interval_s is None

    def test_timed(self):
        cfg = parse_loop_args("every 30m keep the build green")
        assert cfg.interval_s == 1800
        assert cfg.goal == "keep the build green"

    def test_timed_with_check(self):
        cfg = parse_loop_args('every 1h --check "make build" keep the build green')
        assert cfg.interval_s == 3600
        assert cfg.check_cmd == "make build"
        assert cfg.goal == "keep the build green"

    def test_check_single_quotes(self):
        cfg = parse_loop_args("--check 'pytest -q' fix tests")
        assert cfg.check_cmd == "pytest -q"

    def test_check_bare_word(self):
        cfg = parse_loop_args("--check pytest fix tests")
        assert cfg.check_cmd == "pytest"

    def test_empty_goal_rejected(self):
        with pytest.raises(ValueError):
            parse_loop_args('--check "pytest"')

    def test_bad_interval_rejected(self):
        with pytest.raises(ValueError):
            parse_loop_args("every 5x do things")

    def test_defaults(self):
        cfg = parse_loop_args("do the thing")
        assert cfg.max_rounds == 20
        assert cfg.round_timeout_s == 900
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && python -m pytest tests/test_loop.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'spark_code.loop'` (or ImportError).

- [ ] **Step 3: Write the module**

```python
# spark_code/loop.py
"""/loop engine — until-done and timed autonomous loops.

Usage:
    /loop <goal>                        rounds until verified done or stopped
    /loop --check "<cmd>" <goal>        done only when <cmd> exits 0
    /loop every 30m <task>              run task, sleep, repeat until stopped
    /loop status | /loop stop
"""

import asyncio
import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

# Interval bounds (spec: min 60s, max 24h)
_MIN_INTERVAL_S = 60
_MAX_INTERVAL_S = 24 * 3600

_INTERVAL_RE = re.compile(r"^(\d+(?:\.\d+)?)([smh])$")

# Self-assessment markers used when no --check command is given. The round
# prompt asks the model to end with one of these; the engine only greps for
# _DONE_MARKER (absence means "continue").
_DONE_MARKER = "LOOP_DONE"
_CONTINUE_MARKER = "LOOP_CONTINUE"


def parse_interval(text: str) -> int:
    """'30m' -> 1800 seconds. Raises ValueError with a friendly message."""
    m = _INTERVAL_RE.match(text.strip().lower())
    if not m:
        raise ValueError(f"Bad interval {text!r} — use forms like 90s, 30m, 2h")
    value, unit = float(m.group(1)), m.group(2)
    seconds = int(value * {"s": 1, "m": 60, "h": 3600}[unit])
    if seconds < _MIN_INTERVAL_S:
        raise ValueError(f"Interval {text!r} too short — minimum is 60s")
    if seconds > _MAX_INTERVAL_S:
        raise ValueError(f"Interval {text!r} too long — maximum is 24h")
    return seconds


@dataclass
class LoopConfig:
    goal: str
    check_cmd: str | None = None
    interval_s: int | None = None  # None = until-done flavor
    max_rounds: int = 20           # until-done round cap; timed flavor ignores
    round_timeout_s: int = 900     # 15 min per-round wall clock


@dataclass
class LoopState:
    round_n: int = 0
    status: str = "idle"  # idle|running|sleeping|done|stalled|stopped|capped
    last_verify: str = ""  # PASS|FAIL|SELF-DONE|SELF-NOT-DONE|""
    next_run_at: float | None = None
    stop_requested: bool = False
    log_path: str = ""


def parse_loop_args(args: str) -> LoopConfig:
    """Parse the argument string of `/loop ...`.

    Grammar: [every <interval>] [--check "<cmd>"] <goal...>
    Raises ValueError on bad interval or missing goal.
    """
    text = args.strip()
    interval_s = None
    if text.startswith("every "):
        rest = text[len("every "):].lstrip()
        head, _, tail = rest.partition(" ")
        interval_s = parse_interval(head)
        text = tail.strip()
    check_cmd = None
    m = re.search(
        r"--check\s+\"([^\"]+)\"|--check\s+'([^']+)'|--check\s+(\S+)", text)
    if m:
        check_cmd = m.group(1) or m.group(2) or m.group(3)
        text = (text[:m.start()] + text[m.end():]).strip()
    if not text:
        raise ValueError("No goal given — usage: /loop [every 30m] [--check \"cmd\"] <goal>")
    return LoopConfig(goal=text, check_cmd=check_cmd, interval_s=interval_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/spark-code && python -m pytest tests/test_loop.py -q`
Expected: all Task 1 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/loop.py tests/test_loop.py && git commit -m "feat(loop): interval + argument parsing and dataclasses for /loop"
```

---

### Task 2: `loop.py` — subprocess helpers (checkpoint, check, progress fingerprint)

**Files:**
- Modify: `spark_code/loop.py` (append)
- Test: `tests/test_loop.py` (append)

**Interfaces:**
- Produces: `make_checkpoint(label: str) -> str` (returns `"created" | "clean-tree" | "failed" | "git-unavailable"`), `run_check(cmd: str, timeout_s: int = 300) -> tuple[bool, str]` (ok, output tail ≤2000 chars), `tree_fingerprint() -> str` (sha1 of `git diff HEAD` + untracked list; `""` when git unavailable). Task 3's engine calls all three as **module-level functions** (`loop.make_checkpoint(...)`) so tests can monkeypatch them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_loop.py`:

```python
import subprocess as _subprocess
from types import SimpleNamespace

from spark_code import loop as loop_mod
from spark_code.loop import make_checkpoint, run_check, tree_fingerprint


def _fake_run_factory(returncode=0, stdout="", stderr="", raises=None):
    """Build a subprocess.run stand-in capturing calls."""
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        if raises is not None:
            raise raises
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    fake_run.calls = calls
    return fake_run


class TestMakeCheckpoint:
    def test_created(self, monkeypatch):
        fake = _fake_run_factory(returncode=0, stdout="Saved working directory")
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        assert make_checkpoint("spark-checkpoint-loop-round-1") == "created"
        # push then apply --index (the /checkpoint recipe)
        assert ["git", "stash", "push", "-m", "spark-checkpoint-loop-round-1"] == list(fake.calls[0][0][0])
        assert ["git", "stash", "apply", "--index"] == list(fake.calls[1][0][0])

    def test_clean_tree(self, monkeypatch):
        fake = _fake_run_factory(returncode=0, stdout="No local changes to save")
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        assert make_checkpoint("x") == "clean-tree"
        assert len(fake.calls) == 1  # no apply when nothing was stashed

    def test_failed(self, monkeypatch):
        fake = _fake_run_factory(returncode=1, stderr="boom")
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        assert make_checkpoint("x") == "failed"

    def test_git_unavailable(self, monkeypatch):
        fake = _fake_run_factory(raises=FileNotFoundError())
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        assert make_checkpoint("x") == "git-unavailable"


class TestRunCheck:
    def test_pass(self, monkeypatch):
        fake = _fake_run_factory(returncode=0, stdout="1 passed")
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        ok, tail = run_check("pytest -q")
        assert ok is True and "1 passed" in tail
        # shell=True so user command strings work as typed
        assert fake.calls[0][1].get("shell") is True

    def test_fail(self, monkeypatch):
        fake = _fake_run_factory(returncode=1, stdout="1 failed")
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        ok, tail = run_check("pytest -q")
        assert ok is False and "1 failed" in tail

    def test_timeout(self, monkeypatch):
        fake = _fake_run_factory(raises=_subprocess.TimeoutExpired(cmd="x", timeout=1))
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        ok, tail = run_check("sleep 999", timeout_s=1)
        assert ok is False and "timed out" in tail

    def test_tail_truncated(self, monkeypatch):
        fake = _fake_run_factory(returncode=0, stdout="x" * 5000)
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        ok, tail = run_check("big")
        assert len(tail) <= 2000


class TestTreeFingerprint:
    def test_changes_change_fingerprint(self, monkeypatch):
        fake_a = _fake_run_factory(returncode=0, stdout="diff --git a/x b/x\n+new line")
        monkeypatch.setattr(loop_mod.subprocess, "run", fake_a)
        fp_a = tree_fingerprint()
        fake_b = _fake_run_factory(returncode=0, stdout="diff --git a/x b/x\n+other")
        monkeypatch.setattr(loop_mod.subprocess, "run", fake_b)
        fp_b = tree_fingerprint()
        assert fp_a and fp_b and fp_a != fp_b

    def test_git_unavailable_empty(self, monkeypatch):
        fake = _fake_run_factory(raises=FileNotFoundError())
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        assert tree_fingerprint() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && python -m pytest tests/test_loop.py -q`
Expected: FAIL — ImportError on `make_checkpoint` etc.

- [ ] **Step 3: Implement the helpers**

Append to `spark_code/loop.py`:

```python
def make_checkpoint(label: str) -> str:
    """Git-stash checkpoint, same recipe as /checkpoint (cli.py):
    stash push to record, immediately apply --index so the working tree is
    untouched while the stash entry remains as the restore point.

    Returns "created" | "clean-tree" | "failed" | "git-unavailable".
    """
    try:
        result = subprocess.run(
            ["git", "stash", "push", "-m", label],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "git-unavailable"
    if result.returncode != 0:
        return "failed"
    if "No local changes" in result.stdout:
        return "clean-tree"
    try:
        subprocess.run(["git", "stash", "apply", "--index"],
                       capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "failed"
    return "created"


def run_check(cmd: str, timeout_s: int = 300) -> tuple[bool, str]:
    """Run a user check command; ok iff exit 0. Returns (ok, output tail)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"check timed out after {timeout_s}s"
    except FileNotFoundError:
        return False, "check command not found"
    tail = ((r.stdout or "") + (r.stderr or ""))[-2000:]
    return r.returncode == 0, tail


def tree_fingerprint() -> str:
    """Content-sensitive snapshot of the working tree, for no-progress
    detection. `git status --porcelain` is NOT enough (a file edited twice
    still shows the same 'M path' line) — hash actual diff content plus the
    untracked-file list.
    """
    try:
        diff = subprocess.run(["git", "diff", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    blob = (diff.stdout or "") + "\n" + (untracked.stdout or "")
    return hashlib.sha1(blob.encode()).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/spark-code && python -m pytest tests/test_loop.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/loop.py tests/test_loop.py && git commit -m "feat(loop): checkpoint/check/fingerprint subprocess helpers"
```

---

### Task 3: `loop.py` — LoopEngine (round orchestration, stop conditions, audit log)

**Files:**
- Modify: `spark_code/loop.py` (append)
- Test: `tests/test_loop.py` (append)

**Interfaces:**
- Consumes: Task 1 dataclasses, Task 2 helpers (as module-level `make_checkpoint` / `run_check` / `tree_fingerprint` so tests monkeypatch `spark_code.loop.<name>`).
- Produces: `class LoopEngine` — `__init__(self, config: LoopConfig, console, runner: Callable[[str], Awaitable[str]], loops_dir: str = ".spark/loops")`, `async run(self) -> str` (returns final status), `request_stop(self) -> None`, `state: LoopState`, `status_line(self) -> str`. Task 4 constructs it with `runner=agent.run`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_loop.py`:

```python
import asyncio

from spark_code.loop import LoopEngine, LoopState


class FakeRunner:
    """Injectable stand-in for Agent.run — scripted responses per round."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "LOOP_CONTINUE"


class _QuietConsole:
    def print(self, *args, **kwargs):
        pass


def _engine(cfg, runner, tmp_path, monkeypatch, checkpoints=None,
            fingerprints=None, checks=None):
    """Engine with all subprocess helpers stubbed out."""
    monkeypatch.setattr(loop_mod, "make_checkpoint",
                        lambda label: (checkpoints or ["created"]).pop(0)
                        if checkpoints else "created")
    fp_iter = iter(fingerprints or [])
    monkeypatch.setattr(loop_mod, "tree_fingerprint",
                        lambda: next(fp_iter, "same"))
    check_iter = iter(checks or [])
    monkeypatch.setattr(loop_mod, "run_check",
                        lambda cmd, timeout_s=300: next(check_iter, (False, "")))
    return LoopEngine(loop_mod.LoopConfig(**cfg), _QuietConsole(), runner,
                      loops_dir=str(tmp_path / "loops"))


class TestLoopEngineUntilDone:
    def test_stops_on_self_done(self, tmp_path, monkeypatch):
        runner = FakeRunner(["did some work LOOP_CONTINUE", "all fixed LOOP_DONE"])
        eng = _engine({"goal": "fix it"}, runner, tmp_path, monkeypatch,
                      fingerprints=["a", "b", "c", "d"])  # changes every round
        status = asyncio.run(eng.run())
        assert status == "done"
        assert eng.state.round_n == 2

    def test_check_gates_done_over_model_claim(self, tmp_path, monkeypatch):
        # Model claims DONE both rounds; check fails then passes — check wins.
        runner = FakeRunner(["LOOP_DONE", "LOOP_DONE"])
        eng = _engine({"goal": "fix it", "check_cmd": "pytest"}, runner,
                      tmp_path, monkeypatch,
                      fingerprints=["a", "b", "c", "d"],
                      checks=[(False, "1 failed"), (True, "ok")])
        status = asyncio.run(eng.run())
        assert status == "done"
        assert eng.state.round_n == 2
        assert eng.state.last_verify == "PASS"

    def test_stalled_after_two_unproductive_rounds(self, tmp_path, monkeypatch):
        runner = FakeRunner(["LOOP_CONTINUE", "LOOP_CONTINUE", "LOOP_CONTINUE"])
        # fingerprint never changes -> no progress; verify keeps failing
        eng = _engine({"goal": "fix it"}, runner, tmp_path, monkeypatch,
                      fingerprints=["same"] * 10)
        status = asyncio.run(eng.run())
        assert status == "stalled"
        assert eng.state.round_n == 2

    def test_round_cap(self, tmp_path, monkeypatch):
        runner = FakeRunner(["LOOP_CONTINUE"] * 5)
        eng = _engine({"goal": "fix it", "max_rounds": 3}, runner, tmp_path,
                      monkeypatch,
                      fingerprints=[str(i) for i in range(20)])  # always progress
        status = asyncio.run(eng.run())
        assert status == "capped"
        assert eng.state.round_n == 3

    def test_round_timeout_counts_as_no_progress(self, tmp_path, monkeypatch):
        async def slow_runner(prompt):
            await asyncio.sleep(5)
            return "LOOP_DONE"

        eng = _engine({"goal": "fix it", "round_timeout_s": 0}, slow_runner,
                      tmp_path, monkeypatch, fingerprints=["same"] * 10)
        status = asyncio.run(eng.run())
        assert status == "stalled"  # two timed-out, changeless rounds

    def test_checkpoint_called_before_every_round(self, tmp_path, monkeypatch):
        labels = []
        monkeypatch.setattr(loop_mod, "tree_fingerprint", lambda: "x")
        monkeypatch.setattr(loop_mod, "run_check", lambda c, timeout_s=300: (False, ""))

        def record_checkpoint(label):
            labels.append(label)
            return "created"

        monkeypatch.setattr(loop_mod, "make_checkpoint", record_checkpoint)
        runner = FakeRunner(["LOOP_CONTINUE", "LOOP_DONE"])
        eng = LoopEngine(loop_mod.LoopConfig(goal="fix"), _QuietConsole(),
                         runner, loops_dir=str(tmp_path / "loops"))
        asyncio.run(eng.run())
        assert len(labels) == eng.state.round_n
        assert all("spark-checkpoint" in lb for lb in labels)

    def test_log_written(self, tmp_path, monkeypatch):
        runner = FakeRunner(["LOOP_DONE"])
        eng = _engine({"goal": "fix the thing"}, runner, tmp_path, monkeypatch,
                      fingerprints=["a", "b"])
        asyncio.run(eng.run())
        text = open(eng.state.log_path).read()
        assert "fix the thing" in text
        assert "Round 1" in text
        assert "ended: done" in text


class TestLoopEngineTimed:
    def test_refires_and_stop_flag_ends_sleep(self, tmp_path, monkeypatch):
        eng_holder = {}

        class StopAfterTwo(FakeRunner):
            async def __call__(self, prompt):
                if len(self.prompts) >= 1:
                    eng_holder["eng"].request_stop()
                return await super().__call__(prompt)

        runner = StopAfterTwo(["LOOP_DONE", "LOOP_DONE"])
        eng = _engine({"goal": "check build", "interval_s": 1}, runner,
                      tmp_path, monkeypatch,
                      fingerprints=[str(i) for i in range(10)])
        eng_holder["eng"] = eng
        status = asyncio.run(eng.run())
        # Round 1 succeeded (timed loops don't exit on success), slept,
        # round 2 ran with stop already requested -> stopped after it.
        assert status == "stopped"
        assert eng.state.round_n == 2

    def test_timed_success_does_not_exit(self, tmp_path, monkeypatch):
        # 3 successful rounds, then stop via flag — proves success ≠ exit.
        class StopOnThird(FakeRunner):
            async def __call__(self, prompt):
                if len(self.prompts) >= 2:
                    self.eng.request_stop()
                return await super().__call__(prompt)

        runner = StopOnThird(["LOOP_DONE"] * 5)
        eng = _engine({"goal": "patrol", "interval_s": 1}, runner, tmp_path,
                      monkeypatch, fingerprints=[str(i) for i in range(20)])
        runner.eng = eng
        status = asyncio.run(eng.run())
        assert eng.state.round_n == 3
        assert status == "stopped"


class TestStatusLine:
    def test_status_line_mentions_round(self, tmp_path, monkeypatch):
        runner = FakeRunner(["LOOP_DONE"])
        eng = _engine({"goal": "fix"}, runner, tmp_path, monkeypatch,
                      fingerprints=["a", "b"])
        asyncio.run(eng.run())
        line = eng.status_line()
        assert "done" in line and "1" in line
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && python -m pytest tests/test_loop.py -q`
Expected: FAIL — ImportError on `LoopEngine`.

- [ ] **Step 3: Implement LoopEngine**

Append to `spark_code/loop.py`:

```python
class LoopEngine:
    """Drives /loop rounds in the foreground. The model runner is injected
    (tests use fakes; cli passes agent.run). All git/check side effects go
    through the module-level helpers so tests can monkeypatch them.
    """

    def __init__(self, config: LoopConfig, console,
                 runner: Callable[[str], Awaitable[str]],
                 loops_dir: str = ".spark/loops"):
        self.config = config
        self.console = console
        self.runner = runner
        self.state = LoopState()
        os.makedirs(loops_dir, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", config.goal.lower())[:40].strip("-") or "loop"
        self.state.log_path = os.path.join(
            loops_dir, f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.md")

    def request_stop(self) -> None:
        """Thread-safe-enough stop signal (bool set; read between awaits)."""
        self.state.stop_requested = True

    def status_line(self) -> str:
        st = self.state
        nxt = ""
        if st.status == "sleeping" and st.next_run_at:
            nxt = f" · next {time.strftime('%H:%M:%S', time.localtime(st.next_run_at))}"
        return f"🔁 loop {st.status} · round {st.round_n}{nxt}"

    def _log(self, text: str) -> None:
        with open(self.state.log_path, "a") as f:
            f.write(text + "\n")

    def _round_prompt(self, verify_tail: str) -> str:
        cfg, st = self.config, self.state
        if st.round_n == 1:
            base = cfg.goal
        else:
            base = (f"Round {st.round_n} of a /loop. Goal: {cfg.goal}\n"
                    f"The goal is not yet complete — continue working on it.")
            if verify_tail:
                base += f"\nLast check output (tail):\n{verify_tail}"
        if cfg.check_cmd:
            base += (f"\nThe goal is verified by `{cfg.check_cmd}` exiting 0; "
                     f"run it yourself to confirm before finishing.")
        else:
            base += (f"\nWhen the goal is fully complete, end your reply with "
                     f"{_DONE_MARKER}. If more work remains, end with "
                     f"{_CONTINUE_MARKER}.")
        return base

    async def run(self) -> str:
        cfg, st = self.config, self.state
        flavor = "timed" if cfg.interval_s else "until-done"
        self._log(f"# /loop — {cfg.goal}\n")
        self._log(f"flavor: {flavor}"
                  + (f" · every {cfg.interval_s}s" if cfg.interval_s else "")
                  + (f" · check: `{cfg.check_cmd}`" if cfg.check_cmd else ""))
        no_progress = 0
        verify_tail = ""
        while not st.stop_requested:
            if cfg.interval_s is None and st.round_n >= cfg.max_rounds:
                st.status = "capped"
                break
            st.round_n += 1
            st.status = "running"
            self.console.print(
                f"[#88c0d0]🔁 Round {st.round_n}"
                + (f"/{cfg.max_rounds}" if cfg.interval_s is None else "")
                + f" — {cfg.goal}[/#88c0d0]")
            cp = make_checkpoint(f"spark-checkpoint-loop-round-{st.round_n}")
            before = tree_fingerprint()
            t0 = time.monotonic()
            timed_out = False
            try:
                response = await asyncio.wait_for(
                    self.runner(self._round_prompt(verify_tail)),
                    timeout=cfg.round_timeout_s)
            except asyncio.TimeoutError:
                response, timed_out = "", True
            after = tree_fingerprint()
            changed = before != after
            if cfg.check_cmd:
                ok, verify_tail = run_check(cfg.check_cmd)
                st.last_verify = "PASS" if ok else "FAIL"
            else:
                ok = _DONE_MARKER in (response or "")
                st.last_verify = "SELF-DONE" if ok else "SELF-NOT-DONE"
                verify_tail = ""
            self._log(f"\n## Round {st.round_n} — {time.strftime('%H:%M:%S')}\n"
                      f"- checkpoint: {cp}\n"
                      f"- files changed: {'yes' if changed else 'no'}\n"
                      f"- verify: {st.last_verify}"
                      + (" (round TIMEOUT)" if timed_out else "") + "\n"
                      f"- took: {int(time.monotonic() - t0)}s")
            # No-progress guard (both flavors): a failing round that changed
            # nothing, twice in a row, means we're spinning — stop.
            if not ok and not changed:
                no_progress += 1
                if no_progress >= 2:
                    st.status = "stalled"
                    break
            else:
                no_progress = 0
            if cfg.interval_s is None:
                if ok:
                    st.status = "done"
                    break
            else:
                # Timed flavor: success does not exit; sleep until next tick,
                # waking every second to honor stop requests.
                st.status = "sleeping"
                st.next_run_at = time.time() + cfg.interval_s
                self.console.print(f"[#4c566a]{self.status_line()}[/#4c566a]")
                for _ in range(cfg.interval_s):
                    if st.stop_requested:
                        break
                    await asyncio.sleep(1)
        if st.stop_requested and st.status != "done":
            st.status = "stopped"
        self._log(f"\n**ended: {st.status} after {st.round_n} round(s)**")
        return st.status
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/spark-code && python -m pytest tests/test_loop.py -q`
Expected: all PASS. If `test_round_timeout_counts_as_no_progress` hangs, the `asyncio.wait_for` wrapper is missing.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/loop.py tests/test_loop.py && git commit -m "feat(loop): LoopEngine — rounds, stop conditions, audit log"
```

---

### Task 4: CLI wiring — slash handler, sentinel consumption, trust save/restore, Esc stop

**Files:**
- Modify: `spark_code/cli.py` (slash handler near `/watch` at ~line 2449; sentinel consumption near `__WATCH__` at ~line 3579; `/help` text near line 1131)
- Modify: `spark_code/ui/input.py` (command menu dict at ~line 44 where `"/auto"` is listed)
- Modify: `.gitignore` (add `.spark/loops/`)
- Test: `tests/test_loop.py` (append sentinel round-trip tests)

**Interfaces:**
- Consumes: `parse_loop_args`, `LoopEngine` (Task 1/3), `EscWatcher` (`spark_code/ui/esc_watcher.py` — context manager, `EscWatcher(on_escape, ...)`), `AGENTIC_PROMPT`/`SYSTEM_PROMPT` swap + `permissions.mode = "trust"` (the `/yolo` recipe, cli.py:1574-1591), `agent.run` (agent.py:623).
- Produces: `/loop` command; sentinel protocol `__LOOP__<raw args>` with `stop`/`status` as raw values (mirrors `__WATCH__`).

- [ ] **Step 1: Write the failing tests (sentinel handler contract)**

Append to `tests/test_loop.py`:

```python
class TestSlashSentinel:
    """The /loop slash handler returns a __LOOP__ sentinel like /watch's
    __WATCH__ (cli.py). We test the pure helper that builds it."""

    def test_loop_args_pass_through(self):
        from spark_code.loop import build_loop_sentinel
        assert build_loop_sentinel("fix the tests") == "__LOOP__fix the tests"

    def test_stop_and_status_pass_through(self):
        from spark_code.loop import build_loop_sentinel
        assert build_loop_sentinel("stop") == "__LOOP__stop"
        assert build_loop_sentinel("status") == "__LOOP__status"

    def test_empty_returns_none_for_usage_help(self):
        from spark_code.loop import build_loop_sentinel
        assert build_loop_sentinel("") is None
        assert build_loop_sentinel("   ") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && python -m pytest tests/test_loop.py::TestSlashSentinel -q`
Expected: FAIL — ImportError on `build_loop_sentinel`.

- [ ] **Step 3: Add the sentinel helper to `loop.py`**

Append to `spark_code/loop.py`:

```python
def build_loop_sentinel(args: str) -> str | None:
    """Slash-handler helper: raw /loop args -> __LOOP__ sentinel.
    None means "print usage" (no args given)."""
    text = (args or "").strip()
    if not text:
        return None
    return f"__LOOP__{text}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/spark-code && python -m pytest tests/test_loop.py -q`
Expected: all PASS.

- [ ] **Step 5: Wire the slash handler in `cli.py`**

Directly after the `/watch` handler block (which ends with `return f"__WATCH__{args.strip()}"` at ~line 2456), add:

```python
    elif command == "/loop":
        from .loop import build_loop_sentinel
        sentinel = build_loop_sentinel(args)
        if sentinel is None:
            console.print("[#ebcb8b]Usage: /loop [every <interval>] [--check \"<cmd>\"] <goal>[/#ebcb8b]")
            console.print("[#8899aa]  /loop fix the failing tests            — rounds until done[/#8899aa]")
            console.print("[#8899aa]  /loop --check \"pytest -q\" fix tests    — done when check passes[/#8899aa]")
            console.print("[#8899aa]  /loop every 30m keep the build green   — repeat on a timer[/#8899aa]")
            console.print("[#8899aa]  /loop status · /loop stop · Esc stops a running loop[/#8899aa]")
            return None
        return sentinel
```

- [ ] **Step 6: Consume the sentinel in the REPL**

In the REPL sentinel chain, directly after the `elif result.startswith("__WATCH__"):` block (~line 3591), add. Note: `last_loop_engine` must be initialized to `None` next to where `file_watcher = None` is initialized before the REPL loop.

```python
                elif result.startswith("__LOOP__"):
                    from .loop import LoopEngine, parse_loop_args
                    from .ui.esc_watcher import EscWatcher
                    loop_arg = result[len("__LOOP__"):]
                    if loop_arg == "stop":
                        console.print("[#8899aa]No loop is running (a running loop owns the prompt — stop it with Esc).[/#8899aa]")
                    elif loop_arg == "status":
                        if last_loop_engine is not None:
                            console.print(f"[#88c0d0]{last_loop_engine.status_line()}[/#88c0d0]")
                            console.print(f"[#8899aa]log: {last_loop_engine.state.log_path}[/#8899aa]")
                        else:
                            console.print("[#8899aa]No loop has run this session.[/#8899aa]")
                    else:
                        try:
                            loop_cfg = parse_loop_args(loop_arg)
                        except ValueError as e:
                            console.print(f"[#bf616a]{e}[/#bf616a]")
                            continue
                        engine = LoopEngine(loop_cfg, console, runner=agent.run)
                        last_loop_engine = engine
                        # /yolo recipe for the loop's duration: agentic prompt
                        # + trusted tools, restored in finally NO MATTER WHAT.
                        prev_prompt = context.system_prompt
                        prev_mode = permissions.mode if permissions else None
                        if SYSTEM_PROMPT in context.system_prompt:
                            context.system_prompt = context.system_prompt.replace(
                                SYSTEM_PROMPT, AGENTIC_PROMPT)
                        if permissions:
                            permissions.mode = "trust"
                        console.print("[bold #ebcb8b]🔁 Loop started[/bold #ebcb8b] [#8899aa]— Esc to stop; checkpoint before every round[/#8899aa]")
                        try:
                            with EscWatcher(on_escape=engine.request_stop):
                                final_status = await engine.run()
                        except KeyboardInterrupt:
                            engine.request_stop()
                            final_status = "stopped"
                        finally:
                            context.system_prompt = prev_prompt
                            if permissions and prev_mode is not None:
                                permissions.mode = prev_mode
                        color = "#a3be8c" if final_status == "done" else "#ebcb8b"
                        console.print(f"[{color}]Loop ended: {final_status} after {engine.state.round_n} round(s)[/{color}]")
                        console.print(f"[#8899aa]log: {engine.state.log_path} · /rollback lists round checkpoints[/#8899aa]")
```

- [ ] **Step 7: Register the command in menus and help**

In `spark_code/ui/input.py`, in the command-description dict (where `"/auto"` lives at ~line 44), add:

```python
    "/loop": "Work in rounds until done, or repeat on a timer",
```

In `cli.py`'s `/help` text (the section listing `/yolo` at ~line 1131), add alongside the watch/yolo lines:

```markdown
- `/loop [every 30m] [--check "cmd"] <goal>` — Autonomous rounds until done (or on a timer); Esc stops
```

In `.gitignore`, add the line:

```
.spark/loops/
```

- [ ] **Step 8: Run the full test suite**

Run: `cd ~/spark-code && python -m pytest -q`
Expected: entire suite PASS (1332 existing + new loop tests). Fix any import or wiring breakage before committing.

- [ ] **Step 9: Manual smoke test**

Run: `cd /tmp && mkdir -p loop-smoke && cd loop-smoke && git init -q && spark`
Then inside Spark Code: `/loop` (expect usage help), `/loop status` (expect "No loop has run"), `/loop every 5x bad` (expect friendly interval error), then `/loop --check "true" say hello` (expect: round 1 runs, check passes, "Loop ended: done after 1 round(s)"). Verify `.spark/loops/*.md` log exists and `/rollback` shows no crash.
Expected: all behaviors as listed; Esc during the round stops gracefully.

- [ ] **Step 10: Commit**

```bash
cd ~/spark-code && git add spark_code/cli.py spark_code/ui/input.py spark_code/loop.py tests/test_loop.py .gitignore && git commit -m "feat(loop): /loop command — until-done + timed autonomous loops with checkpoint safety net"
```

---

### Task 5: Docs

**Files:**
- Modify: `README.md` (feature list — add `/loop` one-liner where `/watch`//`/yolo` are described)

**Interfaces:**
- Consumes: final command surface from Task 4.

- [ ] **Step 1: Add the README line**

Find the feature/commands section of `README.md` (where `/watch` or `/team` appear) and add:

```markdown
- **`/loop`** — autonomous rounds: `/loop --check "pytest -q" fix the failing tests` keeps working until the tests actually pass; `/loop every 30m keep the build green` patrols on a timer. Git checkpoint before every round, full audit log in `.spark/loops/`, Esc to stop.
```

- [ ] **Step 2: Verify suite still green + commit**

Run: `cd ~/spark-code && python -m pytest -q`
Expected: PASS.

```bash
cd ~/spark-code && git add README.md && git commit -m "docs: /loop command in README"
```
