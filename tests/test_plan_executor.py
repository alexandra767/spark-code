"""Tests for plan_executor worker-name uniqueness (audit finding 7).

Two parallel steps whose titles share the same first 15 sanitized chars used
to produce the SAME worker name; team.spawn's duplicate-running guard then
returned None for the second and the step was silently dropped from tracking
and execution. These tests pin the dedup + the loud "could not be spawned"
surfacing.
"""

import io
from types import SimpleNamespace

from rich.console import Console

from spark_code.plan_executor import _make_worker_name, execute_plan


def test_make_worker_name_dedupes_colliding_prefixes():
    used: set[str] = set()
    a = _make_worker_name("Implement the authentication module alpha", 1, used=used)
    b = _make_worker_name("Implement the authentication module bravo", 2, used=used)
    assert a != b                     # no silent collision
    assert a == "implement-the-a"     # first keeps the 15-char base
    assert b.startswith("implement")  # second gets a numeric suffix
    assert len(b) <= 15               # ...that stays within the name budget


def test_make_worker_name_third_collision_bumps_again():
    used: set[str] = set()
    names = [_make_worker_name("Same Prefix Title Overflowing", i, used=used)
             for i in range(1, 4)]
    assert len(set(names)) == 3       # all three distinct


def test_make_worker_name_without_used_is_plain_truncation():
    # Back-compat: no `used` set → deterministic 15-char base (old behavior).
    first = _make_worker_name("Some Long Title That Overflows Fifteen", 1)
    second = _make_worker_name("Some Long Title That Overflows Fifteen", 1)
    assert first == second
    assert len(first) <= 15


class _FakeContext:
    def __init__(self):
        self.messages = []

    def add_user(self, c):
        self.messages.append(("user", c))

    def add_assistant(self, c):
        self.messages.append(("assistant", c))


class _FakeAgent:
    def __init__(self):
        self.context = _FakeContext()

    async def run(self, task_desc):
        return "ok"


class _FakeSpawnTeam:
    """Records spawn names; mimics team.spawn's duplicate-running guard and an
    optional set of names whose spawn is forced to fail (returns None)."""

    max_workers = 4

    def __init__(self, reject_names=None):
        self.spawned = []
        self._running = {}
        self.reject = set(reject_names or ())

    @property
    def active_count(self):
        return sum(1 for w in self._running.values() if w.status == "running")

    async def spawn(self, task_desc, name=""):
        if name in self.reject:
            return None
        if name in self._running and self._running[name].status == "running":
            return None  # duplicate-running guard, like the real TeamManager
        w = SimpleNamespace(name=name, status="completed", result="done")
        self.spawned.append(name)
        self._running[name] = w
        return w


_PARALLEL_COLLIDE_PLAN = """# Plan

## Steps

1. **Implement the authentication module alpha**
   - do a

2. **Implement the authentication module bravo**
   - do b

## Parallelization
Parallel: 1, 2
"""


async def test_colliding_parallel_steps_get_unique_workers():
    console = Console(file=io.StringIO(), force_terminal=True, width=200)
    team = _FakeSpawnTeam()
    await execute_plan(_PARALLEL_COLLIDE_PLAN, team, _FakeAgent(), console)
    assert len(team.spawned) == 2       # neither step silently dropped
    assert len(set(team.spawned)) == 2  # ...and the two names are distinct


_SINGLE_PARALLEL_PLAN = """# Plan

## Steps

1. **Build the thing**
   - do it

## Parallelization
Parallel: 1
"""


async def test_spawn_failure_is_reported_not_silently_dropped():
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, width=200)
    team = _FakeSpawnTeam(reject_names={"build-the-thing"})
    await execute_plan(_SINGLE_PARALLEL_PLAN, team, _FakeAgent(), console)
    assert "could not be spawned" in out.getvalue()
