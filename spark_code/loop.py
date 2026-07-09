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
        apply_result = subprocess.run(["git", "stash", "apply", "--index"],
                                      capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "failed"
    if apply_result.returncode != 0:
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
    if diff.returncode != 0 or untracked.returncode != 0:
        return ""
    blob = (diff.stdout or "") + "\n" + (untracked.stdout or "")
    return hashlib.sha1(blob.encode()).hexdigest()


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


def build_loop_sentinel(args: str) -> str | None:
    """Slash-handler helper: raw /loop args -> __LOOP__ sentinel.
    None means "print usage" (no args given)."""
    text = (args or "").strip()
    if not text:
        return None
    return f"__LOOP__{text}"
