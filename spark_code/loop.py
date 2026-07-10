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


def _response_signals_done(response: str | None) -> bool:
    """True only when the self-assessment DONE marker TERMINATES the reply.

    The round prompt asks the model to *end* its reply with LOOP_DONE. A plain
    substring test wrongly completes the loop when the model merely MENTIONS
    the marker while still working ("I won't write LOOP_DONE until the tests
    pass"). So require it at the very end: the last non-empty line ends with
    LOOP_DONE (optionally trailed by whitespace/punctuation).
    """
    lines = [ln.strip() for ln in (response or "").splitlines() if ln.strip()]
    if not lines:
        return False
    return bool(re.search(rf"{_DONE_MARKER}[\s.!]*$", lines[-1]))


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
    # idle|running|sleeping|done|stalled|stopped|capped|checkpoint-failed
    status: str = "idle"
    last_verify: str = ""  # PASS|FAIL|SELF-DONE|SELF-NOT-DONE|""
    next_run_at: float | None = None
    stop_requested: bool = False
    log_path: str = ""


def parse_loop_args(args: str) -> LoopConfig:
    """Parse the argument string of `/loop ...`.

    Grammar: [every <interval>] [--check "<cmd>"] <goal...>
    Raises ValueError on bad interval or missing goal.

    ``--check`` is recognized ONLY as a LEADING flag (right after an optional
    ``every <interval>``). Scanning for it anywhere would mangle a goal that
    merely mentions it, e.g. ``/loop add a --check "verbose" flag to the CLI``.
    A goal that must contain a literal ``--check`` therefore has to place the
    real flag form first (or drop ``--check`` from the leading position).
    """
    text = args.strip()
    interval_s = None
    if text.startswith("every "):
        rest = text[len("every "):].lstrip()
        head, _, tail = rest.partition(" ")
        interval_s = parse_interval(head)
        text = tail.strip()
    check_cmd = None
    m = re.match(
        r"--check\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))\s*", text)
    if m:
        check_cmd = m.group(1) or m.group(2) or m.group(3)
        text = text[m.end():].strip()
    if not text:
        raise ValueError("No goal given — usage: /loop [every 30m] [--check \"cmd\"] <goal>")
    return LoopConfig(goal=text, check_cmd=check_cmd, interval_s=interval_s)


def make_checkpoint(label: str) -> str:
    """Git-stash checkpoint, same recipe as /checkpoint (cli.py):
    stash push to record, immediately apply --index so the working tree is
    untouched while the stash entry remains as the restore point.

    Returns "created" | "clean-tree" | "failed" | "apply-failed" |
    "git-unavailable". "apply-failed" is the DANGEROUS case: the push
    succeeded but the re-apply did not, so the working tree is now reverted
    and the only copy of the WIP is the stash entry — distinct from "failed"
    (the push itself failed, tree untouched) so the caller can react to it.
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
        # Push succeeded but re-apply failed: the tree is reverted and the WIP
        # lives only in the stash. Flag it distinctly so the loop can stop
        # loudly instead of running the model against the wrong state.
        return "apply-failed"
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
        # Flipped off the first time git proves unavailable: with no git there
        # is no checkpoint safety net AND no fingerprint, so the no-progress
        # stall guard (below) would false-positive on every round.
        git_available = True
        while not st.stop_requested:
            if cfg.interval_s is None and st.round_n >= cfg.max_rounds:
                st.status = "capped"
                break
            st.round_n += 1
            st.status = "running"
            round_started_at = time.time()
            self.console.print(
                f"[#88c0d0]🔁 Round {st.round_n}"
                + (f"/{cfg.max_rounds}" if cfg.interval_s is None else "")
                + f" — {cfg.goal}[/#88c0d0]")
            cp = make_checkpoint(f"spark-checkpoint-loop-round-{st.round_n}")
            # Surface the checkpoint safety-net status ONCE, at loop start, so a
            # non-git dir (or a broken git) doesn't silently promise a restore
            # point that isn't there.
            if st.round_n == 1:
                if cp == "git-unavailable":
                    git_available = False
                    self.console.print(
                        "[#ebcb8b]⚠ git unavailable — running with no checkpoint "
                        "safety net (no per-round restore point); the "
                        "no-progress stall guard is disabled.[/#ebcb8b]")
                    self._log("- note: git unavailable — no checkpoint safety "
                              "net; stall guard disabled")
                elif cp == "failed":
                    self.console.print(
                        "[#ebcb8b]⚠ checkpoint failed — running without a "
                        "restore point this round.[/#ebcb8b]")
            # A stash pushed but not re-applied means the working tree is
            # reverted and the WIP is stranded in the stash. Stop LOUDLY with
            # recovery steps rather than run the model against the wrong state.
            if cp == "apply-failed":
                st.status = "checkpoint-failed"
                self.console.print(
                    "[#bf616a]✗ Checkpoint could not be restored after "
                    "stashing. Your uncommitted work is safe in the git stash, "
                    "but the working tree was reverted — STOPPING the loop to "
                    "avoid running against the wrong state.\n"
                    "  Recover it with:  git stash pop[/#bf616a]")
                self._log(f"\n## Round {st.round_n} — checkpoint apply-failed\n"
                          "- WIP left in the git stash; loop HALTED before "
                          "running the round. Recover with `git stash pop`.")
                break
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
                ok = _response_signals_done(response)
                st.last_verify = "SELF-DONE" if ok else "SELF-NOT-DONE"
                verify_tail = ""
            self._log(f"\n## Round {st.round_n} — {time.strftime('%H:%M:%S')}\n"
                      f"- checkpoint: {cp}\n"
                      f"- files changed: {'yes' if changed else 'no'}\n"
                      f"- verify: {st.last_verify}"
                      + (" (round TIMEOUT)" if timed_out else "") + "\n"
                      f"- took: {int(time.monotonic() - t0)}s")
            # No-progress guard — until-done flavor ONLY, and only when git can
            # actually observe progress. A timed patrol's idle rounds are its
            # HEALTHY state (nothing to do this tick), and with no git `changed`
            # is always False — so in both cases this guard would false-positive
            # into a permanent STALLED.
            if cfg.interval_s is None and git_available:
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
                # Timed flavor: success does not exit. Anchor the NEXT run to
                # when THIS round STARTED, so the period stays a steady
                # interval_s regardless of how long the round took (a 4-min
                # round on a 30-min interval still fires every 30 min, not 34).
                # Wake at most every second so a stop request is honored fast.
                st.status = "sleeping"
                st.next_run_at = round_started_at + cfg.interval_s
                self.console.print(f"[#4c566a]{self.status_line()}[/#4c566a]")
                while not st.stop_requested:
                    remaining = st.next_run_at - time.time()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(1.0, remaining))
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
