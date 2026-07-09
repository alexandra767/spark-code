"""/loop engine — until-done and timed autonomous loops.

Usage:
    /loop <goal>                        rounds until verified done or stopped
    /loop --check "<cmd>" <goal>        done only when <cmd> exits 0
    /loop every 30m <task>              run task, sleep, repeat until stopped
    /loop status | /loop stop
"""

import hashlib
import re
import subprocess
from dataclasses import dataclass

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
