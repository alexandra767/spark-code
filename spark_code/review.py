"""/review — a find-verify multi-agent review swarm (Phase 2 Task 7).

The same pattern that built this codebase, now inside it: fan out THREE reviewer
sub-agents over a diff, each with a distinct lens (correctness/bugs, edge cases +
error handling, project conventions), collect their findings, then run a
skeptic pass where each finding is handed to a fresh reviewer whose only job is
to REFUTE it against the actual code. Only findings that survive the skeptic are
shown — the rest are counted as one dim line.

Everything runs on Task 4's ``run_subagent`` core: reviewer sub-agents run in
fresh isolated contexts with read-only tools, and every dispatch (lens AND
skeptic) acquires the SAME session-wide concurrency semaphore inside
``run_subagent`` — so ``asyncio.gather`` here can launch all of them at once and
still never exceed ``agents.max_concurrent`` in flight.

Entry points:
  * ``run_review(...)`` — the ``/review [target]`` command: collect the diff,
    render results; used by both the slash command and auto-review.
  * ``review_diff(...)`` — the pure orchestration over a diff string (tested
    with a mocked ``run_subagent``).
  * ``maybe_auto_review(...)`` — the config-gated post-write-turn hook.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .agent import _truncate_result
from .config import get
from .dispatch import run_subagent
from .instructions import load_instructions

# Diff budget: head+tail cap so a giant refactor doesn't blow the sub-agent's
# context (reuses the agent's head+tail truncation so exit codes / tails stay).
MAX_DIFF_CHARS = 20000

# A single lens can hallucinate a wall of nitpicks; never spawn a skeptic per
# nit. Verify the top-N by severity; the rest are reported as an overflow count.
SKEPTIC_CAP = 12

# Per-file hunk handed to a skeptic (falls back to the whole diff, capped).
_SKEPTIC_CONTEXT_CHARS = 8000

SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "SUGGESTION": 2, "NOTE": 3}
_SEVERITY_COLOR = {
    "CRITICAL": "#bf616a",
    "WARNING": "#ebcb8b",
    "SUGGESTION": "#88c0d0",
    "NOTE": "#8899aa",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    severity: str
    location: str            # "file:line" or "" when the lens omitted it
    description: str
    lens: str                # which lens surfaced it (correctness/edge/conventions)
    raw: bool = False        # True when the line couldn't be parsed structurally
    verdict: str | None = None       # CONFIRMED / REFUTED after the skeptic pass
    skeptic_note: str = ""


@dataclass
class DiffResult:
    text: str = ""
    label: str = ""
    error: str | None = None


@dataclass
class ReviewOutcome:
    dispatched: bool = False
    confirmed: list[Finding] = field(default_factory=list)
    refuted: list[Finding] = field(default_factory=list)
    overflow: list[Finding] = field(default_factory=list)  # not verified (cap)
    label: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Lens definitions
# ---------------------------------------------------------------------------

_FINDINGS_FORMAT = (
    "Report EACH issue as ONE line, in EXACTLY this pipe-delimited format:\n"
    "  SEVERITY|file:line|description\n"
    "where SEVERITY is one of CRITICAL, WARNING, or SUGGESTION:\n"
    "  - CRITICAL: a real bug — wrong output, crash, data loss, security hole.\n"
    "  - WARNING: likely-incorrect or fragile behavior; missing error handling.\n"
    "  - SUGGESTION: clarity/style/minor improvement.\n"
    "Take file:line from the diff's added (`+`) side. One finding per line, no "
    "wrapping. Output ONLY finding lines — no prose, no headings, no markdown "
    "fences. If you genuinely find NO issue in your lens, output the single "
    "word: NONE"
)

# key/title/focus per lens. `conventions` additionally receives the project
# instructions text (CLAUDE.md/SPARK.md) — the plan's carry-forward: sub-agents
# get NO project instructions by default, so the conventions lens must be told.
LENSES = [
    {
        "key": "correctness",
        "title": "Correctness & Bugs",
        "focus": (
            "Focus ONLY on correctness: logic errors, off-by-one and boundary "
            "mistakes, wrong operators/conditions, incorrect return values, "
            "state that is mutated wrongly, concurrency hazards, and anything "
            "that makes the code produce a wrong result or crash."
        ),
    },
    {
        "key": "edge",
        "title": "Edge Cases & Error Handling",
        "focus": (
            "Focus ONLY on edge cases and error handling: empty/None/zero "
            "inputs, unchecked failures, missing try/except or error returns, "
            "resource leaks, unhandled exceptions, and inputs the new code "
            "silently mishandles."
        ),
    },
    {
        "key": "conventions",
        "title": "Project Conventions",
        "focus": (
            "Focus ONLY on whether the change follows this project's own "
            "conventions and patterns (naming, structure, error style, tests, "
            "the rules stated below). Do not repeat generic bug findings."
        ),
    },
]


def _lens_prompt(lens: dict, diff_text: str, instructions_text: str) -> str:
    parts = [
        f"You are reviewing a code diff through the '{lens['title']}' lens.",
        "",
        lens["focus"],
    ]
    if lens["key"] == "conventions" and instructions_text.strip():
        # Truncate sensibly — the plan says pass the instructions INTO the lens
        # prompt (sub-agents don't get them otherwise), capped ~4000 chars.
        rules = instructions_text.strip()[:4000]
        parts += [
            "",
            "PROJECT CONVENTIONS the code is expected to follow:",
            rules,
        ]
    parts += [
        "",
        _FINDINGS_FORMAT,
        "",
        "DIFF UNDER REVIEW:",
        "```diff",
        diff_text,
        "```",
    ]
    return "\n".join(parts)


def _hunk_for_file(diff_text: str, location: str) -> str:
    """Return the ``diff --git`` block for the finding's file, or ``""``."""
    if not location:
        return ""
    fname = location.split(":", 1)[0].strip()
    if not fname:
        return ""
    base = os.path.basename(fname)
    blocks = re.split(r"(?=^diff --git )", diff_text, flags=re.M)
    for block in blocks:
        if not block.strip():
            continue
        header = block.splitlines()[0]
        if fname in header or (base and base in header):
            return block.strip()
    return ""


def _skeptic_prompt(finding: Finding, diff_text: str) -> str:
    context = _hunk_for_file(diff_text, finding.location) or diff_text
    context = _truncate_result(context, _SKEPTIC_CONTEXT_CHARS)
    loc = finding.location or "(no location given)"
    return "\n".join([
        "A prior reviewer flagged the finding below. Your job is to REFUTE it: "
        "read the ACTUAL code (you have read-only tools — open the file at the "
        "path if you need to) and decide whether it is a REAL problem or a "
        "false alarm.",
        "",
        f"FINDING: [{finding.severity}] {loc} — {finding.description}",
        "",
        "RELEVANT DIFF:",
        "```diff",
        context,
        "```",
        "",
        "Answer on the FIRST line with exactly one word:",
        "  CONFIRMED  — it is a genuine problem, or",
        "  REFUTED    — it is not a problem / the reviewer was mistaken.",
        "Then add one short sentence of justification. When in doubt, CONFIRM.",
    ])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NOISE_PREFIXES = (
    "here are", "below are", "the following", "findings:", "review:",
    "summary:", "issues:", "no issues", "note:", "overall",
)
_NOISE_EXACT = {
    "none", "none.", "n/a", "no issues", "no issues found", "no findings",
    "lgtm", "looks good", "nothing", "no problems found",
}


def _norm_severity(token: str) -> str:
    up = token.strip().upper()
    if "CRIT" in up:
        return "CRITICAL"
    if "WARN" in up:
        return "WARNING"
    if any(x in up for x in ("SUGG", "NIT", "MINOR", "INFO", "STYLE", "LOW")):
        return "SUGGESTION"
    return "WARNING"  # default bucket when the token isn't a clear severity


# A leading `path:line` (or `path:line:col`) token followed by a separator —
# models often write `SEVERITY|calc.py:4 — desc` (one pipe) instead of two.
_LEADING_LOC_RE = re.compile(
    r"^([\w./\\-]+:\d+(?::\d+)?)\s*[—–:\-]\s*(.+)$")


def _extract_leading_location(desc: str) -> tuple[str, str]:
    m = _LEADING_LOC_RE.match(desc.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return "", desc.strip()


def _is_severity_token(token: str) -> bool:
    up = token.strip().upper()
    return any(k in up for k in
               ("CRIT", "WARN", "SUGG", "NIT", "MINOR", "INFO", "STYLE", "LOW"))


def _is_noise(line: str) -> bool:
    low = line.strip().lower()
    if not low:
        return True
    if low in _NOISE_EXACT:
        return True
    if line.strip().startswith(("#", "```", "---", "===", "|--", "==")):
        return True
    if low.startswith(_NOISE_PREFIXES):
        return True
    return False


def parse_findings(raw: str, lens: str) -> list[Finding]:
    """Tolerant parser: structured ``SEVERITY|file:line|desc`` lines become
    typed findings; anything else non-noise is kept as a raw NOTE."""
    findings: list[Finding] = []
    for line in (raw or "").splitlines():
        s = _BULLET_RE.sub("", line.strip())
        if not s or _is_noise(s):
            continue
        if "|" in s:
            parts = [p.strip() for p in s.split("|")]
            if len(parts) >= 3:
                findings.append(Finding(
                    severity=_norm_severity(parts[0]),
                    location=parts[1],
                    description=" | ".join(p for p in parts[2:] if p).strip(),
                    lens=lens,
                ))
                continue
            if len(parts) == 2:
                a, b = parts
                if _is_severity_token(a):
                    # `SEVERITY|file:line — desc` — recover a merged location.
                    loc, rest = _extract_leading_location(b)
                    findings.append(Finding(_norm_severity(a), loc, rest, lens))
                    continue
                if ":" in a or "." in a:  # looks like file / file:line
                    findings.append(Finding("WARNING", a, b, lens))
                    continue
            # A single stray pipe with no useful structure → raw note.
            findings.append(Finding("NOTE", "", s, lens, raw=True))
            continue
        # No pipe at all — tolerant: keep as a raw note (plan requirement), but
        # still recover a leading `file:line` so it renders with a location.
        loc, rest = _extract_leading_location(s)
        findings.append(Finding("NOTE", loc, rest, lens, raw=True))
    return findings


def _dedup(findings: list[Finding]) -> list[Finding]:
    """Collapse findings the lenses reported in common (same location + gist),
    keeping the first (and its severity)."""
    seen: set[tuple[str, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.location.strip().lower(), f.description.strip().lower()[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _parse_verdict(text: str) -> str:
    """CONFIRMED unless the skeptic clearly says REFUTED (surfacing bias)."""
    if not text:
        return "CONFIRMED"
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    first = lines[0].upper() if lines else ""
    if "REFUTED" in first:
        return "REFUTED"
    if "CONFIRMED" in first:
        return "CONFIRMED"
    up = text.upper()
    if "REFUTED" in up and "CONFIRMED" not in up:
        return "REFUTED"
    return "CONFIRMED"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def review_diff(model, config: dict, lead_mode: str, diff_text: str,
                      instructions_text: str = "") -> ReviewOutcome:
    """Run the swarm over a diff string and return the verified outcome.

    Empty diff → no dispatches. Lenses and skeptics run concurrently; each
    ``run_subagent`` acquires the shared semaphore, so total concurrency is
    bounded by ``agents.max_concurrent`` regardless of how many we launch.
    """
    if not diff_text or not diff_text.strip():
        return ReviewOutcome(dispatched=False)

    # --- 1. lenses (concurrent) --------------------------------------------
    lens_coros = [
        run_subagent(model, _lens_prompt(lens, diff_text, instructions_text),
                     "reviewer", config, lead_mode)
        for lens in LENSES
    ]
    lens_results = await asyncio.gather(*lens_coros, return_exceptions=True)

    findings: list[Finding] = []
    for lens, res in zip(LENSES, lens_results):
        # run_subagent already catches its own errors and returns a string;
        # the guard here is belt-and-suspenders so one bad lens can't sink it.
        if isinstance(res, BaseException):
            continue
        if not isinstance(res, str) or res.startswith("[dispatch_agent error]"):
            continue
        findings.extend(parse_findings(res, lens["key"]))

    findings = _dedup(findings)
    findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))

    if not findings:
        return ReviewOutcome(dispatched=True)

    # --- 2. skeptic pass (concurrent, capped) ------------------------------
    to_verify = findings[:SKEPTIC_CAP]
    overflow = findings[SKEPTIC_CAP:]

    skeptic_coros = [
        run_subagent(model, _skeptic_prompt(f, diff_text),
                     "reviewer", config, lead_mode)
        for f in to_verify
    ]
    skeptic_results = await asyncio.gather(*skeptic_coros, return_exceptions=True)

    confirmed: list[Finding] = []
    refuted: list[Finding] = []
    for f, res in zip(to_verify, skeptic_results):
        text = res if isinstance(res, str) else ""
        f.verdict = _parse_verdict(text)
        f.skeptic_note = text.strip()
        (confirmed if f.verdict == "CONFIRMED" else refuted).append(f)

    return ReviewOutcome(
        dispatched=True,
        confirmed=confirmed,
        refuted=refuted,
        overflow=overflow,
    )


# ---------------------------------------------------------------------------
# Diff collection
# ---------------------------------------------------------------------------


def _git(cwd: str, *args: str) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", "git executable not found"
    except subprocess.TimeoutExpired:
        return 124, "", "git timed out"
    except OSError as e:
        return 1, "", str(e)


def collect_diff(cwd: str | None = None, target: str = "") -> DiffResult:
    """Collect the diff to review.

    Default: working-tree changes (``git diff HEAD``); if the tree is clean,
    the last commit (``git diff HEAD~1..HEAD``). An explicit ``target`` path
    scopes the diff to that path. Non-git directory → a friendly error, never
    a crash. The result is capped head+tail to ``MAX_DIFF_CHARS``.
    """
    cwd = cwd or os.getcwd()
    path_args = ["--", target] if target and target.strip() else []
    scope = f" for {target}" if path_args else ""

    rc, _, _ = _git(cwd, "rev-parse", "--is-inside-work-tree")
    if rc != 0:
        return DiffResult(error=f"Not a git repository{scope} — nothing to review.")

    # Working-tree changes vs HEAD.
    rc, out, err = _git(cwd, "diff", "HEAD", *path_args)
    label = f"working tree vs HEAD{scope}"
    if rc != 0:
        # No commits yet (unborn HEAD) → fall back to unstaged + staged diffs.
        rc2, out2, _ = _git(cwd, "diff", *path_args)
        rc3, out3, _ = _git(cwd, "diff", "--cached", *path_args)
        out = (out2 or "") + (out3 or "")
        label = f"uncommitted changes{scope}"

    if not out.strip():
        # Clean tree → review the last commit instead.
        rc4, out4, _ = _git(cwd, "diff", "HEAD~1..HEAD", *path_args)
        if rc4 == 0 and out4.strip():
            out = out4
            label = f"last commit (HEAD~1..HEAD){scope}"
        else:
            return DiffResult(text="", label=label)

    return DiffResult(text=_truncate_result(out, MAX_DIFF_CHARS), label=label)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_review(console: Console, outcome: ReviewOutcome) -> None:
    if outcome.error:
        console.print(f"[#8899aa]{outcome.error}[/#8899aa]")
        return
    if not outcome.dispatched:
        console.print("[#8899aa]Nothing to review — no changes vs HEAD.[/#8899aa]")
        return

    title = "Code Review"
    if outcome.label:
        title += f" — {outcome.label}"

    if not outcome.confirmed:
        body = Text("No issues survived the skeptic pass. Clean.",
                    style="#a3be8c")
    else:
        body = Text()
        by_sev: dict[str, list[Finding]] = {}
        for f in outcome.confirmed:
            by_sev.setdefault(f.severity, []).append(f)
        first = True
        for sev in ("CRITICAL", "WARNING", "SUGGESTION", "NOTE"):
            group = by_sev.get(sev)
            if not group:
                continue
            if not first:
                body.append("\n")
            first = False
            color = _SEVERITY_COLOR.get(sev, "#d8dee9")
            body.append(f"{sev} ({len(group)})\n", style=f"bold {color}")
            for f in group:
                loc = f.location or "?"
                body.append("  • ", style=color)
                body.append(f"{loc}", style="#d8dee9")
                body.append(f" — {f.description}", style="#cdd0d6")
                body.append(f"  [{f.lens}]\n", style="#4c566a")

    console.print(Panel(body, title=title, title_align="left",
                        border_style="#88c0d0"))

    if outcome.refuted:
        n = len(outcome.refuted)
        console.print(
            f"[#4c566a]{n} finding{'s' if n != 1 else ''} refuted by the "
            f"skeptic pass and hidden.[/#4c566a]")
    if outcome.overflow:
        n = len(outcome.overflow)
        console.print(
            f"[#4c566a]{n} additional finding{'s' if n != 1 else ''} not "
            f"verified (skeptic cap {SKEPTIC_CAP}).[/#4c566a]")


# ---------------------------------------------------------------------------
# Entry points used by the CLI
# ---------------------------------------------------------------------------


async def run_review(model, config: dict, console: Console, lead_mode: str,
                     target: str = "", cwd: str | None = None,
                     instructions_text: str | None = None) -> ReviewOutcome:
    """The ``/review [target]`` command (and auto-review): collect the diff,
    run the swarm, render the result. Returns the outcome for callers/tests."""
    cwd = cwd or os.getcwd()
    if instructions_text is None:
        try:
            instructions_text = load_instructions(cwd).text
        except Exception:
            instructions_text = ""

    diff = collect_diff(cwd=cwd, target=target)
    if diff.error:
        console.print(f"[#8899aa]{diff.error}[/#8899aa]")
        return ReviewOutcome(dispatched=False, error=diff.error)
    if not diff.text.strip():
        console.print(
            "[#8899aa]Nothing to review — no changes vs HEAD.[/#8899aa]")
        return ReviewOutcome(dispatched=False, label=diff.label)

    console.print(
        f"[#88c0d0]Reviewing {diff.label} with 3 lenses "
        f"(correctness · edge cases · conventions)…[/#88c0d0]")
    outcome = await review_diff(model, config, lead_mode, diff.text,
                                instructions_text or "")
    outcome.label = diff.label
    render_review(console, outcome)
    return outcome


def should_auto_review(config: dict, wrote: bool) -> bool:
    """Auto-review fires only when ``review.auto`` is on AND the turn wrote."""
    return bool(get(config, "review", "auto", default=False)) and bool(wrote)


async def maybe_auto_review(model, config: dict, console: Console,
                            lead_mode: str, wrote_this_turn: bool,
                            cwd: str | None = None,
                            instructions_text: str | None = None
                            ) -> ReviewOutcome | None:
    """Config-gated post-turn hook. No-op (returns None, zero dispatches) unless
    ``review.auto`` is enabled and a write/edit succeeded this turn."""
    if not should_auto_review(config, wrote_this_turn):
        return None
    console.print("[#4c566a]auto-review: changes detected, running "
                  "review swarm…[/#4c566a]")
    return await run_review(model, config, console, lead_mode,
                            cwd=cwd, instructions_text=instructions_text)
