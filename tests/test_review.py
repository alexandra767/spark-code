"""Tests for the /review multi-agent swarm (Phase 2 Task 7).

The swarm composes Task 4's ``run_subagent`` core: three reviewer lenses run
concurrently, each returns ``SEVERITY|file:line|description`` findings, and a
skeptic pass tries to REFUTE each surviving finding. These tests drive the
orchestration with a mocked ``run_subagent`` so they run offline and fast — the
real engine is exercised in the live verification step.
"""

import subprocess

from spark_code import review

# ---------------------------------------------------------------------------
# Mock run_subagent
# ---------------------------------------------------------------------------


class RecordingSubagent:
    """Stands in for dispatch.run_subagent.

    Distinguishes a lens call from a skeptic call by the word "refute" (only
    the skeptic prompt contains it). Replies are either a fixed string or a
    callable(prompt) -> str for per-call control.
    """

    def __init__(self, lens_reply="NONE", skeptic_reply="CONFIRMED: real"):
        self.lens_calls = []
        self.skeptic_calls = []
        self.agent_types = []
        self._lens_reply = lens_reply
        self._skeptic_reply = skeptic_reply

    async def __call__(self, model, prompt, agent_type, config, lead_mode):
        self.agent_types.append(agent_type)
        if "refute" in prompt.lower():
            self.skeptic_calls.append(prompt)
            r = self._skeptic_reply
            return r(prompt) if callable(r) else r
        self.lens_calls.append(prompt)
        r = self._lens_reply
        return r(prompt) if callable(r) else r


def _cfg():
    return {
        "model": {"context_window": 32768},
        "agents": {"max_concurrent": 3, "max_rounds": 15},
    }


# ---------------------------------------------------------------------------
# Lens fan-out
# ---------------------------------------------------------------------------


async def test_three_lenses_called_once_each_with_diff(monkeypatch):
    rec = RecordingSubagent(lens_reply="NONE")
    monkeypatch.setattr(review, "run_subagent", rec)
    diff = "diff --git a/foo.py b/foo.py\n@@ -1 +1 @@\n+bad = code"
    out = await review.review_diff(None, _cfg(), "auto", diff)
    assert len(rec.lens_calls) == 3
    for p in rec.lens_calls:
        assert diff in p            # each lens sees the whole diff
    assert all(t == "reviewer" for t in rec.agent_types)
    assert out.dispatched is True
    # No findings surfaced → no skeptic dispatched.
    assert rec.skeptic_calls == []
    assert out.confirmed == []


async def test_conventions_lens_gets_instructions(monkeypatch):
    rec = RecordingSubagent(lens_reply="NONE")
    monkeypatch.setattr(review, "run_subagent", rec)
    await review.review_diff(
        None, _cfg(), "auto", "diff --git a/x b/x\n+x",
        instructions_text="ALWAYS_SNAKE_CASE_RULE")
    hits = [p for p in rec.lens_calls if "ALWAYS_SNAKE_CASE_RULE" in p]
    assert len(hits) == 1  # exactly the conventions lens carries the rules


# ---------------------------------------------------------------------------
# Tolerant parser
# ---------------------------------------------------------------------------


def test_parse_structured_finding():
    findings = review.parse_findings(
        "CRITICAL|foo.py:12|off-by-one in loop bound", "correctness")
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "CRITICAL"
    assert f.location == "foo.py:12"
    assert "off-by-one" in f.description
    assert f.lens == "correctness"
    assert f.raw is False


def test_parse_tolerant_keeps_unparseable_as_raw_note():
    text = (
        "WARNING|bar.py:3|missing null check\n"
        "this line has no pipes but still describes a real bug\n"
        "NONE"  # sentinel for 'no issues' — must be filtered out
    )
    findings = review.parse_findings(text, "edge")
    sev = [f.severity for f in findings]
    assert "WARNING" in sev
    raws = [f for f in findings if f.raw]
    assert len(raws) == 1
    assert "no pipes" in raws[0].description
    assert all(f.description != "NONE" for f in findings)


def test_parse_strips_bullets_and_normalizes_severity():
    findings = review.parse_findings(
        "- warn | baz.py:9 | fragile cast", "edge")
    assert len(findings) == 1
    assert findings[0].severity == "WARNING"
    assert findings[0].location == "baz.py:9"


def test_parse_recovers_merged_location():
    # Real 80B output shape: SEVERITY|file:line — description (only one pipe).
    findings = review.parse_findings(
        "WARNING|calc.py:4 — loop skips the last element", "correctness")
    assert len(findings) == 1
    assert findings[0].location == "calc.py:4"
    assert findings[0].description == "loop skips the last element"


# ---------------------------------------------------------------------------
# Empty diff short-circuit
# ---------------------------------------------------------------------------


async def test_empty_diff_short_circuits(monkeypatch):
    rec = RecordingSubagent()
    monkeypatch.setattr(review, "run_subagent", rec)
    out = await review.review_diff(None, _cfg(), "auto", "")
    assert out.dispatched is False
    assert rec.lens_calls == []
    assert rec.skeptic_calls == []
    assert out.confirmed == []


async def test_whitespace_diff_short_circuits(monkeypatch):
    rec = RecordingSubagent()
    monkeypatch.setattr(review, "run_subagent", rec)
    out = await review.review_diff(None, _cfg(), "auto", "   \n\t\n")
    assert out.dispatched is False
    assert rec.lens_calls == []


# ---------------------------------------------------------------------------
# Skeptic pass
# ---------------------------------------------------------------------------


async def test_skeptic_refuted_excluded(monkeypatch):
    lens = "CRITICAL|foo.py:1|bug A\nWARNING|bar.py:2|bug B"

    def skeptic(prompt):
        # Key on the finding's unique description, not its file — the fallback
        # full-diff context can mention any file for either skeptic.
        return "REFUTED: not actually a bug" if "bug A" in prompt else "CONFIRMED: yes"

    rec = RecordingSubagent(lens_reply=lens, skeptic_reply=skeptic)
    monkeypatch.setattr(review, "run_subagent", rec)
    out = await review.review_diff(
        None, _cfg(), "auto", "diff --git a/foo.py b/foo.py\n+x")
    # 3 lenses each return the same two findings → dedup to 2 → 2 skeptics.
    assert len(rec.skeptic_calls) == 2
    assert len(out.confirmed) == 1
    assert out.confirmed[0].location == "bar.py:2"
    assert len(out.refuted) == 1
    assert out.refuted[0].location == "foo.py:1"


async def test_skeptic_gets_code_context(monkeypatch):
    """The skeptic prompt must carry the finding AND the relevant diff hunk."""
    captured = {}

    def skeptic(prompt):
        captured["prompt"] = prompt
        return "CONFIRMED: yes"

    rec = RecordingSubagent(
        lens_reply="CRITICAL|foo.py:2|off-by-one", skeptic_reply=skeptic)
    monkeypatch.setattr(review, "run_subagent", rec)
    diff = ("diff --git a/foo.py b/foo.py\n@@ -1,3 +1,3 @@\n"
            "-    for i in range(n - 1):\n+    for i in range(n + 1):")
    await review.review_diff(None, _cfg(), "auto", diff)
    p = captured["prompt"]
    assert "off-by-one" in p            # the finding
    assert "range(n + 1)" in p          # the relevant hunk


async def test_lens_crash_does_not_sink_review(monkeypatch):
    def lens(prompt):
        if "correctness" in prompt.lower():
            return "[dispatch_agent error] boom in the engine"
        return "WARNING|z.py:1|real issue"

    rec = RecordingSubagent(lens_reply=lens, skeptic_reply="CONFIRMED: yes")
    monkeypatch.setattr(review, "run_subagent", rec)
    out = await review.review_diff(
        None, _cfg(), "auto", "diff --git a/z.py b/z.py\n+z")
    # The crashed lens contributes nothing; the others still produce findings.
    assert any(f.location == "z.py:1" for f in out.confirmed)


# ---------------------------------------------------------------------------
# Skeptic cap (a lens dumping 100 findings must not spawn 100 skeptics)
# ---------------------------------------------------------------------------


async def test_skeptic_pass_capped_at_12(monkeypatch):
    big = "\n".join(f"WARNING|f{i}.py:{i}|issue number {i}" for i in range(100))
    rec = RecordingSubagent(lens_reply=big, skeptic_reply="CONFIRMED: yes")
    monkeypatch.setattr(review, "run_subagent", rec)
    out = await review.review_diff(
        None, _cfg(), "auto", "diff --git a/x b/x\n+x")
    assert review.SKEPTIC_CAP == 12
    assert len(rec.skeptic_calls) == 12       # capped, not 100
    assert len(out.confirmed) == 12
    assert len(out.overflow) == 88            # the rest noted, not verified


# ---------------------------------------------------------------------------
# Auto-review gating
# ---------------------------------------------------------------------------


def test_should_auto_review_gating():
    assert review.should_auto_review(_cfg(), wrote=True) is False        # default off
    assert review.should_auto_review({"review": {"auto": True}}, wrote=True) is True
    assert review.should_auto_review({"review": {"auto": True}}, wrote=False) is False
    assert review.should_auto_review({"review": {"auto": False}}, wrote=True) is False


async def test_maybe_auto_review_short_circuits_when_disabled(monkeypatch):
    rec = RecordingSubagent()
    monkeypatch.setattr(review, "run_subagent", rec)
    from rich.console import Console
    # auto off → no dispatch even though a write happened
    out = await review.maybe_auto_review(
        None, _cfg(), Console(quiet=True), "auto", wrote_this_turn=True)
    assert out is None
    assert rec.lens_calls == []
    # auto on but nothing written → still no dispatch
    out = await review.maybe_auto_review(
        None, {"review": {"auto": True}}, Console(quiet=True), "auto",
        wrote_this_turn=False)
    assert out is None
    assert rec.lens_calls == []


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


def test_parse_verdict():
    assert review._parse_verdict("REFUTED: the index is correct") == "REFUTED"
    assert review._parse_verdict("CONFIRMED: real off-by-one") == "CONFIRMED"
    assert review._parse_verdict("The finding is REFUTED.") == "REFUTED"
    # Ambiguous / errored skeptic → bias toward surfacing (CONFIRMED).
    assert review._parse_verdict("") == "CONFIRMED"
    assert review._parse_verdict("[dispatch_agent error] boom") == "CONFIRMED"


# ---------------------------------------------------------------------------
# Diff collection
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _init_repo(path):
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t.test")
    _git(path, "config", "user.name", "Test")


def test_collect_diff_non_git_friendly_error(tmp_path):
    res = review.collect_diff(cwd=str(tmp_path))
    assert res.error is not None
    assert "git" in res.error.lower()
    assert res.text == ""


def test_collect_diff_working_tree(tmp_path):
    repo = str(tmp_path)
    _init_repo(repo)
    f = tmp_path / "foo.py"
    f.write_text("def f(n):\n    return n\n")
    _git(repo, "add", "foo.py")
    _git(repo, "commit", "-q", "-m", "init")
    # Modify in the working tree (uncommitted).
    f.write_text("def f(n):\n    return n + 1\n")
    res = review.collect_diff(cwd=repo)
    assert res.error is None
    assert res.text
    assert "foo.py" in res.text
    assert "+1" in res.text or "n + 1" in res.text


def test_collect_diff_falls_back_to_last_commit(tmp_path):
    repo = str(tmp_path)
    _init_repo(repo)
    f = tmp_path / "foo.py"
    f.write_text("x = 1\n")
    _git(repo, "add", "foo.py")
    _git(repo, "commit", "-q", "-m", "one")
    f.write_text("x = 2\n")
    _git(repo, "add", "foo.py")
    _git(repo, "commit", "-q", "-m", "two")
    # Working tree is clean → collect_diff falls back to HEAD~1..HEAD.
    res = review.collect_diff(cwd=repo)
    assert res.error is None
    assert res.text
    assert "foo.py" in res.text


def test_collect_diff_caps_large_diff(tmp_path):
    repo = str(tmp_path)
    _init_repo(repo)
    f = tmp_path / "big.txt"
    f.write_text("seed\n")
    _git(repo, "add", "big.txt")
    _git(repo, "commit", "-q", "-m", "seed")
    # Rewrite with a diff far larger than the 20K cap.
    f.write_text("\n".join(f"line {i}" for i in range(6000)))
    res = review.collect_diff(cwd=repo)
    assert res.error is None
    assert len(res.text) <= review.MAX_DIFF_CHARS + 200  # + truncation marker
    assert "truncated" in res.text


def test_collect_diff_explicit_path_target(tmp_path):
    repo = str(tmp_path)
    _init_repo(repo)
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    (tmp_path / "a.py").write_text("a = 99\n")
    (tmp_path / "b.py").write_text("b = 99\n")
    res = review.collect_diff(cwd=repo, target="a.py")
    assert res.error is None
    assert "a.py" in res.text
    assert "b.py" not in res.text  # target scoping worked


# ---------------------------------------------------------------------------
# Rendering smoke
# ---------------------------------------------------------------------------


def test_render_review_groups_by_severity():
    from rich.console import Console
    out = review.ReviewOutcome(
        dispatched=True,
        confirmed=[
            review.Finding("CRITICAL", "a.py:1", "crash on None", "correctness",
                           verdict="CONFIRMED"),
            review.Finding("SUGGESTION", "b.py:2", "rename var", "conventions",
                           verdict="CONFIRMED"),
        ],
        refuted=[review.Finding("WARNING", "c.py:3", "nope", "edge",
                                verdict="REFUTED")],
        overflow=[review.Finding("WARNING", "d.py:4", "unchecked", "edge")],
        label="working tree vs HEAD",
    )
    console = Console(record=True, width=100)
    review.render_review(console, out)
    text = console.export_text()
    assert "CRITICAL" in text
    assert "crash on None" in text
    assert "1 finding" in text or "refuted" in text.lower()
    assert "not verified" in text.lower() or "cap" in text.lower()
