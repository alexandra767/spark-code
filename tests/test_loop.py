"""Tests for the /loop engine (spark_code/loop.py)."""

import asyncio
import subprocess as _subprocess
import time
from types import SimpleNamespace

import pytest

from spark_code import loop as loop_mod
from spark_code.loop import (
    _response_signals_done,
    make_checkpoint,
    parse_interval,
    parse_loop_args,
    run_check,
    tree_fingerprint,
)


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

    def test_check_only_parsed_as_leading_flag(self):
        # A goal that merely MENTIONS --check must NOT be misparsed as the flag
        # (regression: re.search grabbed --check anywhere, mangling the goal).
        cfg = parse_loop_args('add a --check "verbose" flag to the CLI')
        assert cfg.check_cmd is None
        assert cfg.goal == 'add a --check "verbose" flag to the CLI'

    def test_leading_check_still_parsed(self):
        cfg = parse_loop_args('--check "pytest -q" fix the tests')
        assert cfg.check_cmd == "pytest -q"
        assert cfg.goal == "fix the tests"

    def test_check_after_every_still_parsed(self):
        cfg = parse_loop_args('every 30m --check "make build" keep it green')
        assert cfg.interval_s == 1800
        assert cfg.check_cmd == "make build"
        assert cfg.goal == "keep it green"

    def test_goal_starting_with_checkpoint_word_not_a_flag(self):
        # "--checkpoint ..." begins with "--check" but is NOT the flag: the
        # leading matcher requires whitespace after --check.
        cfg = parse_loop_args("--checkpoint the database before migrating")
        assert cfg.check_cmd is None
        assert cfg.goal == "--checkpoint the database before migrating"


class TestResponseSignalsDone:
    def test_marker_at_end_of_line(self):
        assert _response_signals_done("all fixed LOOP_DONE") is True

    def test_marker_on_own_final_line(self):
        assert _response_signals_done("did the work\nLOOP_DONE") is True

    def test_marker_with_trailing_punctuation(self):
        assert _response_signals_done("done here. LOOP_DONE.") is True

    def test_marker_mentioned_mid_reply_is_not_done(self):
        # The false-completion the fix targets.
        assert _response_signals_done(
            "I won't write LOOP_DONE until the tests pass") is False

    def test_marker_followed_by_more_text_is_not_done(self):
        assert _response_signals_done(
            "LOOP_DONE\nActually, one more thing to check") is False

    def test_continue_marker_is_not_done(self):
        assert _response_signals_done("still working LOOP_CONTINUE") is False

    def test_empty_is_not_done(self):
        assert _response_signals_done("") is False
        assert _response_signals_done(None) is False


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


def _fake_run_sequence(*results):
    """Build a subprocess.run that returns different results for each call.
    Each result is (returncode, stdout, stderr) or raises an exception.
    """
    calls = []
    result_list = list(results)

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        if not result_list:
            raise RuntimeError("fake_run called more times than expected")
        result = result_list.pop(0)
        if isinstance(result, Exception):
            raise result
        returncode, stdout, stderr = result
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

    def test_apply_fails(self, monkeypatch):
        # Push succeeds but apply --index fails: DANGEROUS (tree reverted, WIP
        # stranded in the stash) — distinct from "failed" so the loop can halt.
        fake = _fake_run_sequence(
            (0, "Saved working directory", ""),  # push succeeds
            (1, "", "conflict"),  # apply fails
        )
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        assert make_checkpoint("x") == "apply-failed"


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

    def test_git_diff_fails(self, monkeypatch):
        fake = _fake_run_sequence(
            (1, "", "fatal: not a git repository"),  # git diff fails
            (0, "", ""),  # git ls-files still runs but doesn't matter
        )
        monkeypatch.setattr(loop_mod.subprocess, "run", fake)
        assert tree_fingerprint() == ""


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
    return loop_mod.LoopEngine(loop_mod.LoopConfig(**cfg), _QuietConsole(), runner,
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
        eng = loop_mod.LoopEngine(loop_mod.LoopConfig(goal="fix"), _QuietConsole(),
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


class _RecConsole:
    """Console that records the raw (markup-laden) strings printed to it."""

    def __init__(self):
        self.msgs: list[str] = []

    def print(self, *args, **kwargs):
        self.msgs.append(str(args[0]) if args else "")


class TestLoopDoneMarkerGating:
    def test_mid_reply_mention_does_not_complete_loop(self, tmp_path, monkeypatch):
        # Round 1 MENTIONS the marker while continuing — must NOT complete.
        # Round 2 ends with it — completes there.
        runner = FakeRunner([
            "I won't write LOOP_DONE until the tests pass. LOOP_CONTINUE",
            "all fixed now LOOP_DONE",
        ])
        eng = _engine({"goal": "fix it"}, runner, tmp_path, monkeypatch,
                      fingerprints=["a", "b", "c", "d"])  # changes each round
        status = asyncio.run(eng.run())
        assert status == "done"
        assert eng.state.round_n == 2


class TestTimedStallGuard:
    def test_timed_loop_does_not_stall_on_idle_rounds(self, tmp_path, monkeypatch):
        # A timed patrol whose rounds change nothing and self-report NOT-DONE
        # must keep patrolling — idle rounds are its HEALTHY state, not a stall.
        class StopOnThird(FakeRunner):
            async def __call__(self, prompt):
                if len(self.prompts) >= 2:
                    self.eng.request_stop()
                return await super().__call__(prompt)

        runner = StopOnThird(["LOOP_CONTINUE"] * 6)
        eng = _engine({"goal": "patrol", "interval_s": 1}, runner, tmp_path,
                      monkeypatch, fingerprints=["same"] * 20)  # never changes
        runner.eng = eng
        status = asyncio.run(eng.run())
        assert status == "stopped"      # NOT "stalled"
        assert eng.state.round_n == 3   # ran past the 2-round stall cap


class TestNonGitDir:
    def test_git_unavailable_warns_and_disables_stall(self, tmp_path, monkeypatch):
        # No git → make_checkpoint "git-unavailable", tree_fingerprint "" (never
        # "changed"). The until-done loop must NOT stall (real progress is
        # invisible to git) and must warn once about the missing safety net.
        class StopAfterFour(FakeRunner):
            async def __call__(self, prompt):
                if len(self.prompts) >= 3:
                    self.eng.request_stop()
                return await super().__call__(prompt)

        runner = StopAfterFour(["LOOP_CONTINUE"] * 8)
        monkeypatch.setattr(loop_mod, "make_checkpoint",
                            lambda label: "git-unavailable")
        monkeypatch.setattr(loop_mod, "tree_fingerprint", lambda: "")
        monkeypatch.setattr(loop_mod, "run_check",
                            lambda c, timeout_s=300: (False, ""))
        console = _RecConsole()
        eng = loop_mod.LoopEngine(loop_mod.LoopConfig(goal="fix"), console,
                                  runner, loops_dir=str(tmp_path / "loops"))
        runner.eng = eng
        status = asyncio.run(eng.run())
        assert status == "stopped"       # NOT "stalled"
        assert eng.state.round_n == 4    # ran well past the 2-round stall cap
        assert any("checkpoint safety net" in m for m in console.msgs)

    def test_checkpoint_apply_failed_halts_loudly(self, tmp_path, monkeypatch):
        # A stash pushed but not re-applied leaves the WIP stranded and the tree
        # reverted — the loop must STOP with recovery steps and never run the
        # model against the wrong state.
        runner = FakeRunner(["LOOP_CONTINUE", "LOOP_DONE"])  # would run if alive
        monkeypatch.setattr(loop_mod, "make_checkpoint",
                            lambda label: "apply-failed")
        monkeypatch.setattr(loop_mod, "tree_fingerprint", lambda: "x")
        monkeypatch.setattr(loop_mod, "run_check",
                            lambda c, timeout_s=300: (False, ""))
        console = _RecConsole()
        eng = loop_mod.LoopEngine(loop_mod.LoopConfig(goal="fix"), console,
                                  runner, loops_dir=str(tmp_path / "loops"))
        status = asyncio.run(eng.run())
        assert status == "checkpoint-failed"
        assert runner.prompts == []      # the round never ran — model untouched
        assert any("git stash pop" in m for m in console.msgs)


class TestTimedSleepAnchoring:
    def test_timed_sleep_anchored_to_round_start(self, tmp_path, monkeypatch):
        # A slow round must NOT push out the next run: next_run_at is anchored
        # to when the round STARTED (round_start + interval), not when it ended,
        # so a 30-min interval with 4-min rounds fires every 30 min, not 34.
        eng_holder = {}

        async def slow_runner(prompt):
            eng_holder["eng"].request_stop()   # stop so we don't really sleep
            await asyncio.sleep(0.4)            # round takes measurable time
            return "LOOP_CONTINUE"

        eng = _engine({"goal": "patrol", "interval_s": 1}, slow_runner, tmp_path,
                      monkeypatch, fingerprints=["a", "b"])
        eng_holder["eng"] = eng
        t_start = time.time()
        asyncio.run(eng.run())
        period = eng.state.next_run_at - t_start
        # Anchored ≈ interval_s (1.0); the un-anchored bug would give ≈1.4
        # (interval + the 0.4s round).
        assert 0.85 <= period <= 1.3, period


class TestStatusLine:
    def test_status_line_mentions_round(self, tmp_path, monkeypatch):
        runner = FakeRunner(["LOOP_DONE"])
        eng = _engine({"goal": "fix"}, runner, tmp_path, monkeypatch,
                      fingerprints=["a", "b"])
        asyncio.run(eng.run())
        line = eng.status_line()
        assert "done" in line and "1" in line


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
