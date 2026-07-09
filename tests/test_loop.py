"""Tests for the /loop engine (spark_code/loop.py)."""

import subprocess as _subprocess
from types import SimpleNamespace

import pytest

from spark_code import loop as loop_mod
from spark_code.loop import (
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
