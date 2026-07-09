"""Tests for the /loop engine (spark_code/loop.py)."""

import pytest

from spark_code.loop import parse_interval, parse_loop_args


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
