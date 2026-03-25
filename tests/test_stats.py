"""Tests for session statistics tracking."""

import time

from spark_code.stats import SessionStats


def test_stats_initial_state():
    stats = SessionStats()
    assert stats.total_tool_calls == 0
    assert len(stats.files_read) == 0
    assert len(stats.files_written) == 0
    assert len(stats.files_edited) == 0
    assert stats.commands_run == 0


def test_record_read_file():
    stats = SessionStats()
    stats.record_tool_call("read_file", {"file_path": "/tmp/foo.py"})
    assert stats.tool_calls["read_file"] == 1
    assert "/tmp/foo.py" in stats.files_read
    assert stats.total_tool_calls == 1


def test_record_write_file():
    stats = SessionStats()
    stats.record_tool_call("write_file", {"file_path": "/tmp/bar.py", "content": "hello"})
    assert stats.tool_calls["write_file"] == 1
    assert "/tmp/bar.py" in stats.files_written


def test_record_edit_file():
    stats = SessionStats()
    stats.record_tool_call("edit_file", {"file_path": "/tmp/baz.py", "old_string": "a", "new_string": "b"})
    assert stats.tool_calls["edit_file"] == 1
    assert "/tmp/baz.py" in stats.files_edited


def test_record_bash():
    stats = SessionStats()
    stats.record_tool_call("bash", {"command": "ls -la"})
    assert stats.commands_run == 1
    assert stats.tool_calls["bash"] == 1


def test_record_multiple():
    stats = SessionStats()
    stats.record_tool_call("read_file", {"file_path": "/a"})
    stats.record_tool_call("read_file", {"file_path": "/b"})
    stats.record_tool_call("bash", {"command": "echo hi"})
    stats.record_tool_call("write_file", {"file_path": "/c", "content": "x"})
    assert stats.total_tool_calls == 4
    assert stats.tool_calls["read_file"] == 2
    assert len(stats.files_read) == 2


def test_dedup_files():
    stats = SessionStats()
    stats.record_tool_call("read_file", {"file_path": "/tmp/same.py"})
    stats.record_tool_call("read_file", {"file_path": "/tmp/same.py"})
    assert len(stats.files_read) == 1
    assert stats.tool_calls["read_file"] == 2


def test_elapsed_time():
    stats = SessionStats()
    assert stats.elapsed >= 0
    assert stats.elapsed < 1.0


def test_format_duration_seconds():
    stats = SessionStats()
    stats.start_time = time.monotonic() - 45
    assert stats.format_duration() == "45s"


def test_format_duration_minutes():
    stats = SessionStats()
    stats.start_time = time.monotonic() - 125
    assert stats.format_duration() == "2m 5s"


def test_format_duration_hours():
    stats = SessionStats()
    stats.start_time = time.monotonic() - 3725
    assert stats.format_duration() == "1h 2m"


def test_record_unknown_tool():
    stats = SessionStats()
    stats.record_tool_call("web_search", {"query": "hello"})
    assert stats.tool_calls["web_search"] == 1
    assert stats.total_tool_calls == 1
