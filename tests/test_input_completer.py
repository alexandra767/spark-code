"""Regression tests for the slash-command autocomplete.

Guards the start_position bug where completing a subcommand ("/plan sh" →
"/plan show") corrupted the buffer into "/pl/plan show".
"""

from prompt_toolkit.document import Document

from spark_code.ui.input import _MODE_CYCLE, FilePathCompleter, SlashCommandCompleter


def _apply(text: str):
    """Return the buffer text after accepting each offered completion."""
    completer = SlashCommandCompleter()
    doc = Document(text=text, cursor_position=len(text))
    results = []
    for comp in completer.get_completions(doc, None):
        # start_position is negative offset from the cursor
        applied = text[:len(text) + comp.start_position] + comp.text
        results.append(applied)
    return results


def test_subcommand_completion_not_corrupted():
    applied = _apply("/plan sh")
    assert applied, "expected at least one completion for '/plan sh'"
    for buf in applied:
        assert buf == "/plan show", f"corrupted completion: {buf!r}"
        assert "/pl/" not in buf


def test_bare_command_completes_cleanly():
    applied = _apply("/pla")
    assert all(buf.startswith("/plan") and "/pl/pl" not in buf for buf in applied)


def test_completed_command_offers_subcommands_only():
    # Typing the exact command shouldn't re-offer itself (cmd != word guard).
    applied = _apply("/plan")
    assert "/plan" not in [b for b in applied]  # only subcommand variants
    assert all(b.startswith("/plan") for b in applied)


def test_non_slash_yields_nothing():
    assert _apply("hello world") == []


def _count_file(text: str) -> int:
    completer = FilePathCompleter()
    doc = Document(text=text, cursor_position=len(text))
    return len(list(completer.get_completions(doc, None)))


def test_file_completer_defers_lone_slash_command():
    # A single "/word" token belongs to the slash-command completer.
    assert _count_file("/plan") == 0


def test_file_completer_handles_absolute_path():
    # Absolute path (two slashes) should complete against the filesystem.
    assert _count_file("/tmp/") > 0


def test_shift_tab_mode_cycle_matches_claude_code_order():
    """Task 5 (Phase 1): trust leaves the Shift+Tab cycle — it matches
    Claude Code, where bypassPermissions is never cycled into, only entered
    explicitly (here: /trust, --trust, /mode trust)."""
    assert _MODE_CYCLE == ["ask", "auto", "plan"]


def test_file_completer_handles_path_argument():
    assert _count_file("cat /tmp/") > 0


def test_at_mention_completes_directory(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path)
    completer = FilePathCompleter()
    text = "fix @sr"
    doc = Document(text=text, cursor_position=len(text))
    completions = list(completer.get_completions(doc, None))
    assert "@src/" in [c.text for c in completions]
    # Accepting the completion must reconstruct the buffer with the "@" kept.
    comp = next(c for c in completions if c.text == "@src/")
    applied = text[:len(text) + comp.start_position] + comp.text
    assert applied == "fix @src/"
