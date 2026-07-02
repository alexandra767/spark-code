"""Regression tests for the slash-command autocomplete.

Guards the start_position bug where completing a subcommand ("/plan sh" →
"/plan show") corrupted the buffer into "/pl/plan show".
"""

from prompt_toolkit.document import Document

from spark_code.ui.input import SlashCommandCompleter


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
