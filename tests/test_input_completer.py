"""Regression tests for the slash-command autocomplete.

Guards the start_position bug where completing a subcommand ("/plan sh" →
"/plan show") corrupted the buffer into "/pl/plan show".
"""

from prompt_toolkit.document import Document

from spark_code.plan_mode import MODE_CYCLE
from spark_code.ui.input import (
    _BUILTIN_COMMANDS,
    FilePathCompleter,
    SlashCommandCompleter,
    create_session,
)


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
    explicitly (here: /trust, --trust, /mode trust). The cycle now lives in
    plan_mode.MODE_CYCLE (Phase 2 Task 5) — the single copy."""
    assert MODE_CYCLE == ("ask", "auto", "plan")


def test_file_completer_handles_path_argument():
    assert _count_file("cat /tmp/") > 0


def test_skill_name_colliding_with_real_command_does_not_clobber_it(tmp_path):
    # Phase 2 Task 8: the builtin `review` SKILL shares its name with the real
    # `/review` COMMAND. A naive merge of skill names into the completer's
    # command map would overwrite /review's real description with "" (skill
    # descriptions are looked up separately, not fed into this completer) —
    # "real commands win" must hold in autocomplete, not just dispatch.
    session = create_session(
        history_file=str(tmp_path / "hist"),
        skill_names=["/review", "/commit"],
    )
    slash_completer = next(
        c for c in session.completer.completers if hasattr(c, "_commands")
    )
    assert slash_completer._commands["/review"] == _BUILTIN_COMMANDS["/review"]
    # A skill with no name collision still gets added (empty description —
    # its real description lives on the Skill object, shown via /help).
    assert "/commit" in slash_completer._commands


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


def _completions(completer, text):
    doc = Document(text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, None)]


def test_argument_completion_from_provider():
    c = SlashCommandCompleter(value_providers={"/model": lambda: ["llm", "coder", "gemini"]})
    got = _completions(c, "/model co")
    assert any(x.endswith("coder") for x in got)
    assert not any(x.endswith("gemini") for x in got)


def test_argument_completion_lists_all_on_bare_space():
    c = SlashCommandCompleter(value_providers={"/model": lambda: ["llm", "coder"]})
    got = _completions(c, "/model ")
    assert len(got) == 2


def test_provider_exception_yields_no_completions():
    def boom():
        raise RuntimeError("x")
    c = SlashCommandCompleter(value_providers={"/model": boom})
    assert _completions(c, "/model l") == []


def test_command_completion_unchanged_without_arg():
    c = SlashCommandCompleter(value_providers={"/model": lambda: ["llm"]})
    got = _completions(c, "/mod")
    assert any("/model" in x for x in got)


def test_model_candidates_include_list_subcommand():
    # Review fast-follow: wiring provider names as /model's value provider
    # must not shadow the real "/model list" subcommand — "list" has to stay
    # a suggested argument alongside the provider names.
    from spark_code.cli import _model_arg_candidates

    cfg = {"providers": {"llm": {}, "coder": {}}}
    c = SlashCommandCompleter(
        value_providers={"/model": lambda: _model_arg_candidates(cfg)}
    )
    got = _completions(c, "/model li")
    assert "list" in got
    assert "llm" not in got  # prefix filter still applies
    got = _completions(c, "/model l")
    assert "llm" in got and "list" in got


def test_mode_candidates_derived_from_permissions_constants():
    # Review fast-follow: /mode's candidate list is derived from the
    # permissions constants (single source of truth), not a hand-typed
    # duplicate in cli.py that could drift when a mode is added/renamed.
    from spark_code.cli import _mode_arg_candidates
    from spark_code.permissions import DISPLAY_ALIASES, MODE_ALIASES, NATIVE_MODES

    derived = sorted(NATIVE_MODES) + sorted(
        a for a in DISPLAY_ALIASES if a.lower() not in NATIVE_MODES
    )
    assert _mode_arg_candidates() == derived
    # DISPLAY_ALIASES must cover exactly MODE_ALIASES' keys (it exists only
    # to preserve the camelCase spellings that the lowercased keys destroy).
    assert {a.lower() for a in DISPLAY_ALIASES} == set(MODE_ALIASES)
    # The Claude Code spellings users actually type must be offered verbatim.
    got = _mode_arg_candidates()
    assert "acceptEdits" in got
    assert "bypassPermissions" in got
    for native in NATIVE_MODES:
        assert native in got
