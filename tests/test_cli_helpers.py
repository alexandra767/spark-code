"""Regression tests for pure cli.py helpers touched by the audit fixes."""

from unittest.mock import MagicMock

from rich.console import Console

from spark_code.cli import (
    _auto_read_context,
    _checkpoint_resume_note,
    _context_pct_style,
    _detect_file_mentions,
    _is_shell_command,
    _list_sessions,
    _redacted_config,
    _restore_checkpoint,
    _resume_session_into,
    handle_slash_command,
)
from spark_code.context import Context
from spark_code.model import ModelClient
from spark_code.permissions import PermissionManager
from spark_code.skills.base import SkillRegistry


class TestCheckpointResumeNote:
    def test_resume_note_uses_user_role_not_system(self):
        # Audit 2026-07-06 item 6: /continue previously appended a second
        # role:"system" message mid-history (get_messages already prepends the
        # real system prompt at index 0), which strict OpenAI-compatible servers
        # reject. The resume note must be role:"user" (mirroring compaction).
        note = _checkpoint_resume_note(12)
        assert note["role"] == "user"
        assert "12" in note["content"]


class TestRestoreCheckpoint:
    def test_restores_turn_count_so_exit_autosave_fires(self):
        # Audit 2026-07-06 item 7: /continue restored messages but not
        # turn_count, so in a fresh process the exit-auto-save guard
        # (turn_count > 0) stayed False and post-continue work was never written
        # to ~/.spark/history.
        ctx = Context()
        data = {"messages": [{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "yo"}],
                "round_count": 3}
        _restore_checkpoint(ctx, data)
        assert ctx.turn_count == 3
        assert ctx.messages[0]["content"] == "hi"
        assert ctx.messages[-1]["role"] == "user"  # resume note
        # The original checkpoint data must not be mutated.
        assert len(data["messages"]) == 2

    def test_turn_count_falls_back_to_message_count(self):
        ctx = Context()
        data = {"messages": [{"role": "user", "content": "a"}]}  # no round_count
        _restore_checkpoint(ctx, data)
        assert ctx.turn_count > 0


class TestShellCommandGuard:
    def test_real_commands_run_as_shell(self):
        for cmd in ("python snake.py", "npm test", "git status", "ls -la",
                    "pytest tests/", "./run.sh", "make build", "go build ./..."):
            assert _is_shell_command(cmd), f"{cmd!r} should be a shell command"

    def test_natural_language_not_hijacked(self):
        # These start with shell-verb prefixes but are English sentences.
        for phrase in ("make the tests pass", "go through the code and fix X",
                       "cat the config into one file for me please",
                       "make it work", "go to the next step"):
            assert not _is_shell_command(phrase), f"{phrase!r} should go to the model"

    def test_empty_is_not_shell(self):
        assert not _is_shell_command("")
        assert not _is_shell_command("   ")


class TestRedactedConfig:
    def test_api_keys_masked(self):
        cfg = {
            "model": {"api_key": "secret", "name": "m"},
            "providers": {"openai": {"api_key": "sk-live"}, "ollama": {"api_key": ""}},
        }
        red = _redacted_config(cfg)
        assert red["model"]["api_key"] == "***redacted***"
        assert red["providers"]["openai"]["api_key"] == "***redacted***"
        # Empty keys stay empty (nothing to leak)
        assert red["providers"]["ollama"]["api_key"] == ""
        # Non-secret values preserved
        assert red["model"]["name"] == "m"

    def test_original_not_mutated(self):
        cfg = {"model": {"api_key": "secret"}}
        _redacted_config(cfg)
        assert cfg["model"]["api_key"] == "secret"


class TestContextPctStyle:
    """Toolbar color thresholds for the context-left meter (Phase 2 Task 1):
    green with plenty of room, yellow approaching the limit, red about to
    trigger auto-compaction / a context-length 400."""

    def test_green_above_40(self):
        assert _context_pct_style(41) == "class:bottom-toolbar.context-green"
        assert _context_pct_style(100) == "class:bottom-toolbar.context-green"

    def test_yellow_between_15_and_40_inclusive(self):
        assert _context_pct_style(40) == "class:bottom-toolbar.context-yellow"
        assert _context_pct_style(25) == "class:bottom-toolbar.context-yellow"
        assert _context_pct_style(15) == "class:bottom-toolbar.context-yellow"

    def test_red_below_15(self):
        assert _context_pct_style(14.9) == "class:bottom-toolbar.context-red"
        assert _context_pct_style(0) == "class:bottom-toolbar.context-red"


def _init_command_deps():
    """Minimal handle_slash_command dependencies for /init tests."""
    config = {
        "model": {
            "endpoint": "http://localhost:11434",
            "name": "test-model",
            "temperature": 0.7,
            "max_tokens": 4096,
            "api_key": "",
            "provider": "ollama",
        },
        "permissions": {"mode": "auto", "always_allow": []},
    }
    return {
        "context": Context(),
        "console": Console(record=True, width=120),
        "config": config,
        "skills": SkillRegistry(),
        "model": MagicMock(spec=ModelClient),
        "permissions": PermissionManager(mode="auto"),
    }


class TestAtMentions:
    def test_at_mention_extracted(self, tmp_path, monkeypatch):
        (tmp_path / "auth.py").write_text("x = 1")
        monkeypatch.chdir(tmp_path)
        got = _detect_file_mentions("fix the bug in @auth.py please")
        assert "auth.py" in got

    def test_at_mention_with_path(self, tmp_path, monkeypatch):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1")
        monkeypatch.chdir(tmp_path)
        got = _detect_file_mentions("look at @src/app.py")
        assert "src/app.py" in got

    def test_at_mention_directory(self, tmp_path, monkeypatch):
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        got = _detect_file_mentions("what is in @src")
        assert "src" in got

    def test_email_not_treated_as_mention(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        got = _detect_file_mentions("email me at alexandra@gmail.com")
        assert "gmail.com" not in got

    def test_at_mention_trailing_punctuation(self, tmp_path, monkeypatch):
        # "look at @app.py." — the regex's character class includes ".", so
        # the raw capture is "app.py." which fails os.path.exists; the
        # mention must not be silently dropped for ordinary sentence
        # punctuation.
        (tmp_path / "app.py").write_text("x = 1")
        monkeypatch.chdir(tmp_path)
        got = _detect_file_mentions("look at @app.py.")
        assert got == ["app.py"]

    def test_at_mention_trailing_comma_on_directory(self, tmp_path, monkeypatch):
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        got = _detect_file_mentions("check @src, please")
        assert "src" in got

    def test_at_mention_path_with_trailing_punct(self, tmp_path, monkeypatch):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1")
        monkeypatch.chdir(tmp_path)
        got = _detect_file_mentions("see @src/app.py.")
        assert "src/app.py" in got


class TestAutoReadContext:
    """@dir mentions must render as a listing, not silently vanish.

    The old inline auto-read block only ever called open(fpath), which
    raises IsADirectoryError (an OSError subclass) on a directory and was
    swallowed by `except OSError: pass` — directories were mentioned but
    produced zero context. _auto_read_context adds a directory branch that
    routes through list_dir's tool logic instead.
    """

    async def test_file_mention_reads_contents(self, tmp_path):
        (tmp_path / "auth.py").write_text("x = 1")
        console = Console(record=True, width=120)
        ctx = await _auto_read_context([str(tmp_path / "auth.py")], console)
        assert "x = 1" in ctx
        assert "Auto-read" in console.export_text()

    async def test_directory_mention_lists_contents(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("x = 1")
        console = Console(record=True, width=120)
        ctx = await _auto_read_context([str(tmp_path / "src")], console)
        assert "app.py" in ctx
        assert "Directory listing" in ctx
        assert "Auto-listed" in console.export_text()


class TestInitCommand:
    def test_existing_claude_md_prints_note_and_skips_agent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CLAUDE.md").write_text("# Existing project notes")
        deps = _init_command_deps()

        result = handle_slash_command(
            "/init", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )

        # Handled entirely by the command itself — no prompt for the agent,
        # so the real dispatch loop's "if result is None: continue" means
        # the agent is never invoked for this turn.
        assert result is None
        output = deps["console"].export_text()
        assert "already exists" in output
        # The note names the discovered path so the user knows which file wins.
        from spark_code.instructions import _find_project_claude_md
        expected = _find_project_claude_md(str(tmp_path))
        assert expected is not None
        assert "".join(expected.split()) in "".join(output.split())

    def test_ancestor_claude_md_from_subdir_skips_agent(self, tmp_path, monkeypatch):
        # Finding 3: /init must use the loader's nearest-ancestor discovery, not
        # just $CWD/CLAUDE.md. Running it from a subdirectory of a repo that
        # already has a root CLAUDE.md must NOT write a new one that would
        # permanently shadow the repo's — it must detect the ancestor and skip.
        (tmp_path / "CLAUDE.md").write_text("# Root project notes")
        subdir = tmp_path / "pkg" / "sub"
        subdir.mkdir(parents=True)
        monkeypatch.chdir(subdir)
        deps = _init_command_deps()

        result = handle_slash_command(
            "/init", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )

        assert result is None  # no prompt → no shadowing CLAUDE.md written
        output = deps["console"].export_text()
        assert "already exists" in output
        from spark_code.instructions import _find_project_claude_md
        expected = _find_project_claude_md(str(subdir))
        assert expected is not None
        # The ancestor (root) file is the one named, not a would-be subdir path.
        assert "".join(expected.split()) in "".join(output.split())

    def test_missing_claude_md_returns_prompt_for_agent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        deps = _init_command_deps()

        result = handle_slash_command(
            "/init", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )

        # No CLAUDE.md yet — /init hands a one-shot prompt back to the
        # dispatch loop, which sends it through the existing agent object
        # (same path as any other user message), so the model writes the
        # file itself with write_file.
        assert result is not None
        assert "CLAUDE.md" in result
        assert "write_file" in result


# ---------------------------------------------------------------------------
# _list_sessions / _resume_session_into (Task 4: /resume + /rewind)
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_list_sessions_sorted_newest_first(self, tmp_path, monkeypatch):
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        (tmp_path / "20260101_000000_old-task.json").write_text('{"messages": []}')
        (tmp_path / "20260706_120000_new-task.json").write_text('{"messages": []}')
        sessions = _list_sessions()
        assert len(sessions) == 2
        assert "new-task" in sessions[0]["path"]

    def test_list_sessions_empty_when_dir_missing(self, tmp_path, monkeypatch):
        import spark_code.cli as cli
        monkeypatch.setattr(
            cli, "_sessions_dir", lambda create=True: str(tmp_path / "does-not-exist")
        )
        assert _list_sessions() == []

    def test_list_sessions_reads_label_turns_cwd(self, tmp_path, monkeypatch):
        # Build the fixture with real Context.save so the shape (timestamp,
        # turn_count, label, cwd, messages) matches what /resume will
        # actually encounter on disk (see context.py:440).
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        ctx = Context()
        ctx.add_user("build a weather app")
        ctx.save(
            str(tmp_path / "20260706_120000_weather.json"),
            label="weather-app", cwd="/home/user/proj",
        )
        sessions = _list_sessions()
        assert len(sessions) == 1
        sess = sessions[0]
        assert sess["name"] == "20260706_120000_weather"
        assert sess["label"] == "weather-app"
        assert sess["cwd"] == "/home/user/proj"
        assert sess["turns"] == 1

    def test_metadata_reads_capped_to_display_slice(self, tmp_path, monkeypatch):
        # Review amendment 2026-07-06: _list_sessions used to call
        # Context.read_metadata (a full JSON parse) for EVERY file in
        # ~/.spark/history on each /history or /resume invocation. Only the
        # top-10 slice that actually gets displayed may be parsed eagerly —
        # the rest still appear in the list (for name matching) but carry
        # placeholder metadata until hydrated individually.
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        for i in range(15):
            (tmp_path / f"202606{10 + i}_000000_task-{i}.json").write_text(
                '{"messages": [], "turn_count": 1}'
            )
        calls = []
        real = Context.read_metadata
        monkeypatch.setattr(
            Context, "read_metadata",
            staticmethod(lambda path: (calls.append(path), real(path))[1]),
        )
        sessions = _list_sessions()
        assert len(sessions) == 15  # every session still listed / matchable
        assert len(calls) <= 10     # but only the display slice is parsed


class TestResumeSessionInto:
    def test_loads_valid_session(self, tmp_path):
        ctx = Context()
        ctx.add_user("hi")
        ctx.add_assistant("hello")
        path = str(tmp_path / "session.json")
        ctx.save(path, label="greeting", cwd=str(tmp_path))

        fresh = Context()
        assert _resume_session_into(fresh, path) is True
        assert fresh.turn_count == 1
        assert fresh.messages[0]["content"] == "hi"

    def test_missing_file_returns_false(self, tmp_path):
        fresh = Context()
        assert _resume_session_into(fresh, str(tmp_path / "nope.json")) is False


# ---------------------------------------------------------------------------
# /resume command
# ---------------------------------------------------------------------------

class TestResumeCommand:
    def test_no_sessions_yet(self, tmp_path, monkeypatch):
        import spark_code.cli as cli
        monkeypatch.setattr(
            cli, "_sessions_dir", lambda create=True: str(tmp_path / "empty")
        )
        deps = _init_command_deps()
        result = handle_slash_command(
            "/resume", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert "No saved sessions yet" in deps["console"].export_text()

    def test_numbered_picker_prompts_and_resumes(self, tmp_path, monkeypatch):
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        ctx = Context()
        ctx.add_user("build a snake game")
        ctx.save(
            str(tmp_path / "20260706_120000_snake.json"),
            label="snake-game", cwd=str(tmp_path),
        )
        monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "1")

        deps = _init_command_deps()
        result = handle_slash_command(
            "/resume", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        out = deps["console"].export_text()
        assert "Resumed session" in out
        assert "snake" in out
        # Loaded into the LIVE context, same object the caller holds.
        assert deps["context"].turn_count == 1
        assert deps["context"].messages[0]["content"] == "build a snake game"
        # Banner reprinted after resume (mirrors /clear's print_banner call).
        assert "Spark Code" in out

    def test_direct_number_skips_prompt(self, tmp_path, monkeypatch):
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        ctx = Context()
        ctx.add_user("hello")
        ctx.save(
            str(tmp_path / "20260706_120000_hi.json"),
            label="hi-task", cwd=str(tmp_path),
        )

        def _boom(*a, **k):
            raise AssertionError("Prompt.ask should not run when a number is given")
        monkeypatch.setattr(cli.Prompt, "ask", _boom)

        deps = _init_command_deps()
        result = handle_slash_command(
            "/resume 1", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert deps["context"].turn_count == 1

    def test_direct_name_skips_prompt(self, tmp_path, monkeypatch):
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        ctx = Context()
        ctx.add_user("hello")
        ctx.save(
            str(tmp_path / "20260706_120000_hi-task.json"),
            label="hi-task", cwd=str(tmp_path),
        )

        deps = _init_command_deps()
        result = handle_slash_command(
            "/resume hi-task", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert deps["context"].turn_count == 1

    def _seed_14_decoys_plus_ancient(self, tmp_path):
        """15 sessions: 14 newer decoys + 1 oldest target beyond the top-10
        display slice, with a label distinct from its filename so output
        assertions prove its metadata really was parsed."""
        for i in range(14):
            (tmp_path / f"202607{i + 1:02d}_000000_decoy-{i}.json").write_text(
                '{"messages": [], "turn_count": 1}'
            )
        ctx = Context()
        ctx.add_user("ancient work")
        ctx.save(
            str(tmp_path / "20260101_000000_ancient-task.json"),
            label="the-old-one", cwd=str(tmp_path),
        )

    def test_resume_by_name_beyond_slice_parses_one_extra(self, tmp_path, monkeypatch):
        # Review amendment 2026-07-06: a name match may hydrate exactly one
        # entry beyond the eagerly-parsed display slice — never the whole dir.
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        self._seed_14_decoys_plus_ancient(tmp_path)

        calls = []
        real = Context.read_metadata
        monkeypatch.setattr(
            Context, "read_metadata",
            staticmethod(lambda path: (calls.append(path), real(path))[1]),
        )

        deps = _init_command_deps()
        result = handle_slash_command(
            "/resume ancient-task", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert deps["context"].turn_count == 1
        assert deps["context"].messages[0]["content"] == "ancient work"
        out = deps["console"].export_text()
        assert "Resumed session" in out
        assert "the-old-one" in out  # label shown → matched entry hydrated
        assert len(calls) <= 11      # top-10 slice + the one matched entry

    def test_history_name_match_beyond_slice_still_resumes_with_label(
        self, tmp_path, monkeypatch,
    ):
        # /history <name> must keep matching sessions beyond the display
        # slice and still show their label (output preserved post-cap).
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        self._seed_14_decoys_plus_ancient(tmp_path)

        deps = _init_command_deps()
        result = handle_slash_command(
            "/history ancient-task", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert deps["context"].turn_count == 1
        out = deps["console"].export_text()
        assert "Resumed session" in out
        assert "the-old-one" in out

    def test_invalid_number_reports_error_without_crash(self, tmp_path, monkeypatch):
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        ctx = Context()
        ctx.save(
            str(tmp_path / "20260706_120000_only.json"),
            label="only", cwd=str(tmp_path),
        )

        deps = _init_command_deps()
        result = handle_slash_command(
            "/resume 99", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert "No session #99" in deps["console"].export_text()
        assert deps["context"].turn_count == 0  # untouched

    def test_blank_answer_cancels(self, tmp_path, monkeypatch):
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        ctx = Context()
        ctx.save(
            str(tmp_path / "20260706_120000_only.json"),
            label="only", cwd=str(tmp_path),
        )
        monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "")

        deps = _init_command_deps()
        result = handle_slash_command(
            "/resume", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert "Cancelled" in deps["console"].export_text()
        assert deps["context"].turn_count == 0

    def test_ctrl_c_during_picker_returns_cleanly(self, tmp_path, monkeypatch):
        """Historical bug (see /clean): an uncaught KeyboardInterrupt from an
        interactive prompt used to unwind past the REPL loop and exit the
        whole app. /resume's picker must catch it locally and return None."""
        import spark_code.cli as cli
        monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
        ctx = Context()
        ctx.save(
            str(tmp_path / "20260706_120000_only.json"),
            label="only", cwd=str(tmp_path),
        )

        def _raise_kbd(*a, **k):
            raise KeyboardInterrupt()
        monkeypatch.setattr(cli.Prompt, "ask", _raise_kbd)

        deps = _init_command_deps()
        result = handle_slash_command(
            "/resume", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None  # did not propagate/crash the app
        assert deps["context"].turn_count == 0


# ---------------------------------------------------------------------------
# /rewind command
# ---------------------------------------------------------------------------

class TestRewindCommand:
    def test_menu_shows_honest_labels_and_option1_delegates_to_undo(self, monkeypatch):
        # Plan amendment 2026-07-06 (Phase 1): the menu must label options by
        # what they actually do (undo stack / git stash checkpoint), not fake
        # a message rewind. Phase 5 Task 7 made "conversation" real (true
        # per-turn snapshot rewind), so the menu now legitimately shows it.
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_undo_last", lambda console, n: calls.append(("undo", n)))
        monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "1")

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == [("undo", 1)]
        out = deps["console"].export_text()
        assert "Rewind" in out
        assert "last file edits (undo stack)" in out
        assert "checkpoint (git stash)" in out
        assert "conversation" in out

    def test_word_alias_undo_skips_prompt(self, monkeypatch):
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_undo_last", lambda console, n: calls.append(("undo", n)))

        def _boom(*a, **k):
            raise AssertionError("Prompt.ask should not run when an option is given")
        monkeypatch.setattr(cli.Prompt, "ask", _boom)

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind undo", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == [("undo", 1)]

    def test_word_alias_checkpoint_delegates_to_rewind_files(self, monkeypatch):
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_rewind_files", lambda console: calls.append("files"))

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind checkpoint", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == ["files"]

    def test_autocomplete_text_is_honest(self):
        from spark_code.ui.input import _BUILTIN_COMMANDS
        assert _BUILTIN_COMMANDS["/rewind"] == "Restore files (undo/checkpoint) or the conversation itself"
        assert _BUILTIN_COMMANDS["/resume"] == "Pick a past session to resume"

    def test_files_option_delegates_to_rewind_files(self, monkeypatch):
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_rewind_files", lambda console: calls.append("files"))

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind 2", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == ["files"]

    def test_both_option_delegates_to_both_in_order(self, monkeypatch):
        # Phase 5 Task 7: "both" now also rewinds the conversation, in
        # addition to the two file-restore actions it already did.
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_undo_last", lambda console, n: calls.append(("undo", n)))
        monkeypatch.setattr(cli, "_rewind_files", lambda console: calls.append("files"))
        monkeypatch.setattr(cli, "_rewind_conversation",
                            lambda console, context: calls.append("conversation"))

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind both", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == [("undo", 1), "files", "conversation"]

    def test_word_alias_conversation_delegates_to_rewind_conversation(self, monkeypatch):
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_rewind_conversation",
                            lambda console, context: calls.append("conversation"))

        def _boom(*a, **k):
            raise AssertionError("Prompt.ask should not run when an option is given")
        monkeypatch.setattr(cli.Prompt, "ask", _boom)

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind conversation", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == ["conversation"]

    def test_numeric_option4_delegates_to_rewind_conversation(self, monkeypatch):
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_rewind_conversation",
                            lambda console, context: calls.append("conversation"))

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind 4", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == ["conversation"]

    def test_unknown_option_reports_error_without_crash(self, monkeypatch):
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_undo_last", lambda console, n: calls.append(("undo", n)))
        monkeypatch.setattr(cli, "_rewind_files", lambda console: calls.append("files"))

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind bogus", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == []
        assert "Unknown /rewind option" in deps["console"].export_text()

    def test_ctrl_c_during_menu_returns_cleanly(self, monkeypatch):
        import spark_code.cli as cli
        calls = []
        monkeypatch.setattr(cli, "_undo_last", lambda console, n: calls.append(("undo", n)))
        monkeypatch.setattr(cli, "_rewind_files", lambda console: calls.append("files"))

        def _raise_kbd(*a, **k):
            raise KeyboardInterrupt()
        monkeypatch.setattr(cli.Prompt, "ask", _raise_kbd)

        deps = _init_command_deps()
        result = handle_slash_command(
            "/rewind", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None  # did not propagate/crash the app
        assert calls == []


# ---------------------------------------------------------------------------
# /compact — model-summary flow (__COMPACT__ sentinel → _handle_compact)
# ---------------------------------------------------------------------------

class TestHandleCompact:
    """Review round 1 (Important #2): the __COMPACT__ REPL branch awaited the
    summary request WITHOUT the KeyboardInterrupt guard every sibling sentinel
    branch has — Ctrl+C during /compact unwound past the REPL and killed the
    whole session. _handle_compact must catch it locally, print a cancel note,
    and leave the history untouched."""

    def _console(self):
        import io
        return Console(file=io.StringIO(), record=True, force_terminal=True)

    def _filled_context(self):
        ctx = Context()
        pad = " lorem ipsum dolor sit amet" * 40
        for i in range(10):
            ctx.add_user(f"user {i}{pad}")
            ctx.add_assistant(f"reply {i}{pad}")
        return ctx

    async def test_ctrl_c_during_summary_cancels_cleanly(self):
        from spark_code.cli import _handle_compact

        class KbdModel:
            async def chat(self, **kwargs):
                raise KeyboardInterrupt()
                yield  # pragma: no cover — makes this an async generator

        ctx = self._filled_context()
        before = list(ctx.messages)
        console = self._console()

        # Must NOT propagate (a raise here is the session-killing bug).
        await _handle_compact(KbdModel(), ctx, console, None)

        assert "Compaction cancelled" in console.export_text()
        assert ctx.messages == before  # nothing compacted, nothing lost

    async def test_success_path_compacts_with_model_summary(self):
        from spark_code.cli import _handle_compact

        class OkModel:
            async def chat(self, **kwargs):
                yield {"type": "text", "content": "THE RECAP"}
                yield {"type": "done", "usage": {}}

        ctx = self._filled_context()
        before = len(ctx.messages)
        console = self._console()

        await _handle_compact(OkModel(), ctx, console, "focus on parser work")

        assert len(ctx.messages) < before
        assert "THE RECAP" in ctx.messages[0]["content"]
        assert "Context compacted" in console.export_text()

    async def test_nothing_to_compact_message(self):
        from spark_code.cli import _handle_compact

        class OkModel:
            async def chat(self, **kwargs):
                yield {"type": "text", "content": "RECAP"}
                yield {"type": "done", "usage": {}}

        ctx = Context()
        ctx.add_user("just one message")
        console = self._console()

        await _handle_compact(OkModel(), ctx, console, None)

        assert "Nothing to compact" in console.export_text()
        assert len(ctx.messages) == 1
