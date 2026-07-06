"""Regression tests for pure cli.py helpers touched by the audit fixes."""

from unittest.mock import MagicMock

from rich.console import Console

from spark_code.cli import (
    _auto_read_context,
    _checkpoint_resume_note,
    _detect_file_mentions,
    _is_shell_command,
    _redacted_config,
    _restore_checkpoint,
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
        assert "already exists" in deps["console"].export_text()

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
