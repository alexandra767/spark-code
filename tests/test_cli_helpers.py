"""Regression tests for pure cli.py helpers touched by the audit fixes."""

from spark_code.cli import (
    _checkpoint_resume_note,
    _is_shell_command,
    _redacted_config,
    _restore_checkpoint,
)
from spark_code.context import Context


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
