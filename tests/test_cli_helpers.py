"""Regression tests for pure cli.py helpers touched by the audit fixes."""

from spark_code.cli import _is_shell_command, _redacted_config


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
