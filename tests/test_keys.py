"""Tests for spark_code.keys — presets, keys-file storage, masking, precedence,
and the `/setkey` CLI command.

SECURITY: every test here uses a MONKEYPATCHED keys_path() pointed at a
pytest tmp_path — never the real ~/.spark/keys — and only ever writes DUMMY
values (never a real API key). See the module docstring in spark_code/keys.py
for the full design.
"""

import json
import os
import stat

import pytest
from rich.console import Console

from spark_code.context import Context
from spark_code.keys import (
    PROVIDER_PRESETS,
    keys_path,
    load_keys,
    mask,
    resolve_provider_key,
    save_key,
)
from spark_code.model import ModelClient
from spark_code.permissions import PermissionManager
from spark_code.skills.base import SkillRegistry

DUMMY_ANTHROPIC_KEY = "sk-ant-api03-" + ("d" * 40)
DUMMY_OPENROUTER_KEY = "sk-or-v1-" + ("e" * 40)


# ---------------------------------------------------------------------------
# PROVIDER_PRESETS
# ---------------------------------------------------------------------------


class TestProviderPresets:
    def test_has_the_four_providers(self):
        assert set(PROVIDER_PRESETS.keys()) == {
            "anthropic", "openai", "gemini", "openrouter",
        }

    @pytest.mark.parametrize("name", ["anthropic", "openai", "gemini", "openrouter"])
    def test_each_preset_has_required_fields(self, name):
        preset = PROVIDER_PRESETS[name]
        for field in ("endpoint", "model", "key_env", "doc_url"):
            assert field in preset
            assert isinstance(preset[field], str)
            assert preset[field]

    def test_anthropic_preset_values(self):
        preset = PROVIDER_PRESETS["anthropic"]
        assert preset["endpoint"] == "https://api.anthropic.com/v1/"
        assert preset["model"] == "claude-sonnet-4-5"
        assert preset["key_env"] == "ANTHROPIC_API_KEY"

    def test_openrouter_preset_values(self):
        preset = PROVIDER_PRESETS["openrouter"]
        assert preset["endpoint"] == "https://openrouter.ai/api/v1"
        assert preset["model"] == "openrouter/auto"
        assert preset["key_env"] == "OPENROUTER_API_KEY"


# ---------------------------------------------------------------------------
# keys_path / load_keys / save_key
# ---------------------------------------------------------------------------


class TestKeysPath:
    def test_default_path_is_under_dot_spark(self):
        assert keys_path() == os.path.expanduser("~/.spark/keys")


class TestLoadKeysAbsentOrMalformed:
    def test_absent_file_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / "keys"))
        assert load_keys() == {}

    def test_malformed_json_returns_empty_dict_not_raise(self, tmp_path, monkeypatch):
        path = tmp_path / "keys"
        path.write_text("not valid json {{{")
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(path))
        assert load_keys() == {}

    def test_non_dict_json_returns_empty_dict(self, tmp_path, monkeypatch):
        path = tmp_path / "keys"
        path.write_text(json.dumps(["not", "a", "dict"]))
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(path))
        assert load_keys() == {}

    def test_unreadable_directory_returns_empty_dict(self, tmp_path, monkeypatch):
        # Points at a path whose parent doesn't exist at all — must not raise.
        monkeypatch.setattr(
            "spark_code.keys.keys_path",
            lambda: str(tmp_path / "does" / "not" / "exist" / "keys"),
        )
        assert load_keys() == {}


class TestSaveKeyRoundTrip:
    def test_save_then_load_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / ".spark" / "keys"))
        path = save_key("openrouter", DUMMY_OPENROUTER_KEY)
        assert os.path.exists(path)
        keys = load_keys()
        assert keys["openrouter"] == DUMMY_OPENROUTER_KEY

    def test_chmod_600_applied(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / ".spark" / "keys"))
        path = save_key("anthropic", DUMMY_ANTHROPIC_KEY)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600

    def test_creates_parent_dir_if_absent(self, tmp_path, monkeypatch):
        target = tmp_path / "brand-new" / "keys"
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(target))
        assert not target.parent.exists()
        save_key("gemini", "dummy-gemini-key-value")
        assert target.parent.exists()

    def test_preserves_other_providers_on_update(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / "keys"))
        save_key("openai", "dummy-openai-key")
        save_key("anthropic", DUMMY_ANTHROPIC_KEY)
        keys = load_keys()
        assert keys["openai"] == "dummy-openai-key"
        assert keys["anthropic"] == DUMMY_ANTHROPIC_KEY

    def test_updating_a_provider_overwrites_only_that_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / "keys"))
        save_key("openai", "dummy-key-v1")
        save_key("openai", "dummy-key-v2")
        keys = load_keys()
        assert keys["openai"] == "dummy-key-v2"
        assert len(keys) == 1

    def test_file_content_is_valid_json_object(self, tmp_path, monkeypatch):
        target = tmp_path / "keys"
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(target))
        save_key("gemini", "dummy-gemini-key")
        with open(target) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["gemini"] == "dummy-gemini-key"


# ---------------------------------------------------------------------------
# .gitignore belt-and-suspenders (only fires if ~/.spark is ever a repo)
# ---------------------------------------------------------------------------


class TestGitignoreBeltAndSuspenders:
    def test_no_gitignore_present_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / ".spark" / "keys"))
        save_key("openai", "dummy-key")
        assert not (tmp_path / ".spark" / ".gitignore").exists()

    def test_existing_gitignore_gets_keys_entry_appended(self, tmp_path, monkeypatch):
        spark_dir = tmp_path / ".spark"
        spark_dir.mkdir()
        (spark_dir / ".gitignore").write_text("*.log\n")
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(spark_dir / "keys"))

        save_key("openai", "dummy-key")

        content = (spark_dir / ".gitignore").read_text()
        assert "keys" in content.splitlines()
        assert "*.log" in content.splitlines()

    def test_gitignore_already_covering_keys_is_not_duplicated(self, tmp_path, monkeypatch):
        spark_dir = tmp_path / ".spark"
        spark_dir.mkdir()
        (spark_dir / ".gitignore").write_text("keys\n")
        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(spark_dir / "keys"))

        save_key("openai", "dummy-key")

        content = (spark_dir / ".gitignore").read_text()
        assert content.splitlines().count("keys") == 1


# ---------------------------------------------------------------------------
# mask() — must NEVER return the full value
# ---------------------------------------------------------------------------


class TestMask:
    def test_never_returns_full_value_long_key(self):
        value = DUMMY_ANTHROPIC_KEY
        masked = mask(value)
        assert masked != value
        assert value not in masked

    def test_shows_first_three_and_last_four(self):
        value = "sk-or-v1-" + ("z" * 40)
        masked = mask(value)
        assert masked.startswith(value[:3])
        assert masked.endswith(value[-4:])
        assert "…" in masked

    def test_empty_value_masks_to_empty(self):
        assert mask("") == ""

    def test_short_value_never_leaks_full_value(self):
        for value in ("a", "ab", "abcdefg", "sk-1234"):
            masked = mask(value)
            assert value not in masked or masked == ""
            assert masked != value

    def test_masked_output_never_contains_middle_of_long_key(self):
        value = "sk-ant-api03-THISISASECRETMIDDLEPORTION-abcd"
        masked = mask(value)
        assert "THISISASECRETMIDDLEPORTION" not in masked


# ---------------------------------------------------------------------------
# resolve_provider_key precedence
# ---------------------------------------------------------------------------


class TestResolveProviderKeyPrecedence:
    def test_explicit_conf_key_wins_over_keys_file(self):
        conf = {"api_key": "explicit-resolved-key"}
        keys = {"openrouter": "keys-file-key"}
        assert resolve_provider_key("openrouter", conf, keys) == "explicit-resolved-key"

    def test_keys_file_used_when_conf_has_no_key(self):
        conf = {"endpoint": "https://openrouter.ai/api/v1"}
        keys = {"openrouter": "keys-file-key"}
        assert resolve_provider_key("openrouter", conf, keys) == "keys-file-key"

    def test_empty_string_when_neither_present(self):
        conf = {}
        keys = {}
        assert resolve_provider_key("openrouter", conf, keys) == ""

    def test_unresolved_env_placeholder_falls_through_to_keys_file(self):
        # config.py's expand_env_vars leaves an undefined ${VAR} literal —
        # that must NOT be treated as "an explicit key is configured".
        conf = {"api_key": "${OPENROUTER_API_KEY}"}
        keys = {"openrouter": "keys-file-key"}
        assert resolve_provider_key("openrouter", conf, keys) == "keys-file-key"

    def test_unresolved_env_placeholder_with_no_keys_file_entry_is_empty(self):
        conf = {"api_key": "${OPENROUTER_API_KEY}"}
        assert resolve_provider_key("openrouter", conf, {}) == ""

    def test_looks_up_by_provider_name_not_first_entry(self):
        conf = {}
        keys = {"openai": "openai-key", "anthropic": "anthropic-key"}
        assert resolve_provider_key("anthropic", conf, keys) == "anthropic-key"
        assert resolve_provider_key("gemini", conf, keys) == ""

    def test_none_provider_conf_does_not_raise(self):
        assert resolve_provider_key("openrouter", None, {"openrouter": "k"}) == "k"

    def test_none_keys_does_not_raise(self):
        assert resolve_provider_key("openrouter", {}, None) == ""


# ---------------------------------------------------------------------------
# /setkey CLI command (handle_slash_command)
# ---------------------------------------------------------------------------


@pytest.fixture
def console():
    return Console(force_terminal=True, width=120)


@pytest.fixture
def context():
    return Context()


@pytest.fixture
def config():
    return {
        "model": {
            "endpoint": "http://localhost:11434",
            "name": "test-model",
            "temperature": 0.7,
            "max_tokens": 4096,
            "context_window": 32768,
            "api_key": "",
            "provider": "ollama",
        },
        "permissions": {"mode": "auto", "always_allow": []},
    }


@pytest.fixture
def skills():
    return SkillRegistry()


@pytest.fixture
def model():
    from unittest.mock import MagicMock
    m = MagicMock(spec=ModelClient)
    m.total_input_tokens = 0
    m.total_output_tokens = 0
    return m


@pytest.fixture
def permissions():
    return PermissionManager(mode="auto")


def _patch_keys_and_global_config(monkeypatch, tmp_path):
    """Redirect BOTH the keys store and the global config.yaml to tmp_path —
    the two on-disk side effects /setkey can have. Never touches the real
    ~/.spark/keys or ~/.spark/config.yaml."""
    monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / "keys"))
    monkeypatch.setattr("spark_code.config.GLOBAL_CONFIG_FILE", tmp_path / "config.yaml")
    monkeypatch.setattr("spark_code.config.GLOBAL_CONFIG_DIR", tmp_path)


class TestSetKeyCommandNoArgs:
    def test_lists_presets_without_prompting(self, context, config, skills, model, permissions):
        import io
        import unittest.mock as mock

        import spark_code.cli as cli_mod

        def _boom(*a, **k):
            raise AssertionError("Prompt.ask should not run for bare /setkey")

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=200)
        with mock.patch.object(cli_mod.Prompt, "ask", _boom):
            result = cli_mod.handle_slash_command(
                "/setkey", context, console, config, skills, model,
                permissions=permissions,
            )
        assert result is None
        output = buf.getvalue()
        # All four presets must be listed by name — pins real preset-listing
        # behavior down (not just "any command handled without crashing").
        for name in PROVIDER_PRESETS:
            assert name in output


class TestSetKeyCommandUnknownProvider:
    def test_unknown_provider_reports_error_without_prompting(
        self, context, config, skills, model, permissions
    ):
        import io
        import unittest.mock as mock

        import spark_code.cli as cli_mod

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=200)

        def _boom(*a, **k):
            raise AssertionError("Prompt.ask should not run for an unknown provider")

        with mock.patch.object(cli_mod.Prompt, "ask", _boom):
            result = cli_mod.handle_slash_command(
                "/setkey nonexistent-provider", context, console, config, skills, model,
                permissions=permissions,
            )
        assert result is None
        output = buf.getvalue()
        assert "nonexistent-provider" in output
        assert "Unknown" in output or "unknown" in output


class TestSetKeyCommandSavesDummyKey:
    def test_setkey_openrouter_dummy_writes_chmod600_and_masks_output(
        self, console, context, config, skills, model, permissions, tmp_path, monkeypatch, capsys
    ):
        import unittest.mock as mock

        import spark_code.cli as cli_mod

        _patch_keys_and_global_config(monkeypatch, tmp_path)

        with mock.patch.object(cli_mod.Prompt, "ask", lambda *a, **k: DUMMY_OPENROUTER_KEY):
            result = cli_mod.handle_slash_command(
                "/setkey openrouter", context, console, config, skills, model,
                permissions=permissions,
            )
        assert result is None

        # Keys file written, chmod 600, contains the dummy value.
        path = tmp_path / "keys"
        assert path.exists()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600
        saved = json.loads(path.read_text())
        assert saved["openrouter"] == DUMMY_OPENROUTER_KEY

        # A provider block was added to the in-memory config from the preset.
        assert "openrouter" in config.get("providers", {})
        assert config["providers"]["openrouter"]["endpoint"] == PROVIDER_PRESETS["openrouter"]["endpoint"]
        assert config["providers"]["openrouter"]["model"] == PROVIDER_PRESETS["openrouter"]["model"]
        # The provider block must NOT contain the raw key.
        assert DUMMY_OPENROUTER_KEY not in json.dumps(config["providers"]["openrouter"])

    def test_full_key_never_printed_to_console(
        self, context, config, skills, model, permissions, tmp_path, monkeypatch
    ):
        import io
        import unittest.mock as mock

        import spark_code.cli as cli_mod

        _patch_keys_and_global_config(monkeypatch, tmp_path)

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=200)

        with mock.patch.object(cli_mod.Prompt, "ask", lambda *a, **k: DUMMY_ANTHROPIC_KEY):
            cli_mod.handle_slash_command(
                "/setkey anthropic", context, console, config, skills, model,
                permissions=permissions,
            )

        output = buf.getvalue()
        assert DUMMY_ANTHROPIC_KEY not in output
        # The masked form (or at least the first-3 prefix) should appear so
        # the user gets *some* confirmation of what was saved.
        assert DUMMY_ANTHROPIC_KEY[:3] in output

    def test_empty_key_entry_saves_nothing(
        self, console, context, config, skills, model, permissions, tmp_path, monkeypatch
    ):
        import unittest.mock as mock

        import spark_code.cli as cli_mod

        _patch_keys_and_global_config(monkeypatch, tmp_path)

        with mock.patch.object(cli_mod.Prompt, "ask", lambda *a, **k: "   "):
            result = cli_mod.handle_slash_command(
                "/setkey gemini", context, console, config, skills, model,
                permissions=permissions,
            )
        assert result is None
        assert not (tmp_path / "keys").exists()
        assert "gemini" not in config.get("providers", {})

    def test_existing_provider_block_is_not_overwritten(
        self, console, context, config, skills, model, permissions, tmp_path, monkeypatch
    ):
        import unittest.mock as mock

        import spark_code.cli as cli_mod

        _patch_keys_and_global_config(monkeypatch, tmp_path)
        config["providers"] = {
            "openrouter": {"endpoint": "https://custom.example/v1", "model": "custom-model"},
        }

        with mock.patch.object(cli_mod.Prompt, "ask", lambda *a, **k: DUMMY_OPENROUTER_KEY):
            cli_mod.handle_slash_command(
                "/setkey openrouter", context, console, config, skills, model,
                permissions=permissions,
            )

        # Key still saved to the keys file...
        assert json.loads((tmp_path / "keys").read_text())["openrouter"] == DUMMY_OPENROUTER_KEY
        # ...but the pre-existing custom provider block is untouched.
        assert config["providers"]["openrouter"]["endpoint"] == "https://custom.example/v1"
        assert config["providers"]["openrouter"]["model"] == "custom-model"


class TestSetKeyPausesEscWatcher:
    """Mirrors test_permissions.py's TestPromptSuspendsEscWatcher — the masked
    /setkey prompt must suspend any active Esc watcher around Prompt.ask, the
    same way the permission prompt does, or the watcher's background thread
    steals the user's keystrokes."""

    def test_setkey_pauses_and_resumes_watcher_around_ask(
        self, console, context, config, skills, model, permissions, tmp_path, monkeypatch
    ):
        import unittest.mock as mock

        import spark_code.cli as cli_mod
        from spark_code.ui import esc_watcher

        _patch_keys_and_global_config(monkeypatch, tmp_path)

        events = []

        class _FakeWatcher:
            def pause(self):
                events.append("pause")

            def resume(self):
                events.append("resume")

        fake = _FakeWatcher()
        esc_watcher._register(fake)

        def _fake_ask(*a, **k):
            events.append("ask")
            return DUMMY_OPENROUTER_KEY

        try:
            with mock.patch.object(cli_mod.Prompt, "ask", _fake_ask):
                cli_mod.handle_slash_command(
                    "/setkey openrouter", context, console, config, skills, model,
                    permissions=permissions,
                )
        finally:
            esc_watcher._unregister(fake)

        assert events == ["pause", "ask", "resume"]


# ---------------------------------------------------------------------------
# Sanity: ModelClient can be built from a resolved dummy key (endpoint/header
# shape only — no real network call in this unit test).
# ---------------------------------------------------------------------------


class TestModelClientPicksUpResolvedKey:
    @pytest.mark.asyncio
    async def test_dummy_key_lands_in_authorization_header(self):
        preset = PROVIDER_PRESETS["openrouter"]
        client = ModelClient(
            endpoint=preset["endpoint"],
            model=preset["model"],
            api_key=DUMMY_OPENROUTER_KEY,
            provider="openrouter",
        )
        try:
            assert client._client.headers["Authorization"] == f"Bearer {DUMMY_OPENROUTER_KEY}"
            assert client.api_url == "https://openrouter.ai/api/v1/chat/completions"
        finally:
            await client.close()
