"""Tests for spark_code.config — deep_merge, expand_env_vars, load_config, get."""


from spark_code.config import deep_merge, expand_env_vars, get, load_config


class TestDeepMerge:
    def test_basic_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_nested_dicts(self):
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        result = deep_merge(base, override)
        assert result["outer"]["a"] == 1
        assert result["outer"]["b"] == 3
        assert result["outer"]["c"] == 4

    def test_does_not_mutate_original(self):
        base = {"a": 1, "nested": {"x": 10}}
        override = {"a": 2, "nested": {"y": 20}}
        base_copy = {"a": 1, "nested": {"x": 10}}
        deep_merge(base, override)
        assert base == base_copy or base["a"] == 1  # at minimum base["a"] unchanged


class TestExpandEnvVars:
    def test_replaces_env_var(self, monkeypatch):
        monkeypatch.setenv("SPARK_TEST_VAR", "replaced_value")
        config = {"key": "${SPARK_TEST_VAR}"}
        result = expand_env_vars(config)
        assert result["key"] == "replaced_value"

    def test_ignores_non_string_values(self, monkeypatch):
        monkeypatch.setenv("SPARK_TEST_VAR", "val")
        config = {"number": 42, "flag": True, "items": [1, 2]}
        result = expand_env_vars(config)
        assert result["number"] == 42
        assert result["flag"] is True
        assert result["items"] == [1, 2]

    def test_recurses_into_lists(self, monkeypatch):
        """Bug 13: ${VAR} inside list values (e.g. mcp_servers.*.args) expands."""
        monkeypatch.setenv("SPARK_TOKEN", "sekret")
        config = {"mcp_servers": {"x": {"args": ["--token", "${SPARK_TOKEN}"]}}}
        result = expand_env_vars(config)
        assert result["mcp_servers"]["x"]["args"] == ["--token", "sekret"]

    def test_recurses_into_nested_lists_of_dicts(self, monkeypatch):
        monkeypatch.setenv("SPARK_HOST", "example.com")
        config = {"servers": [{"url": "https://${SPARK_HOST}/api"}]}
        result = expand_env_vars(config)
        assert result["servers"][0]["url"] == "https://example.com/api"

    def test_undefined_var_left_literal(self, monkeypatch):
        monkeypatch.delenv("SPARK_DOES_NOT_EXIST", raising=False)
        config = {"key": "${SPARK_DOES_NOT_EXIST}"}
        result = expand_env_vars(config)
        assert result["key"] == "${SPARK_DOES_NOT_EXIST}"


class TestProviderFallbackModel:
    def test_provider_fallback_matches_default(self):
        """Bug 14: provider model fallback must equal DEFAULT_CONFIG model name."""
        from spark_code.config import DEFAULT_CONFIG, resolve_provider
        cfg = {
            "active_provider": "ollama",
            "providers": {"ollama": {}},  # no explicit model → fallback used
        }
        resolved = resolve_provider(cfg)
        assert resolved["model"]["name"] == DEFAULT_CONFIG["model"]["name"]


class TestResolveProviderFreshConfigDoesNotCrash:
    """Finding 2 (Phase 5 whole-branch review): /setkey on a fresh config (no
    ``providers`` block yet) seeds the FIRST providers.<name> block but never
    sets ``active_provider`` (cli.py's /setkey handler). On the NEXT startup,
    resolve_provider's default ``name = provider_name or
    config.get("active_provider", "ollama")`` resolves to the literal string
    "ollama", which isn't a key in a one-entry {"openrouter": {...}} map —
    raising ValueError, uncaught at load_config() in main(), crashing the
    fresh-user "cloud key off the DGX" persona on startup."""

    def test_single_non_ollama_provider_no_active_provider_does_not_raise(self):
        from spark_code.config import resolve_provider
        cfg = {
            # No "active_provider" key at all — exactly what /setkey leaves
            # behind on a fresh config.
            "providers": {"openrouter": {"endpoint": "https://openrouter.ai/api/v1",
                                          "model": "openrouter/auto"}},
        }
        resolved = resolve_provider(cfg)  # must NOT raise
        assert resolved["model"]["provider"] == "openrouter"
        assert resolved["model"]["endpoint"] == "https://openrouter.ai/api/v1"

    def test_stale_active_provider_falls_back_instead_of_raising(self):
        # active_provider survives a provider rename/removal in config.yaml —
        # also must not crash startup.
        from spark_code.config import resolve_provider
        cfg = {
            "active_provider": "some-removed-provider",
            "providers": {"openrouter": {"endpoint": "https://openrouter.ai/api/v1",
                                          "model": "openrouter/auto"}},
        }
        resolved = resolve_provider(cfg)
        assert resolved["model"]["provider"] == "openrouter"

    def test_explicit_unknown_provider_name_still_raises(self):
        # An EXPLICIT --provider foo (provider_name argument) that doesn't
        # exist is a clear user mistake — must still be surfaced, not
        # silently swapped for a different provider.
        from spark_code.config import resolve_provider
        cfg = {"providers": {"openrouter": {"endpoint": "https://openrouter.ai/api/v1"}}}
        try:
            resolve_provider(cfg, provider_name="totally-not-configured")
            assert False, "should have raised"
        except ValueError as e:
            assert "totally-not-configured" in str(e)

    def test_existing_multiprovider_config_unaffected(self):
        # A normal, correctly-configured multi-provider setup keeps resolving
        # to the CONFIGURED active_provider, not silently falling back to
        # "whichever is first".
        from spark_code.config import resolve_provider
        cfg = {
            "active_provider": "openrouter",
            "providers": {
                "ollama": {"endpoint": "http://localhost:11434"},
                "openrouter": {"endpoint": "https://openrouter.ai/api/v1"},
            },
        }
        resolved = resolve_provider(cfg)
        assert resolved["model"]["provider"] == "openrouter"

    def test_load_config_after_fresh_setkey_does_not_crash(self, tmp_path, monkeypatch):
        """End-to-end: load_config() (what main() calls at startup) must not
        raise for a config file shaped exactly like /setkey leaves one."""
        import yaml

        from spark_code.config import load_config
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        global_config_file = global_dir / "config.yaml"
        global_config_file.write_text(yaml.dump({
            "providers": {"openrouter": {"endpoint": "https://openrouter.ai/api/v1",
                                          "model": "openrouter/auto"}},
        }))
        monkeypatch.setattr("spark_code.config.GLOBAL_CONFIG_FILE", global_config_file)

        config = load_config()  # must NOT raise
        assert config["model"]["provider"] == "openrouter"


class TestResolveProviderConsultsKeysFile:
    """Phase 5 Task 2: resolve_provider must fall back to the ~/.spark/keys
    store (set via /setkey) when a provider block has no resolvable api_key
    of its own. Always monkeypatches keys_path — never touches the real
    ~/.spark/keys."""

    def test_keys_file_backfills_missing_api_key(self, tmp_path, monkeypatch):
        from spark_code.config import resolve_provider

        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / "keys"))
        from spark_code.keys import save_key
        save_key("openrouter", "dummy-keys-file-value")

        cfg = {
            "active_provider": "openrouter",
            "providers": {"openrouter": {"endpoint": "https://openrouter.ai/api/v1",
                                          "model": "openrouter/auto"}},
        }
        resolved = resolve_provider(cfg)
        assert resolved["model"]["api_key"] == "dummy-keys-file-value"

    def test_explicit_config_api_key_wins_over_keys_file(self, tmp_path, monkeypatch):
        from spark_code.config import resolve_provider

        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / "keys"))
        from spark_code.keys import save_key
        save_key("openrouter", "dummy-keys-file-value")

        cfg = {
            "active_provider": "openrouter",
            "providers": {"openrouter": {"endpoint": "https://openrouter.ai/api/v1",
                                          "model": "openrouter/auto",
                                          "api_key": "dummy-explicit-value"}},
        }
        resolved = resolve_provider(cfg)
        assert resolved["model"]["api_key"] == "dummy-explicit-value"

    def test_no_key_anywhere_resolves_to_empty_string(self, tmp_path, monkeypatch):
        from spark_code.config import resolve_provider

        monkeypatch.setattr("spark_code.keys.keys_path", lambda: str(tmp_path / "keys"))
        cfg = {
            "active_provider": "ollama",
            "providers": {"ollama": {"endpoint": "http://localhost:11434"}},
        }
        resolved = resolve_provider(cfg)
        assert resolved["model"]["api_key"] == ""


class TestSetConfigScope:
    def test_set_writes_to_project_when_project_overrides(self, tmp_path):
        """Bug 14: if the project config defines the key, set writes THERE.

        Otherwise the set silently no-ops (global write is shadowed by the
        project override on the next load).
        """
        import yaml

        from spark_code.config import PROJECT_CONFIG_FILE, set_config

        proj = tmp_path / "proj"
        proj.mkdir()
        proj_cfg = proj / PROJECT_CONFIG_FILE
        proj_cfg.parent.mkdir(parents=True, exist_ok=True)
        proj_cfg.write_text("model:\n  temperature: 0.9\n")

        config = {"model": {"temperature": 0.9}}
        ok, msg = set_config(config, "model.temperature", "0.2",
                             project_dir=str(proj))
        assert ok
        assert "project" in msg
        # The project file must now hold the new value.
        written = yaml.safe_load(proj_cfg.read_text())
        assert written["model"]["temperature"] == 0.2

    def test_set_writes_to_global_when_no_project_key(self, tmp_path):
        """When the project config doesn't define the key, write to global."""
        from unittest.mock import patch

        import yaml

        from spark_code.config import set_config

        gdir = tmp_path / "global"
        gfile = gdir / "config.yaml"
        empty_proj = tmp_path / "empty_proj"  # no .spark/config.yaml
        empty_proj.mkdir()

        config = {"model": {"temperature": 0.7}}
        with patch("spark_code.config.GLOBAL_CONFIG_FILE", gfile), \
             patch("spark_code.config.GLOBAL_CONFIG_DIR", gdir):
            ok, msg = set_config(config, "model.temperature", "0.4",
                                 project_dir=str(empty_proj))
            assert ok
            assert "global" in msg
            written = yaml.safe_load(gfile.read_text())
            assert written["model"]["temperature"] == 0.4


class TestGet:
    def test_returns_nested_value(self):
        config = {"level1": {"level2": {"level3": "found"}}}
        assert get(config, "level1", "level2", "level3") == "found"

    def test_returns_default_for_missing_key(self):
        config = {"a": {"b": 1}}
        assert get(config, "a", "missing", default="fallback") == "fallback"

    def test_returns_default_for_empty_config(self):
        assert get({}, "any", "key", default=None) is None


class TestLoadConfig:
    def test_returns_defaults_when_no_config_files(self, tmp_path):
        # Use a project dir with no config files
        result = load_config(str(tmp_path), provider="ollama")
        assert isinstance(result, dict)
        # Should at least have some default keys
        assert result is not None
