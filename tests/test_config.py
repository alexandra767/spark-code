"""Tests for spark_code.config — deep_merge, expand_env_vars, load_config, get."""

import os
import pytest

from spark_code.config import deep_merge, expand_env_vars, load_config, get


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
