"""Configuration loading for Spark Code."""

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = {
    "model": {
        "endpoint": "http://localhost:11434",
        "name": "qwen3.5:122b",
        "temperature": 0.7,
        "max_tokens": 8192,
        "context_window": 262144,
    },
    "permissions": {
        "mode": "ask",  # ask | auto | trust
        "always_allow": ["read_file", "glob", "grep", "list_dir"],
    },
    "ui": {
        "theme": "dark",
        "syntax_highlighting": True,
        "show_token_count": True,
        "markdown_rendering": True,
        "show_diffs": True,
    },
    "mcp_servers": {},
    "memory": {
        "enabled": True,
        "global_path": "~/.spark/memory/",
    },
}

GLOBAL_CONFIG_DIR = Path.home() / ".spark"
GLOBAL_CONFIG_FILE = GLOBAL_CONFIG_DIR / "config.yaml"
PROJECT_CONFIG_DIR = ".spark"
PROJECT_CONFIG_FILE = os.path.join(PROJECT_CONFIG_DIR, "config.yaml")


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def expand_env_vars(config: dict) -> dict:
    """Expand ${VAR} references in string values."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = expand_env_vars(value)
        elif isinstance(value, str) and "${" in value:
            for env_key, env_val in os.environ.items():
                value = value.replace(f"${{{env_key}}}", env_val)
            result[key] = value
        else:
            result[key] = value
    return result


def resolve_provider(config: dict, provider_name: str | None = None) -> dict:
    """Resolve active provider into model config."""
    providers = config.get("providers", {})
    if not providers:
        return config  # Old-style config without providers

    # Pick provider
    name = provider_name or config.get("active_provider", "ollama")
    if name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    provider_conf = providers[name]

    # Map provider config to model config
    config["model"] = {
        "endpoint": provider_conf.get("endpoint", "http://localhost:11434"),
        "name": provider_conf.get("model", "qwen2.5:72b"),
        "temperature": provider_conf.get("temperature", 0.7),
        "max_tokens": provider_conf.get("max_tokens", 4096),
        "context_window": provider_conf.get("context_window", 32768),
        "api_key": provider_conf.get("api_key", ""),
        "provider": name,
    }
    resolved = config["model"]
    resolved["system_prompt"] = provider_conf.get("system_prompt", "")
    resolved["cost_per_million_input"] = provider_conf.get("cost_per_million_input", 0)
    resolved["cost_per_million_output"] = provider_conf.get("cost_per_million_output", 0)
    return config


def load_config(project_dir: str | None = None,
                provider: str | None = None) -> dict:
    """Load config from defaults → global → project, with env var expansion."""
    config = DEFAULT_CONFIG.copy()

    # Load global config
    if GLOBAL_CONFIG_FILE.exists():
        with open(GLOBAL_CONFIG_FILE) as f:
            global_conf = yaml.safe_load(f) or {}
        config = deep_merge(config, global_conf)

    # Load project config
    if project_dir:
        project_conf_path = os.path.join(project_dir, PROJECT_CONFIG_FILE)
        if os.path.exists(project_conf_path):
            with open(project_conf_path) as f:
                project_conf = yaml.safe_load(f) or {}
            config = deep_merge(config, project_conf)

    config = expand_env_vars(config)

    # Resolve provider into model config
    config = resolve_provider(config, provider)

    return config


def get(config: dict, *keys: str, default: Any = None) -> Any:
    """Get nested config value. Usage: get(config, 'model', 'endpoint')"""
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def set_config(config: dict, key_path: str, value: str) -> tuple[bool, str]:
    """Set a nested config value and save to global config.

    key_path: dot-separated path like "model.temperature" or "permissions.mode"
    value: string value (auto-converted to int/float/bool where appropriate)

    Returns (success, message).
    """
    keys = key_path.split(".")
    if len(keys) < 2:
        return False, f"Invalid key: {key_path} (use dot notation like model.temperature)"

    # Auto-convert value types
    converted: str | int | float | bool = value
    if value.lower() in ("true", "false"):
        converted = value.lower() == "true"
    else:
        try:
            converted = int(value)
        except ValueError:
            try:
                converted = float(value)
            except ValueError:
                pass  # Keep as string

    # Update in-memory config
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    old_value = current.get(keys[-1])
    current[keys[-1]] = converted

    # Save to global config file (merge with existing)
    try:
        existing = {}
        if GLOBAL_CONFIG_FILE.exists():
            with open(GLOBAL_CONFIG_FILE) as f:
                existing = yaml.safe_load(f) or {}

        # Set the value in the file config too
        file_current = existing
        for key in keys[:-1]:
            if key not in file_current or not isinstance(file_current[key], dict):
                file_current[key] = {}
            file_current = file_current[key]
        file_current[keys[-1]] = converted

        GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(GLOBAL_CONFIG_FILE, "w") as f:
            yaml.dump(existing, f, default_flow_style=False)

        return True, f"{key_path}: {old_value} → {converted}"
    except Exception as e:
        return False, f"Failed to save config: {e}"


def ensure_dirs():
    """Create necessary directories."""
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (GLOBAL_CONFIG_DIR / "memory").mkdir(exist_ok=True)
    (GLOBAL_CONFIG_DIR / "skills").mkdir(exist_ok=True)
