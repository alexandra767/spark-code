"""Configuration loading for Spark Code."""

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = {
    "model": {
        "endpoint": "http://localhost:11434",
        "name": "qwen2.5:72b",
        "temperature": 0.7,
        "max_tokens": 4096,
        "context_window": 32768,
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


def ensure_dirs():
    """Create necessary directories."""
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (GLOBAL_CONFIG_DIR / "memory").mkdir(exist_ok=True)
    (GLOBAL_CONFIG_DIR / "skills").mkdir(exist_ok=True)
