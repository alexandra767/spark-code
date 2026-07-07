"""Configuration loading for Spark Code."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .permissions import resolve_mode_name

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

DEFAULT_CONFIG = {
    "model": {
        "endpoint": "http://localhost:11434",
        "name": "qwen3.5:122b",
        "temperature": 0.7,
        "max_tokens": 4096,
        "context_window": 262144,
    },
    "permissions": {
        "mode": "ask",  # ask | auto | trust
        "always_allow": ["read_file", "glob", "grep", "list_dir", "todo_write"],
    },
    "ui": {
        "theme": "dark",
        "syntax_highlighting": True,
        "show_token_count": True,
        "markdown_rendering": True,
        "show_diffs": True,
        # "auto" (toolbar when CPR works, inline one-line fallback when it
        # doesn't) | "toolbar" (always) | "inline" (always) | "off" (neither).
        # See spark_code/ui/input.py's terminal_supports_cpr investigation
        # notes (Phase 3 Task 3) for why "auto" needs a runtime confirmation
        # signal, not just a pre-flight guess.
        "statusline": "auto",
        # "auto" (detect Cursor/VS Code/Xcode via spark_code.editor.detect_editor)
        # | "cursor" | "code" | "xed" (force one) | "none" (disable /open +
        # OSC 8 file-link styling entirely). See spark_code/editor.py.
        "editor": "auto",
    },
    "mcp_servers": {},
    "memory": {
        "enabled": True,
        "global_path": "~/.spark/memory/",
        # Cap on the injected Claude Code MEMORY.md bridge content (chars).
        # 0 disables the bridge entirely. See spark_code/memory.py.
        "bridge_budget_chars": 4000,
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


def _expand_env_str(value: str) -> str:
    """Expand ${VAR} references in a single string.

    Undefined variables are left literal (unchanged) and a warning is logged.
    """
    def _repl(match: "re.Match") -> str:
        var = match.group(1)
        if var in os.environ:
            return os.environ[var]
        logger.warning(
            "Undefined environment variable ${%s} — left literal in config", var
        )
        return match.group(0)

    return _ENV_VAR_RE.sub(_repl, value)


def expand_env_vars(config: Any) -> Any:
    """Expand ${VAR} references in string values, recursing dicts AND lists.

    Recursing into lists matters for things like mcp_servers.*.args, whose
    values are lists — previously ${VAR} inside a list item was passed literally.
    """
    if isinstance(config, dict):
        return {key: expand_env_vars(value) for key, value in config.items()}
    if isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    if isinstance(config, str) and "${" in config:
        return _expand_env_str(config)
    return config


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
        # Keep in sync with DEFAULT_CONFIG["model"]["name"].
        "name": provider_conf.get("model", DEFAULT_CONFIG["model"]["name"]),
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
    resolved["worker_model"] = provider_conf.get("worker_model", "")
    resolved["worker_endpoint"] = provider_conf.get("worker_endpoint", "")
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

    # Normalize permissions.mode: accept Claude Code's mode names
    # (default/acceptEdits/bypassPermissions) in the config file too,
    # case-insensitively. Unrecognized values are left as-is (unchanged
    # behavior — PermissionManager already treats any non-trust/non-auto
    # mode string like "ask", so a typo here doesn't newly break anything).
    perms = config.get("permissions")
    if isinstance(perms, dict) and isinstance(perms.get("mode"), str):
        resolved_mode = resolve_mode_name(perms["mode"])
        if resolved_mode is not None:
            perms["mode"] = resolved_mode

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


def _project_config_file(project_dir: str | None) -> Path:
    """Path to the project config file (relative to cwd if project_dir is None)."""
    if project_dir:
        return Path(project_dir) / PROJECT_CONFIG_FILE
    return Path(PROJECT_CONFIG_FILE)


def _key_in_config(data: dict, keys: list[str]) -> bool:
    """Return True if the nested key path exists in data."""
    current: Any = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False
    return True


def set_config(config: dict, key_path: str, value: str,
               project_dir: str | None = None,
               scope: str = "auto") -> tuple[bool, str]:
    """Set a nested config value and persist it to the correct scope.

    key_path: dot-separated path like "model.temperature" or "permissions.mode"
    value: string value (auto-converted to int/float/bool where appropriate)
    scope: "auto" | "global" | "project".
        - "auto" (default): write to the PROJECT config if a project config
          file exists AND already defines this key (i.e. it overrides global);
          otherwise write to the GLOBAL config. This fixes the bug where a
          project-overridden key was written to the global file and the change
          silently had no effect (project config wins on the next load).
        - "global"/"project": force the target scope.

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

    # Decide which file to persist to.
    proj_path = _project_config_file(project_dir)
    if scope == "project":
        target_path, scope_label = proj_path, "project"
    elif scope == "global":
        target_path, scope_label = GLOBAL_CONFIG_FILE, "global"
    else:  # auto
        proj_existing = {}
        if proj_path.exists():
            try:
                with open(proj_path) as f:
                    proj_existing = yaml.safe_load(f) or {}
            except Exception:
                proj_existing = {}
        if proj_path.exists() and _key_in_config(proj_existing, keys):
            target_path, scope_label = proj_path, "project"
        else:
            target_path, scope_label = GLOBAL_CONFIG_FILE, "global"

    # Save to the chosen file (merge with existing)
    try:
        existing = {}
        if target_path.exists():
            with open(target_path) as f:
                existing = yaml.safe_load(f) or {}

        # Set the value in the file config too
        file_current = existing
        for key in keys[:-1]:
            if key not in file_current or not isinstance(file_current[key], dict):
                file_current[key] = {}
            file_current = file_current[key]
        file_current[keys[-1]] = converted

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False)

        return True, f"{key_path}: {old_value} → {converted} (saved to {scope_label} config)"
    except Exception as e:
        return False, f"Failed to save config: {e}"


def ensure_dirs():
    """Create necessary directories."""
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (GLOBAL_CONFIG_DIR / "memory").mkdir(exist_ok=True)
    (GLOBAL_CONFIG_DIR / "skills").mkdir(exist_ok=True)
