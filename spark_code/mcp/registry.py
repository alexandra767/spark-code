"""MCP server discovery and registration."""

import os
import re
from pathlib import Path

import yaml

# Matches ${VAR} and $VAR references inside config strings.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def find_mcp_configs() -> dict:
    """Find MCP server configs from global and project configs."""
    configs = {}

    # Global MCP config
    global_mcp = Path.home() / ".spark" / "mcp.yaml"
    if global_mcp.exists():
        with open(global_mcp) as f:
            data = yaml.safe_load(f) or {}
        configs.update(data.get("mcpServers", {}))

    # Project MCP config
    project_mcp = Path(".spark") / "mcp.yaml"
    if project_mcp.exists():
        with open(project_mcp) as f:
            data = yaml.safe_load(f) or {}
        configs.update(data.get("mcpServers", {}))

    return configs


def _expand_str(value: str) -> str:
    """Expand ${VAR} / $VAR references in a string.

    Undefined variables are left as the literal placeholder (e.g. ``${MISSING}``
    stays ``${MISSING}``) so a misconfiguration is visible rather than silently
    turning into an empty string.
    """
    def repl(match: "re.Match") -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, match.group(0))

    return _VAR_RE.sub(repl, value)


def expand_mcp_env(config: dict) -> dict:
    """Return a copy of ``config`` with environment variables expanded in ``env``.

    Handles ``{TOKEN: "${GITHUB_TOKEN}"}`` style references (both ``${VAR}`` and
    ``$VAR`` forms, including embedded ones). The original config is not mutated.
    Undefined variables are left as their literal placeholder.
    """
    result = dict(config)
    env = result.get("env")
    if isinstance(env, dict):
        result["env"] = {
            key: (_expand_str(value) if isinstance(value, str) else value)
            for key, value in env.items()
        }
    return result
