"""MCP server discovery and registration."""

import os
import yaml
from pathlib import Path


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


def expand_mcp_env(config: dict) -> dict:
    """Expand environment variables in MCP config."""
    result = config.copy()
    if "env" in result:
        expanded = {}
        for key, value in result["env"].items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_key = value[2:-1]
                expanded[key] = os.environ.get(env_key, value)
            else:
                expanded[key] = value
        result["env"] = expanded
    return result
