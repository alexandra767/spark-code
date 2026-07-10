"""Custom tool definitions via /teach — user-defined tools the model can call.

Persisted to ~/.spark/custom_tools.json.
"""

import asyncio
import json
import logging
import os
import shlex

from .tools.base import Tool

logger = logging.getLogger(__name__)

# Built-in tool names a custom /teach tool must never shadow — a custom tool
# named e.g. "read_file" or "grep" would inherit the built-in's always_allow
# grant and silently run arbitrary shell under a trusted name.
#
# This static set is the DEFAULT / safety net. It had drifted (missing every
# tool added after the Phase-2 era — stack_info, the android_*/ios_* device
# tools, the appstore_*/play_* store tools, code_search, dispatch_agent). The
# authoritative reserved set is derived from the LIVE registry at construction
# time in cli.py (`reserved_names=set(registry.names()) | BUILTIN_TOOL_NAMES`)
# so it can never drift again even if a new tool is added without updating this
# list; keep this list roughly current as the belt to that suspenders.
BUILTIN_TOOL_NAMES = frozenset({
    # core file / shell / search
    "read_file", "write_file", "edit_file", "bash", "glob", "grep",
    "list_dir", "web_search", "web_fetch", "rag_search", "code_search",
    "stack_info",
    # orchestration
    "spawn_worker", "wait_for_workers", "send_message", "todo_write",
    "dispatch_agent", "browse_web",
    # device / vision
    "android_control", "android_screenshot", "ios_launch", "ios_screenshot",
    "watch_phone", "simulator_screenshot",
    # app store / play
    "appstore_status", "appstore_submit", "appstore_build_upload",
    "play_status", "play_publish",
})


class CustomTool(Tool):
    """A user-defined tool created via /teach."""

    def __init__(self, name: str, description: str, command: str,
                 parameters: dict | None = None):
        self._name = name
        self._description = description
        self._command = command
        self._parameters = parameters or {
            "type": "object",
            "properties": {
                "args": {
                    "type": "string",
                    "description": "Optional arguments to pass to the command",
                },
            },
            "required": [],
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    @property
    def is_read_only(self) -> bool:
        return False

    async def execute(self, args: str = "", **kwargs) -> str:
        """Execute the custom command."""
        command = self._command
        # Shell-quote the substituted value so `{args}` can't inject extra
        # commands (e.g. args="; rm -rf ~").
        command = command.replace("{args}", shlex.quote(args) if args else "")

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=120
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            if process.returncode != 0:
                output += f"\n\nExit code: {process.returncode}"
            return output or f"Command completed (exit code {process.returncode})"
        except asyncio.TimeoutError:
            return "Error: Command timed out after 120s"
        except Exception as e:
            return f"Error: {e}"

    def to_dict(self) -> dict:
        return {
            "name": self._name,
            "description": self._description,
            "command": self._command,
        }


class CustomToolRegistry:
    """Persists custom tools to disk."""

    def __init__(self, path: str = "~/.spark/custom_tools.json",
                 reserved_names: set[str] | frozenset[str] | None = None):
        self.path = os.path.expanduser(path)
        # Names a custom tool must not collide with (defaults to the built-ins).
        self.reserved_names = frozenset(
            reserved_names if reserved_names is not None else BUILTIN_TOOL_NAMES
        )
        self._tools: dict[str, CustomTool] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                name = entry["name"]
                # Skip (don't crash) any persisted tool that shadows a built-in.
                if name in self.reserved_names:
                    logger.warning(
                        "Skipping custom tool %r: name collides with a built-in tool.",
                        name,
                    )
                    continue
                tool = CustomTool(
                    name=name,
                    description=entry["description"],
                    command=entry["command"],
                )
                self._tools[tool.name] = tool
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = [t.to_dict() for t in self._tools.values()]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add(self, name: str, description: str, command: str) -> CustomTool:
        """Create and register a custom tool.

        Raises ValueError if *name* collides with a built-in tool name (which
        would let the custom tool inherit the built-in's always_allow grant).
        """
        if name in self.reserved_names:
            raise ValueError(
                f"'{name}' is a built-in tool name and cannot be used for a "
                f"custom tool. Choose a different name."
            )
        tool = CustomTool(name=name, description=description, command=command)
        self._tools[name] = tool
        self._save()
        return tool

    def remove(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            self._save()
            return True
        return False

    def get(self, name: str) -> CustomTool | None:
        return self._tools.get(name)

    def all(self) -> list[CustomTool]:
        return list(self._tools.values())

    @property
    def count(self) -> int:
        return len(self._tools)
