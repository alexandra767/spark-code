"""Base tool interface for Spark Code."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calling."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description for the model."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for parameters."""

    @property
    def requires_permission(self) -> bool:
        """Whether this tool needs user approval."""
        return True

    @property
    def is_read_only(self) -> bool:
        """Whether this tool only reads (doesn't modify anything)."""
        return False

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool and return result as string."""

    def to_schema(self) -> dict:
        """Convert to tool schema for the model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())
