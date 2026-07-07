"""Live, session-scoped todo list — Spark Code's TodoWrite equivalent.

On a local 80B at 32K, a visible checklist is the main defense against
wandering off-task on multi-step work: the model sets exactly one item
`in_progress`, marks it `completed` when done, and the live panel keeps the
user (and the next round's prompt) honest about what's actually finished.

In-memory only, per session — unlike the deleted global-file `tools/todo.py`
(``~/.spark/todos.json``), nothing here is persisted to disk. A fresh
session always starts with an empty list.
"""

from .tools.base import Tool

VALID_STATUSES = {"pending", "in_progress", "completed"}
MAX_ITEMS = 20


class TodoList:
    """In-memory todo list with full-replace ("set_items") semantics.

    Mirrors Claude Code's TodoWrite: every call replaces the whole list
    rather than patching individual items. Validation failures are returned
    as plain error strings (never raised) so a tool call can surface them to
    the model as an ordinary tool result.
    """

    def __init__(self):
        self._items: list[dict] = []

    @property
    def items(self) -> list[dict]:
        """Defensive copy — callers can't mutate internal state through it."""
        return list(self._items)

    def set_items(self, items: list[dict]) -> str:
        """Replace the whole list. Returns a confirmation string, or an
        error string (starting with "Error") if validation fails — in
        which case the existing list is left untouched."""
        if not isinstance(items, list):
            return "Error: items must be a list of {content, status} objects."

        if len(items) > MAX_ITEMS:
            return f"Error: too many items ({len(items)}); max is {MAX_ITEMS}."

        normalized: list[dict] = []
        in_progress_count = 0
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                return f"Error: item {i} must be an object with 'content' and 'status'."

            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                return f"Error: item {i} is missing a non-empty 'content' string."

            status = item.get("status")
            if status not in VALID_STATUSES:
                return (f"Error: item {i} has invalid status {status!r}; "
                        f"must be one of {sorted(VALID_STATUSES)}.")

            if status == "in_progress":
                in_progress_count += 1

            normalized.append({"content": content, "status": status})

        if in_progress_count > 1:
            return (f"Error: at most one item may be in_progress "
                    f"(found {in_progress_count}).")

        self._items = normalized
        completed = sum(1 for i in normalized if i["status"] == "completed")
        return f"Todo list updated: {len(normalized)} items ({completed} completed)."


class TodoWriteTool(Tool):
    """Tool the model calls to create/update the live todo checklist.

    Never touches disk (``is_read_only``) and never needs user approval
    (``requires_permission = False``) — it's pure session bookkeeping, same
    as Claude Code's TodoWrite.
    """

    name = "todo_write"
    description = (
        "Create or update your live todo checklist for this session (full "
        "replace — send the WHOLE list every time, not just the changed "
        "item). Use it for any task with 3 or more steps: set one item to "
        "in_progress when you start it, mark it completed when done, and "
        "keep exactly one item in_progress at a time."
    )
    is_read_only = True
    requires_permission = False

    def __init__(self, todo_list: TodoList):
        self._todo_list = todo_list

    @property
    def items(self) -> list[dict]:
        """Current items — used by the renderer to draw the live panel
        after a successful call, without needing its own TodoList handle."""
        return self._todo_list.items

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": ("The FULL list of todo items, replacing "
                                    "whatever list existed before."),
                    "maxItems": MAX_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Short description of the step.",
                            },
                            "status": {
                                "type": "string",
                                "enum": sorted(VALID_STATUSES),
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        }

    async def execute(self, items: list[dict] | None = None, **kwargs) -> str:
        return self._todo_list.set_items(items if items is not None else [])
