"""Tests for the live, session-scoped todo list (todo_write).

Unlike the deleted global-file `tools/todo.py` (~/.spark/todos.json), this
list is in-memory only: a fresh session always starts empty.
"""

import io

from rich.console import Console

from spark_code.todos import MAX_ITEMS, TodoList, TodoWriteTool
from spark_code.ui.output import render_todos


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=100)


# ---------------------------------------------------------------------------
# TodoList.set_items — full-replace semantics + validation
# ---------------------------------------------------------------------------


class TestTodoListSetItems:
    def test_replaces_wholesale(self):
        todos = TodoList()
        todos.set_items([{"content": "a", "status": "pending"}])
        todos.set_items([{"content": "b", "status": "pending"}])
        assert [i["content"] for i in todos.items] == ["b"]

    def test_empty_list_clears_everything(self):
        todos = TodoList()
        todos.set_items([{"content": "a", "status": "pending"}])
        result = todos.set_items([])
        assert "Error" not in result
        assert todos.items == []

    def test_returns_confirmation_string_not_none(self):
        todos = TodoList()
        result = todos.set_items([{"content": "a", "status": "completed"}])
        assert isinstance(result, str)
        assert "Error" not in result

    def test_one_in_progress_is_allowed(self):
        todos = TodoList()
        result = todos.set_items([
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "in_progress"},
            {"content": "c", "status": "pending"},
        ])
        assert "Error" not in result
        assert todos.items[1]["status"] == "in_progress"

    def test_two_in_progress_rejected_as_error_string(self):
        todos = TodoList()
        result = todos.set_items([
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ])
        assert isinstance(result, str)
        assert "Error" in result
        # Rejected update must not mutate existing state.
        assert todos.items == []

    def test_max_20_items_accepted(self):
        todos = TodoList()
        items = [{"content": f"item {i}", "status": "pending"} for i in range(MAX_ITEMS)]
        result = todos.set_items(items)
        assert "Error" not in result
        assert len(todos.items) == MAX_ITEMS

    def test_21_items_rejected(self):
        todos = TodoList()
        items = [{"content": f"item {i}", "status": "pending"} for i in range(MAX_ITEMS + 1)]
        result = todos.set_items(items)
        assert "Error" in result
        assert todos.items == []

    def test_unknown_status_returns_error_string_not_exception(self):
        todos = TodoList()
        # Must not raise — validation failures come back as plain strings.
        result = todos.set_items([{"content": "a", "status": "bogus"}])
        assert isinstance(result, str)
        assert "Error" in result
        assert todos.items == []

    def test_missing_content_is_an_error(self):
        todos = TodoList()
        result = todos.set_items([{"status": "pending"}])
        assert "Error" in result

    def test_non_dict_item_is_an_error_not_exception(self):
        todos = TodoList()
        result = todos.set_items(["not a dict"])
        assert isinstance(result, str)
        assert "Error" in result

    def test_items_property_is_a_defensive_copy(self):
        todos = TodoList()
        todos.set_items([{"content": "a", "status": "pending"}])
        snapshot = todos.items
        snapshot.append({"content": "hack", "status": "pending"})
        assert len(todos.items) == 1

    def test_fresh_todo_list_starts_empty(self):
        # No file-persistence: a new TodoList (i.e. a new session) is empty.
        assert TodoList().items == []


# ---------------------------------------------------------------------------
# TodoWriteTool — the tool wrapper the model calls
# ---------------------------------------------------------------------------


class TestTodoWriteTool:
    def test_name_and_flags(self):
        tool = TodoWriteTool(TodoList())
        assert tool.name == "todo_write"
        assert tool.is_read_only is True
        assert tool.requires_permission is False

    def test_schema_round_trips_through_openai_tool_format(self):
        from spark_code.model import ModelClient

        tool = TodoWriteTool(TodoList())
        schema = tool.to_schema()
        assert schema["name"] == "todo_write"
        assert schema["parameters"]["type"] == "object"
        assert "items" in schema["parameters"]["properties"]

        client = ModelClient.__new__(ModelClient)
        payload = client._build_tools_payload([schema])
        assert payload[0]["type"] == "function"
        assert payload[0]["function"]["name"] == "todo_write"
        assert payload[0]["function"]["parameters"] == schema["parameters"]

    async def test_execute_delegates_to_todo_list(self):
        todos = TodoList()
        tool = TodoWriteTool(todos)
        result = await tool.execute(items=[{"content": "a", "status": "in_progress"}])
        assert "Error" not in result
        assert todos.items[0]["status"] == "in_progress"

    async def test_execute_surfaces_validation_errors(self):
        tool = TodoWriteTool(TodoList())
        result = await tool.execute(items=[{"content": "a", "status": "nope"}])
        assert "Error" in result

    async def test_execute_defaults_missing_items_to_empty_list(self):
        tool = TodoWriteTool(TodoList())
        result = await tool.execute()
        assert "Error" not in result

    def test_items_property_reflects_shared_todo_list(self):
        todos = TodoList()
        tool = TodoWriteTool(todos)
        todos.set_items([{"content": "a", "status": "pending"}])
        assert tool.items == todos.items


# ---------------------------------------------------------------------------
# build_tools() registration
# ---------------------------------------------------------------------------


class TestBuildToolsRegistration:
    def test_todo_write_registered_by_default(self):
        from spark_code.cli import build_tools

        registry = build_tools()
        tool = registry.get("todo_write")
        assert tool is not None
        assert tool.name in registry.names()

    def test_default_build_tools_todo_list_is_empty(self):
        # A fresh call to build_tools() (i.e. a fresh session) starts with no
        # todo state — never persisted across sessions.
        from spark_code.cli import build_tools

        registry = build_tools()
        assert registry.get("todo_write").items == []

    def test_build_tools_accepts_a_shared_todo_list(self):
        from spark_code.cli import build_tools

        todos = TodoList()
        todos.set_items([{"content": "a", "status": "pending"}])
        registry = build_tools(todo_list=todos)
        assert registry.get("todo_write").items == todos.items


# ---------------------------------------------------------------------------
# render_todos — the checklist panel
# ---------------------------------------------------------------------------


class TestRenderTodos:
    def test_markers_and_content_present(self):
        console = _console()
        render_todos(console, [
            {"content": "done thing", "status": "completed"},
            {"content": "active thing", "status": "in_progress"},
            {"content": "later thing", "status": "pending"},
        ])
        output = console.file.getvalue()
        assert "✓" in output  # ✓ completed
        assert "▸" in output  # ▸ in_progress
        assert "○" in output  # ○ pending
        assert "done thing" in output
        assert "active thing" in output
        assert "later thing" in output

    def test_empty_list_does_not_crash(self):
        console = _console()
        render_todos(console, [])
        assert console.file.getvalue()  # printed something, didn't raise


class TestRenderToolCallHeader:
    """The tool-CALL header (before the result is known) shows an item
    count, not a raw JSON dump of every item — the full list renders as a
    panel separately once the call succeeds."""

    def test_todo_write_header_shows_item_count_not_json(self):
        from spark_code.ui.output import render_tool_call

        console = _console()
        render_tool_call(console, "todo_write", {
            "items": [
                {"content": "a", "status": "pending"},
                {"content": "b", "status": "in_progress"},
            ],
        })
        output = console.file.getvalue()
        assert "2 items" in output
        assert "in_progress" not in output  # no raw JSON of the items


# ---------------------------------------------------------------------------
# Live rendering wiring — the panel appears after a successful todo_write call
# ---------------------------------------------------------------------------


async def test_agent_renders_checklist_after_successful_todo_write():
    from spark_code.agent import Agent
    from spark_code.context import Context
    from spark_code.permissions import PermissionManager
    from spark_code.tools.base import ToolRegistry
    from tests.test_agent import MockModel

    todos = TodoList()
    registry = ToolRegistry()
    registry.register(TodoWriteTool(todos))

    model = MockModel([
        [
            {"type": "tool_call", "id": "call_1", "name": "todo_write",
             "arguments": {"items": [{"content": "step one", "status": "in_progress"}]}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "Working on it."},
            {"type": "done", "usage": {}},
        ],
    ])

    console = _console()
    agent = Agent(
        model=model,
        context=Context(),
        tools=registry,
        permissions=PermissionManager(mode="trust"),
        console=console,
    )
    await agent.run("do a 3-step task")

    output = console.file.getvalue()
    assert "step one" in output
    assert "▸" in output  # in_progress marker rendered
