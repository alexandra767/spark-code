"""Manages a TODO list stored in ~/.spark/todos.json."""
import json
import os

from .base import Tool


class TodoTool(Tool):
    name = "todo"
    description = "Manage a TODO list. Supports add, list, remove, and clear operations."
    is_read_only = False
    requires_permission = False

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "list", "remove", "clear"],
                    "description": "The operation to perform: add, list, remove, or clear."
                },
                "task": {
                    "type": "string",
                    "description": "The task to add or remove. Required for add and remove operations."
                }
            },
            "required": ["operation"]
        }

    def _get_todos_path(self) -> str:
        home_dir = os.path.expanduser("~")
        spark_dir = os.path.join(home_dir, ".spark")
        if not os.path.exists(spark_dir):
            os.makedirs(spark_dir)
        return os.path.join(spark_dir, "todos.json")

    def _load_todos(self) -> list:
        todos_path = self._get_todos_path()
        if os.path.exists(todos_path):
            with open(todos_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        else:
            return []

    def _save_todos(self, todos: list) -> None:
        todos_path = self._get_todos_path()
        with open(todos_path, "w") as f:
            json.dump(todos, f)

    async def execute(self, operation: str, task: str = None, **kwargs) -> str:
        todos = self._load_todos()

        if operation == "add":
            if not task:
                return "Error: Task is required for add operation."
            todos.append(task)
            self._save_todos(todos)
            return f"Added task: {task}"
        elif operation == "list":
            if not todos:
                return "No todos found."
            return "\n".join([f"{i+1}. {task}" for i, task in enumerate(todos)])
        elif operation == "remove":
            if not task:
                return "Error: Task is required for remove operation."
            try:
                todos.remove(task)
                self._save_todos(todos)
                return f"Removed task: {task}"
            except ValueError:
                return f"Error: Task not found: {task}"
        elif operation == "clear":
            self._save_todos([])
            return "Cleared all todos."
        else:
            return f"Error: Invalid operation: {operation}"
