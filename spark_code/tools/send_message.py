"""Send message tool — lets workers communicate with each other and the lead."""

from .base import Tool


class SendMessageTool(Tool):
    """Tool for inter-agent messaging within a team."""

    name = "send_message"
    description = (
        "Send a message to another worker or the lead agent. "
        "Use 'lead' to message the main session, or a worker name like 'worker-1'."
    )
    is_read_only = True
    requires_permission = False

    def __init__(self):
        self._team_manager = None  # Set by TeamManager after registration
        self._sender_name = "unknown"

    def set_context(self, team_manager, sender_name: str):
        """Bind this tool instance to a specific team manager and sender."""
        self._team_manager = team_manager
        self._sender_name = sender_name

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        "Recipient name: 'lead' for the main agent, "
                        "or a worker name like 'worker-1', 'worker-2'. "
                        "Use 'broadcast' to message all workers."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": "The message content to send.",
                },
            },
            "required": ["to", "message"],
        }

    async def execute(self, to: str = "", message: str = "", **kwargs) -> str:
        # Handle fallback when JSON parsing wrapped args in {"raw": "..."}
        if (not to or not message) and "raw" in kwargs:
            import json
            try:
                parsed = json.loads(kwargs["raw"])
                to = parsed.get("to", to)
                message = parsed.get("message", message)
            except (json.JSONDecodeError, TypeError):
                pass

        if not to or not message:
            return "Error: Both 'to' and 'message' are required."
        if not self._team_manager:
            return "Error: Messaging not available (no team manager)."

        return self._team_manager.deliver_message(
            from_name=self._sender_name,
            to_name=to,
            message=message,
        )
