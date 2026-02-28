"""Conversation context and history management."""

import json
import os
from datetime import datetime
from pathlib import Path


SYSTEM_PROMPT = """You are Spark Code, a local AI coding assistant running on DGX Spark.
You help users with software engineering tasks by reading files, writing code, running commands, and searching the web.

You have access to these tools:
- read_file: Read file contents
- write_file: Create new files
- edit_file: Find & replace in files
- bash: Run shell commands
- glob: Find files by pattern
- grep: Search file contents
- list_dir: List directory contents
- web_search: Search the web
- web_fetch: Fetch web pages

Guidelines:
- Read files before editing them
- Use glob/grep to find files before reading
- Run tests after making changes
- Show diffs before applying edits
- Plan before executing complex tasks
- If something fails, try a different approach
- Be concise but thorough
- Use markdown formatting in responses

Important: When the user types what looks like a shell command (e.g. "python snake.py", "npm start",
"ls -la", "pip install X", "make build"), just run it immediately with the bash tool.
Do NOT ask for clarification — the user expects it to run. Use the current working directory.
"""


class Context:
    """Manages conversation history and context window."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, max_tokens: int = 32768):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.turn_count = 0

    def add_user(self, content: str):
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})
        self.turn_count += 1

    def add_user_with_image(self, text: str, image_base64: str,
                            mime_type: str = "image/png"):
        """Add a user message with an embedded image (multimodal)."""
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}",
                    },
                },
            ],
        })
        self.turn_count += 1

    def add_assistant(self, content: str):
        """Add an assistant text response."""
        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_calls(self, tool_calls: list[dict]):
        """Add an assistant message with tool calls."""
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        })

    def add_tool_result(self, tool_call_id: str, name: str, result: str):
        """Add a tool result."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": result,
        })

    def get_messages(self) -> list[dict]:
        """Get full message list with system prompt."""
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def compact(self):
        """Compact conversation by summarizing older messages."""
        if len(self.messages) <= 6:
            return

        # Keep system prompt + last 6 messages, summarize the rest
        old = self.messages[:-6]
        recent = self.messages[-6:]

        summary_parts = []
        for msg in old:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "user" and content:
                summary_parts.append(f"User asked: {content[:100]}")
            elif role == "assistant" and content:
                summary_parts.append(f"Assistant: {content[:100]}")
            elif role == "tool":
                name = msg.get("name", "tool")
                summary_parts.append(f"Tool {name} was called")

        summary = "Previous conversation summary:\n" + "\n".join(summary_parts[-10:])
        self.messages = [{"role": "user", "content": summary}] + recent

    def clear(self):
        """Clear all messages."""
        self.messages = []
        self.turn_count = 0

    def estimate_tokens(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        total = len(self.system_prompt)
        for msg in self.messages:
            content = msg.get("content", "") or ""
            total += len(content)
            for tc in msg.get("tool_calls", []):
                total += len(json.dumps(tc))
        return total // 4

    def save(self, path: str):
        """Save conversation to file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "turn_count": self.turn_count,
            "messages": self.messages,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        """Load conversation from file."""
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.messages = data.get("messages", [])
        self.turn_count = data.get("turn_count", 0)
        return True
