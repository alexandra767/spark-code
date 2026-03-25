"""Conversation context and history management."""

import json
import os
from datetime import datetime

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
- spawn_worker: Spawn a background worker agent for parallel tasks
- send_message: Send a message to another worker or the lead agent

Guidelines:
- For greetings and casual messages (e.g. "hello", "hey", "hi", "thanks"), respond naturally and briefly. Do NOT use tools or explore files — just reply conversationally.
- Only use tools when the user has an actual task or question about code.
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

Xcode/iOS Development Rules:
- NEVER create .xcodeproj or .pbxproj files — Xcode manages these
- When working in an Xcode project, ALWAYS use glob first to find where existing Swift files are located
- Put ALL new Swift files in the SAME directory as the existing ContentView.swift or main app file
- Do NOT create subdirectories (Models/, Views/, etc.) unless they already exist in the project
- For SwiftData models, use @Bindable (not @ObservedObject) for two-way bindings in views
- For SwiftUI, prefer modern APIs: ContentUnavailableView, @Query, .searchable, NavigationStack
- When building iOS projects from CLI, use: xcodebuild -scheme <name> -destination 'platform=iOS Simulator,name=iPhone 17' build
"""

AGENTIC_PROMPT = """You are Spark Code, a fully autonomous AI coding agent running locally.
You complete tasks END TO END without asking for permission or confirmation.

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
- spawn_worker: Spawn a background worker agent for parallel tasks

## Core Behavior — Be Fully Autonomous

You are an AGENT, not an assistant. When given a task:
1. Explore the codebase to understand the context (glob, grep, read_file)
2. Write or modify code to accomplish the task
3. Run tests to verify your changes work
4. If tests fail, read the error, fix the code, and re-run tests
5. Repeat until everything passes
6. Commit with a descriptive message and push if the user asked

NEVER ask "should I proceed?" or "do you approve?" — JUST DO IT.
NEVER say "I'll need to..." or "Let me plan..." — START DOING IT.
NEVER present a plan and wait — EXECUTE the plan immediately.

## When to Use Workers (spawn_worker)

Spawn background workers when:
- Multiple independent files need to be created or modified
- You need to run tests WHILE writing more code
- A task has clearly separable sub-tasks (e.g. "build frontend and backend")
- Research and implementation can happen in parallel

Do NOT spawn workers for:
- Simple single-file changes
- Sequential tasks where each step depends on the previous

## Testing Strategy

- After writing code, ALWAYS run the project's test suite
- If no tests exist, write basic tests first, then implement
- If tests fail, read the failure, fix it, re-run — loop until green
- Common test commands: `pytest`, `npm test`, `cargo test`, `go test ./...`

## Git Strategy

- Use `git status` and `git diff` to understand current state
- Stage specific files (not `git add .`) to avoid committing secrets
- Write concise commit messages focused on WHY, not WHAT
- Only push when explicitly asked or when the task clearly implies it

## Error Recovery

- If a command fails, read the error and try a different approach
- If an edit fails (string not found), re-read the file and adjust
- If tests fail after 3 attempts at fixing, report what's wrong and what you tried
- Never give up after one failure — always try at least 2 approaches

## Communication

- For greetings ("hello", "hi"), respond briefly and naturally
- For tasks, show PROGRESS not plans — e.g. "Creating user model..." then do it
- Keep responses short between tool calls
- At the end, summarize what you did and the current state

## Xcode/iOS Development Rules

- NEVER create .xcodeproj or .pbxproj files — Xcode manages these
- ALWAYS use glob first to find where existing Swift files live in the project
- Put ALL new Swift files in the SAME directory as the existing ContentView.swift
- Do NOT create subdirectories (Models/, Views/) unless they already exist
- For SwiftData models, use @Bindable (not @ObservedObject) for two-way bindings
- For SwiftUI, prefer modern APIs: ContentUnavailableView, @Query, .searchable, NavigationStack
- When building from CLI: xcodebuild -scheme <name> -destination 'platform=iOS Simulator,name=iPhone 17' build
- After build errors, read the error, fix the code, and rebuild automatically
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

    def compact(self, keep_recent: int = 6):
        """Compact conversation with structured summaries.

        Instead of simple truncation, creates a structured summary that
        preserves: files modified, key decisions, errors encountered,
        and the overall task context.
        """
        if len(self.messages) <= keep_recent:
            return

        old = self.messages[:-keep_recent]
        recent = self.messages[-keep_recent:]

        # Build a structured summary
        files_read = set()
        files_written = set()
        files_edited = set()
        commands_run = []
        key_topics = []
        errors = []
        tools_used = set()

        for msg in old:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "user" and isinstance(content, str) and content:
                # Extract the core ask
                first_line = content.strip().split("\n")[0][:120]
                key_topics.append(f"- User: {first_line}")

            elif role == "assistant" and isinstance(content, str) and content:
                # Keep first sentence of assistant responses
                first_sentence = content.strip().split(". ")[0][:100]
                if first_sentence and len(first_sentence) > 10:
                    key_topics.append(f"- Assistant: {first_sentence}")

            elif role == "tool":
                name = msg.get("name", "tool")
                tools_used.add(name)
                if isinstance(content, str):
                    if name == "read_file":
                        # Extract file path from result
                        if content.startswith("File:"):
                            files_read.add(content.split("\n")[0])
                    elif name == "write_file":
                        files_written.add(content[:100])
                    elif name == "edit_file":
                        files_edited.add(content[:100])
                    elif name == "bash":
                        if content and len(content) < 200:
                            commands_run.append(content[:100])
                    if "Error" in content[:50] or "error" in content[:50]:
                        errors.append(f"{name}: {content[:80]}")

            # Check tool_calls in assistant messages
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                tools_used.add(func.get("name", ""))

        # Build structured summary
        parts = ["## Conversation Summary (compacted)\n"]

        if key_topics:
            parts.append("### Key exchanges:")
            parts.extend(key_topics[-8:])
            parts.append("")

        if files_read or files_written or files_edited:
            parts.append("### Files touched:")
            for f in list(files_read)[:5]:
                parts.append(f"  Read: {f}")
            for f in list(files_written)[:5]:
                parts.append(f"  Written: {f}")
            for f in list(files_edited)[:5]:
                parts.append(f"  Edited: {f}")
            parts.append("")

        if errors:
            parts.append("### Errors encountered:")
            for e in errors[-3:]:
                parts.append(f"  {e}")
            parts.append("")

        if tools_used:
            parts.append(f"### Tools used: {', '.join(sorted(tools_used))}")

        summary = "\n".join(parts)
        self.messages = [{"role": "system", "content": summary}] + recent

        # Post-compact check
        if self.max_tokens > 0:
            tokens = self.estimate_tokens()
            if tokens > self.max_tokens * 0.9 and keep_recent > 2:
                self.compact(keep_recent=max(2, keep_recent // 2))

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

    def save(self, path: str, label: str = "", cwd: str = ""):
        """Save conversation to file with metadata."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "turn_count": self.turn_count,
            "label": label,
            "cwd": cwd,
            "messages": self.messages,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        """Load conversation from file."""
        if not os.path.exists(path):
            return False
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.messages = data.get("messages", [])
        self.turn_count = data.get("turn_count", 0)
        return True

    @staticmethod
    def read_metadata(path: str) -> dict:
        """Read session metadata without loading full messages."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "timestamp": data.get("timestamp", ""),
                "turn_count": data.get("turn_count", 0),
                "label": data.get("label", ""),
                "cwd": data.get("cwd", ""),
            }
        except (json.JSONDecodeError, OSError):
            return {}
