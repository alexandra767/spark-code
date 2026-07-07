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
- rag_search: Search your indexed knowledge base (Swift docs, Apple HIG, App Store guidelines, Google Docs, crawled sites). Use this BEFORE writing Swift/SwiftUI code to check for latest patterns, guidelines, and best practices.
- code_search: Semantic search over THIS project's indexed code (path:line results) — finds "where/how is X handled" by meaning, not literal text
- spawn_worker: Spawn a background worker agent for parallel tasks
- send_message: Send a message to another worker or the lead agent
- todo_write: Create or update your live todo checklist for this session
- dispatch_agent: Run an open-ended search/review/task in a fresh sub-agent context

RAG-Powered Development (only if a rag_search tool is available):
- Before writing Swift/SwiftUI code, use rag_search to check for relevant documentation, Apple HIG guidelines, and App Store review rules
- Discover available collections at runtime rather than assuming names; a common one is "claude_documents" for Swift docs, HIG, and App Store guidelines
- Example: before building a settings screen, search for "HIG settings" and "App Store guidelines settings"
- Prefer RAG results over training data for current syntax, patterns, and guidelines

Guidelines:
- For greetings and casual messages (e.g. "hello", "hey", "hi", "thanks"), respond naturally and briefly. Do NOT use tools or explore files — just reply conversationally.
- Only use tools when the user has an actual task or question about code.
- Read files before editing them
- Use glob/grep to find files before reading
- Run tests after making changes
- Show diffs before applying edits
- Plan before executing complex tasks (use /projectplan for RAG-researched plans)
- If something fails, try a different approach
- Be concise but thorough
- Use markdown formatting in responses
- For any task with 3 or more steps, maintain a todo list with the todo_write tool: set statuses as you start and finish each step. Keep exactly one item in_progress.
- Use dispatch_agent for open-ended codebase searches, reviews, or research — it runs in a fresh context and returns a summary, keeping your own context small.
- For semantic "where/how is X handled" questions, use code_search (falls back to grep if unavailable).
- Before declaring code work done, run the project's tests/build and report the result.

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
- code_search: Semantic search over THIS project's indexed code (path:line results) — finds "where/how is X handled" by meaning, not literal text
- spawn_worker: Spawn a background worker agent for parallel tasks
- todo_write: Create or update your live todo checklist for this session
- dispatch_agent: Run an open-ended search/review/task in a fresh sub-agent context

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

## Todo Tracking

For any task with 3 or more steps, maintain a todo list with the todo_write tool: set statuses as you start and finish each step. Keep exactly one item in_progress.

Use dispatch_agent for open-ended codebase searches, reviews, or research — it runs in a fresh context and returns a summary, keeping your own context small.

For semantic "where/how is X handled" questions, use code_search (falls back to grep if unavailable).

## Testing Strategy

- After writing code, ALWAYS run the project's test suite
- If no tests exist, write basic tests first, then implement
- If tests fail, read the failure, fix it, re-run — loop until green
- Common test commands: `pytest`, `npm test`, `cargo test`, `go test ./...`
- Before declaring code work done, run the project's tests/build and report the result.

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


# 80B-friendly skills (Phase 2 Task 8): the system prompt carries only a
# compact skill INDEX (name + one-line description each, capped — see
# SkillRegistry.build_index) plus this STATIC hint teaching the model how to
# invoke one. Full skill bodies never live in the system prompt — they enter
# context only on invocation (slash command fallback in cli.py, or the
# auto-expand below handled in agent.py). Verbatim per the plan.
SKILL_TRIGGER_HINT = (
    "Available skills are listed above. When a user request matches a "
    "skill's description, invoke it by replying with exactly "
    "`/<skill-name> <args>` on its own line — it will be expanded for you."
)


def build_skill_prompt_section(skill_index: str) -> str:
    """Wrap a pre-built skill index (``SkillRegistry.build_index()``) with a
    header and the trigger hint, ready to append to a system prompt.

    Takes the already-built index STRING rather than a ``SkillRegistry`` so
    this module doesn't need to depend on the skills package — callers build
    the index once (cli.py, at startup) and pass the result in. Returns ""
    when there's no index to show (empty registry) so callers never inject an
    empty/useless section into the prompt.
    """
    if not skill_index:
        return ""
    return "\n\n## Available skills\n" + skill_index + "\n\n" + SKILL_TRIGGER_HINT


class Context:
    """Manages conversation history and context window."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, max_tokens: int = 32768,
                 platform_prompt: str = "", provider_prompt: str = ""):
        self.system_prompt = system_prompt
        self.platform_prompt = platform_prompt
        self.provider_prompt = provider_prompt
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.turn_count = 0
        # Fixed per-request overhead of the tool schemas (sent on EVERY request
        # but absent from `messages`). The agent seeds this from the size of its
        # tool schemas so estimate_tokens — and thus auto-compaction — accounts
        # for it instead of overshooting into a context-length 400.
        self.schema_reserve_tokens = 0
        # Server-reported prompt token count from the most recent response's
        # usage block (set via record_usage). None until the first response
        # arrives, or after a history change (compact/clear) makes the old
        # count stale.
        self.last_prompt_tokens: int | None = None

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

    def add_assistant_tool_calls(self, tool_calls: list[dict], content: str = ""):
        """Add an assistant message with tool calls.

        ``content`` preserves any narration the model produced alongside the
        tool calls (OpenAI allows content + tool_calls together); dropping it
        loses the model's own reasoning across rounds.
        """
        self.messages.append({
            "role": "assistant",
            "content": content or None,
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

    def sanitize_orphaned_tool_calls(self) -> int:
        """Backfill synthetic results for assistant tool_calls left unanswered.

        An interrupt (Ctrl+C) between an assistant ``tool_calls`` message and the
        recording of its tool results — or reloading a session saved in that
        state — leaves one or more ``tool_call`` ids with no matching
        ``role:"tool"`` reply. OpenAI-compatible servers reject that sequence with
        a 400, wedging every subsequent request. This walks the history and, for
        each assistant ``tool_calls`` message, appends a synthetic
        ``{"role":"tool", …, "content":"[interrupted]"}`` result for any id that
        the immediately-following tool messages did not answer. Idempotent.

        Returns the number of synthetic results inserted.
        """
        new_messages: list[dict] = []
        backfilled = 0
        i = 0
        n = len(self.messages)
        while i < n:
            msg = self.messages[i]
            new_messages.append(msg)
            tool_calls = (msg.get("tool_calls")
                          if msg.get("role") == "assistant" else None)
            if tool_calls:
                # Consume the run of tool results answering this assistant msg.
                answered: set = set()
                j = i + 1
                while j < n and self.messages[j].get("role") == "tool":
                    new_messages.append(self.messages[j])
                    answered.add(self.messages[j].get("tool_call_id"))
                    j += 1
                for tc in tool_calls:
                    cid = tc.get("id")
                    if cid not in answered:
                        name = tc.get("function", {}).get("name", "")
                        new_messages.append({
                            "role": "tool",
                            "tool_call_id": cid,
                            "name": name,
                            "content": "[interrupted]",
                        })
                        backfilled += 1
                i = j
                continue
            i += 1
        if backfilled:
            self.messages = new_messages
        return backfilled

    def get_messages(self) -> list[dict]:
        """Return messages with system prompt prepended."""
        # Repair any orphaned tool_calls (e.g. from a mid-tool Ctrl+C) before the
        # request goes out, so the server never sees an invalid sequence.
        self.sanitize_orphaned_tool_calls()
        # Prefix-cache ordering (spec): STATIC content first, VOLATILE last, so
        # the server caches the longest stable prefix across turns.
        #   1. system_prompt — the large static base + instructions + memory +
        #      project type (+ any pinned block at its tail, which changes only
        #      when a pinned file's on-disk content does).
        #   2. provider_prompt — static per-session config text.
        #   3. platform_prompt — the volatile slot (cwd today; git status / todo
        #      state land here later) → LAST so per-turn changes don't invalidate
        #      the cached static prefix above it.
        parts = [self.system_prompt]
        if self.provider_prompt:
            parts.append(self.provider_prompt)
        if self.platform_prompt:
            parts.append(self.platform_prompt)
        combined = "\n\n".join(parts)
        return [{"role": "system", "content": combined}] + self.messages

    def compact(self, keep_recent: int = 6, summary: str | None = None) -> str | None:
        """Compact conversation, dropping old turns for a summary.

        When ``summary`` is provided (a model-generated recap from the agent),
        it replaces the internally-generated mechanical digest. When it is None
        the mechanical structured digest is built instead — the offline/test
        fallback that preserves files modified, key decisions, errors, and the
        overall task context. Either way boundary safety (``_safe_split_index``)
        and pinned-file survival (pinned live in ``system_prompt``, never in
        ``messages``) are unchanged.
        """
        if len(self.messages) <= keep_recent:
            return None

        before_count = len(self.messages)
        before_tokens = self.estimate_tokens()

        # Snap the split point to a safe boundary so `recent` never BEGINS with
        # orphaned tool results (a role:"tool" message whose assistant
        # tool_calls parent got summarized away) — that produces an invalid
        # message sequence that OpenAI-compatible servers reject with a 400.
        split = self._safe_split_index(len(self.messages) - keep_recent)
        if split <= 0:
            return None
        old = self.messages[:split]
        recent = self.messages[split:]

        if summary is not None:
            # Model-generated recap replaces the mechanical digest. Wrap it under
            # the same header so downstream (and the model on the next turn) reads
            # it as the compacted context; the text itself is preserved verbatim.
            summary_text = f"## Conversation Summary (compacted)\n\n{summary}"
        else:
            summary_text = self._mechanical_digest(old)

        # Inject summary as a user message (not system — get_messages() adds
        # the system prompt itself).
        self.messages = [{"role": "user", "content": summary_text},
                         {"role": "assistant",
                          "content": "Understood, I have the conversation context."}] + recent
        # The history just changed shape, so any server-reported prompt token
        # count from before this compaction no longer reflects reality —
        # stale usage must not keep reporting the pre-compaction window.
        self.last_prompt_tokens = None

        # Post-compact check
        if self.max_tokens > 0:
            tokens = self.estimate_tokens()
            if tokens > self.max_tokens * 0.9 and keep_recent > 2:
                self.compact(keep_recent=max(2, keep_recent // 2))

        after_count = len(self.messages)
        after_tokens = self.estimate_tokens()
        freed = before_tokens - after_tokens
        return f"Context compacted: {before_count} messages → {after_count} (freed ~{freed:,} tokens)"

    def _mechanical_digest(self, old: list[dict]) -> str:
        """Build a structured, offline digest of the ``old`` messages being
        dropped. Used as the compaction summary when no model-generated summary
        is supplied (offline/test fallback)."""
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

        return "\n".join(parts)

    def _safe_split_index(self, idx: int) -> int:
        """Move ``idx`` earlier until messages[idx:] does not start with an
        orphaned tool result. A ``role:"tool"`` message must be preceded by the
        assistant ``tool_calls`` message it answers; if the boundary lands
        between them, walk back past the assistant tool_calls message too."""
        idx = max(0, min(idx, len(self.messages)))
        # Clamp at 0: without the `idx > 0` guard a pathological history whose
        # head is all tool messages would walk idx negative — Python's negative
        # indexing then reads from the TAIL and the loop ends in IndexError.
        while 0 < idx < len(self.messages) and self.messages[idx].get("role") == "tool":
            idx -= 1
        if (idx == 0 and self.messages
                and self.messages[0].get("role") == "tool"):
            return 0
        return idx

    def clear(self):
        """Clear all messages."""
        self.messages = []
        self.turn_count = 0
        self.last_prompt_tokens = None

    @staticmethod
    def _estimate_messages_chars(messages: list[dict]) -> int:
        """Char count of a message slice, for token estimation."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                # Multimodal message — count text parts and approximate images
                # by their (base64) payload size so a megabyte image isn't
                # counted as ~0 tokens (which would defeat auto-compaction).
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        total += len(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        total += len(url)
            for tc in msg.get("tool_calls", []) or []:
                total += len(json.dumps(tc))
        return total

    def estimate_tokens(self) -> int:
        """Rough token estimate.

        Uses ~3.5 chars/token (tighter than the naive 4 — code and JSON, which
        dominate here, tokenize denser than prose) and adds ``schema_reserve_
        tokens`` for the tool schemas sent on every request but not stored in
        ``messages``. Both make the estimate a safe UPPER-ish bound so
        auto-compaction fires before the real window is exceeded.
        """
        total = len(self.system_prompt) + len(self.platform_prompt) + len(self.provider_prompt)
        total += self._estimate_messages_chars(self.messages)
        return int(total / 3.5) + self.schema_reserve_tokens

    def estimate_compactable_tokens(self, keep_recent: int = 6) -> int:
        """Estimated tokens of what ``compact(keep_recent)`` could remove.

        Counts only the removable slice — the messages before the SAFE split
        point (the same boundary logic compact() uses). The system prompt and
        the keep_recent tail survive compaction and are excluded, so a giant
        static prompt can't make a hopeless compaction look worthwhile (review
        round-1 thrash: summary requests fired every round while freeing ~0).
        Returns 0 when there is nothing to remove.
        """
        if len(self.messages) <= keep_recent:
            return 0
        split = self._safe_split_index(len(self.messages) - keep_recent)
        if split <= 0:
            return 0
        return int(self._estimate_messages_chars(self.messages[:split]) / 3.5)

    def record_usage(self, prompt_tokens: int):
        """Record the server-reported prompt token count from the most recent
        response's usage block. Overrides the char-based estimate for
        context_left() until the history changes (compact/clear reset it —
        a stale count would otherwise keep reporting the pre-compaction
        window)."""
        self.last_prompt_tokens = prompt_tokens

    def context_left(self, window: int) -> float:
        """Fraction of the context window still free, as 1.0 (empty) down to
        0.0 (full). Prefers the last server-reported prompt token count
        (accurate — it's the model's own tokenizer); falls back to
        estimate_tokens() when the server has never reported usage (e.g. some
        Ollama configs omit it). Never divides by zero: an unset/zero window
        reports 0.0 left (treated as full/unknown) rather than raising."""
        if not window:
            return 0.0
        tokens = self.last_prompt_tokens if self.last_prompt_tokens is not None \
            else self.estimate_tokens()
        return max(0.0, min(1.0, 1.0 - (tokens / window)))

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
        """Load conversation from file. Returns False on a missing/corrupt file
        instead of crashing the session on resume."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        self.messages = data.get("messages", [])
        self.turn_count = data.get("turn_count", 0)
        # A session saved mid-interrupt can carry orphaned tool_calls; repair on
        # load so --resume / /history don't immediately 400.
        self.sanitize_orphaned_tool_calls()
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
