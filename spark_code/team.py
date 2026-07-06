"""Team system — spawn background worker agents with messaging."""

import asyncio
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console
from rich.text import Text

from .agent import Agent
from .context import Context
from .model import ModelClient
from .permissions import PermissionManager
from .task_store import TaskStore
from .tools.base import ToolRegistry
from .tools.send_message import SendMessageTool

# Nord palette
_C_TOOL = "#88c0d0"
_C_GREEN = "#a3be8c"
_C_RED = "#bf616a"
_C_YELLOW = "#ebcb8b"
_C_DIM = "#4c566a"
_C_TEXT = "#d8dee9"
_C_MUTED = "#666666"
_C_BLUE = "#5e81ac"

MAX_WORKERS_LOCAL = 2   # Same GPU — limit contention
MAX_WORKERS_CLOUD = 6   # Separate model or cloud API — no contention
WORKER_TIMEOUT = 300  # 5 minutes

WORKER_SYSTEM_PROMPT = """You are a Spark Code worker agent. You have ONE job — complete the task you were given.

CRITICAL RULES:
1. ACT IMMEDIATELY. Start writing code NOW — don't explore the filesystem first.
2. CREATING a new file? Use write_file directly. Don't glob or list_dir first.
3. EDITING an existing file? Read it first with read_file, then use edit_file.
4. Write COMPLETE files in one write_file call. No stubs or placeholders.
5. After writing, test your code with bash (run it, run pytest, etc.)
6. If a test fails, read the file, fix it with edit_file, and re-test.
7. When done, provide a brief summary of what you created and test results.

DO NOT:
- Explore directories before creating new files
- Wait for other workers or check what they've done
- Ask questions — just build what was requested
- Write placeholder or stub code — write the real implementation

Tools available:
- write_file: Create files. Parameters: file_path (required), content (required)
- edit_file: Find & replace. Parameters: file_path (required), old_string (required), new_string (required)
- read_file: Read files. Parameters: file_path (required), offset, limit
- bash: Run commands. Parameters: command (required), timeout
- glob: Find files. Parameters: pattern (required), path
- grep: Search content. Parameters: pattern (required), path
- list_dir: List directory. Parameters: path (required)
- send_message: Message others. Parameters: to (required), message (required)
  - to="lead" for main session, to="worker-name" for specific worker, to="broadcast" for all
"""


@dataclass
class Message:
    """A message between agents."""
    from_name: str
    to_name: str
    content: str


@dataclass
class Worker:
    """A background agent running as an async task."""

    id: str
    name: str
    prompt: str
    status: str = "running"  # running | completed | failed
    result: str = ""
    agent: Agent | None = None
    asyncio_task: asyncio.Task | None = None
    inbox: deque = field(default_factory=deque)
    current_tool: str = ""

    def drain_inbox(self) -> list["Message"]:
        """Inject any pending inbox messages into this worker's context.

        MUST be called only at a SAFE boundary — the TOP of an agent-loop
        iteration, never between an assistant ``tool_calls`` message and its
        matching tool results — otherwise the message sequence becomes invalid
        (a user message wedged between tool_calls and tool_result).

        Returns the list of messages that were drained (for logging/tests).
        """
        if not self.inbox:
            return []
        drained: list[Message] = []
        while self.inbox:
            msg = self.inbox.popleft()
            drained.append(msg)
            if self.agent is not None:
                self.agent.context.add_user(
                    f"[Message from {msg.from_name}]: {msg.content}")
        return drained


class TeamManager:
    """Manages background worker agents with messaging."""

    def __init__(self, model: ModelClient, tools: ToolRegistry,
                 console: Console, task_store: TaskStore,
                 stats=None, worker_model=None,
                 worker_permission_mode: str = "auto"):
        self.model = model
        self.worker_model = worker_model  # separate model for workers (faster)
        self.max_workers = MAX_WORKERS_CLOUD if worker_model else MAX_WORKERS_LOCAL
        self.tools = tools
        self.console = console
        self.task_store = task_store
        self.stats = stats  # shared session stats for file tracking
        # Permission mode workers run under. Defaults to "auto" (read-only
        # auto-allowed, writes gated) rather than "trust" so an approved
        # background worker can't turn into an unrestricted shell. cli.py
        # should pass the LEAD agent's mode here so workers inherit it.
        self.worker_permission_mode = worker_permission_mode
        self.workers: dict[str, Worker] = {}
        self._counter = 0
        self._name_counter = 0  # monotonic — guarantees unique auto worker names
        self.files_changed: list[dict] = []
        self._spawn_queue: deque[tuple[str, str]] = deque()  # (prompt, name) waiting to spawn
        # Lead agent's inbox — messages from workers to "lead"
        self.lead_inbox: deque[Message] = deque()
        # Durable full results keyed by worker name (survives status display
        # racing get_lead_messages, and is not truncated).
        self.worker_results: dict[str, str] = {}

    @property
    def active_count(self) -> int:
        return sum(1 for w in self.workers.values() if w.status == "running")

    def deliver_message(self, from_name: str, to_name: str, message: str) -> str:
        """Deliver a message between agents. Called by SendMessageTool."""
        msg = Message(from_name=from_name, to_name=to_name, content=message)

        # Show message in console
        self.console.print(
            Text(f"  [{from_name}] → [{to_name}]: {message[:80]}", style=_C_BLUE))

        if to_name == "lead":
            self.lead_inbox.append(msg)
            return "Message delivered to lead agent."

        if to_name == "broadcast":
            count = 0
            for w in self.workers.values():
                if w.name != from_name and w.status == "running":
                    # Only enqueue — the worker drains its inbox at a safe
                    # boundary (top of its agent loop). We must NOT mutate the
                    # worker's live context here: injecting a user message
                    # between an assistant tool_calls message and its tool
                    # results yields an invalid message sequence.
                    w.inbox.append(msg)
                    count += 1
            self.lead_inbox.append(msg)  # Lead also sees broadcasts
            return f"Message broadcast to {count} worker(s) and the lead."

        # Find target worker
        target = self._find_worker_by_name(to_name)
        if not target:
            return f"Error: Worker '{to_name}' not found. Active workers: {self._active_worker_names()}"
        if target.status != "running":
            return (f"Error: Worker '{to_name}' is no longer running "
                    f"(status: {target.status}); message not delivered.")

        # Enqueue only — the worker drains its inbox at a safe boundary.
        target.inbox.append(msg)
        return f"Message delivered to {to_name}."

    def get_lead_messages(self) -> list[Message]:
        """Pop all messages for the lead agent."""
        msgs = list(self.lead_inbox)
        self.lead_inbox.clear()
        return msgs

    def _find_worker_by_name(self, name: str) -> Worker | None:
        for w in self.workers.values():
            if w.name == name:
                return w
        return None

    def _active_worker_names(self) -> str:
        names = [w.name for w in self.workers.values() if w.status == "running"]
        return ", ".join(names) if names else "(none)"

    def _next_auto_name(self) -> str:
        """Return a unique auto-generated worker name (never collides)."""
        self._name_counter += 1
        return f"worker-{self._name_counter}"

    async def spawn(self, prompt: str, name: str = "") -> Worker | None:
        """Spawn a new worker agent with the given task."""
        # Reject duplicate names if that worker is still running.
        # NOTE: look up by NAME (get_worker keys by numeric id and would never
        # match a name, silently disabling this guard).
        if name:
            existing = self._find_worker_by_name(name)
            if existing and existing.status == "running":
                self.console.print(
                    Text(f"  Worker '{name}' is already running. "
                         f"Use a different name or wait for it to finish.",
                         style=_C_YELLOW))
                return None

        if self.active_count >= self.max_workers:
            # Queue the worker instead of rejecting. Give unnamed queued
            # workers a UNIQUE auto name so multiple queued workers don't all
            # collide on the same "worker-N".
            worker_name = name or self._next_auto_name()
            self._spawn_queue.append((prompt, worker_name))
            self.console.print(
                Text(f"  [{worker_name}] Queued — will start when a slot opens "
                     f"({len(self._spawn_queue)} in queue)",
                     style=_C_MUTED))
            return None

        self._counter += 1
        worker_id = str(self._counter)
        worker_name = name or self._next_auto_name()

        # Create task in shared store
        task = self.task_store.create(prompt, assigned_to=worker_name)

        # Build a tool registry for this worker with its own send_message instance.
        # Exclude spawn_worker so workers cannot spawn sub-workers (prevents
        # unbounded recursive spawning).
        worker_tools = ToolRegistry()
        for tool in self.tools.all():
            if tool.name == "spawn_worker":
                continue
            worker_tools.register(tool)

        # Create a per-worker send_message tool bound to this worker
        msg_tool = SendMessageTool()
        msg_tool.set_context(self, worker_name)
        worker_tools.register(msg_tool)

        # Build system prompt with team awareness
        active_names = self._active_worker_names()
        team_info = ""
        if active_names and active_names != "(none)":
            team_info = f"\n\nOther active workers: {active_names}\nYou are: {worker_name}\n"

        # Each worker gets its own context. Permissions INHERIT the lead's
        # mode (defaults to "auto") instead of a hardcoded "trust" so a single
        # approved worker can't become an unrestricted background shell.
        context = Context(
            system_prompt=WORKER_SYSTEM_PROMPT + team_info,
            max_tokens=32768,
        )
        # Non-interactive: a worker runs on the shared event loop, so a blocking
        # permission prompt would freeze the whole session. It keeps the lead's
        # mode (for display/policy) but never prompts. It fails SAFE rather than
        # auto-approving: it allows only what the mode would already allow
        # without a prompt (trust allows all; auto allows read-only; always/
        # session-allowed tools) and denies anything that would have required
        # a prompt — so a worker still can't become an unrestricted background
        # shell just by inheriting "ask" or "auto".
        permissions = PermissionManager(mode=self.worker_permission_mode,
                                        interactive=False)

        # Create a prefixed console wrapper for worker output
        worker_console = _PrefixedConsole(self.console, worker_name)

        # Use worker_model if available (faster model for workers)
        agent_model = self.worker_model or self.model
        agent = Agent(
            model=agent_model,
            context=context,
            tools=worker_tools,
            permissions=permissions,
            console=worker_console,
            output_prefix=f"[{worker_name}]",
        )

        worker = Worker(
            id=worker_id,
            name=worker_name,
            prompt=prompt,
            agent=agent,
        )
        # Drain queued inbox messages into context at each agent-loop boundary.
        agent.on_iteration = worker.drain_inbox
        self.workers[worker_id] = worker

        # Print start message
        self.console.print(
            Text(f"  [{worker_name}] Starting: {prompt}", style=_C_TOOL))

        # Launch as background async task
        worker.asyncio_task = asyncio.create_task(
            self._run_worker(worker, task.id)
        )

        return worker

    async def _run_worker(self, worker: Worker, task_id: str):
        """Run a worker agent to completion."""
        try:
            result = await asyncio.wait_for(
                worker.agent.run(worker.prompt),
                timeout=WORKER_TIMEOUT
            )
            worker.status = "completed"
            worker.result = result or "(completed with no text output)"
            # Persist the FULL result durably (not truncated) so the lead can
            # retrieve it via get_worker_result even after the status display
            # consumes the ephemeral lead-message queue.
            self.worker_results[worker.name] = worker.result

            self.task_store.update(
                task_id, status="completed",
                result=worker.result,
            )

            self.lead_inbox.append(Message(
                from_name=worker.name,
                to_name="lead",
                content=f"[team] {worker.name} completed: {worker.result[:200]}"
            ))

            self.console.print(
                Text(f"  [{worker.name}] \u2713 Completed", style=_C_GREEN))

        except asyncio.TimeoutError:
            worker.status = "failed"
            worker.result = f"Timed out after {WORKER_TIMEOUT}s"
            self.worker_results[worker.name] = worker.result
            self.task_store.update(task_id, status="failed",
                                   result=worker.result)
            self.lead_inbox.append(Message(
                from_name=worker.name,
                to_name="lead",
                content=f"[team] {worker.name} timed out after {WORKER_TIMEOUT}s"
            ))
            self.console.print(
                Text(f"  [{worker.name}] \u2717 Timed out ({WORKER_TIMEOUT}s)",
                     style=_C_RED))

        except asyncio.CancelledError:
            # Cancellation means someone is STOPPING us (/team stop or session
            # exit). Record the state, then RE-RAISE so the task actually ends
            # cancelled \u2014 and crucially do NOT fall through to _drain_queue(),
            # which would spawn the queued workers we're trying to stop.
            worker.status = "failed"
            worker.result = "Cancelled"
            self.worker_results[worker.name] = worker.result
            self.task_store.update(task_id, status="failed", result="Cancelled")
            self.console.print(
                Text(f"  [{worker.name}] Stopped", style=_C_YELLOW))
            raise

        except Exception as e:
            worker.status = "failed"
            worker.result = str(e)
            self.worker_results[worker.name] = worker.result
            self.task_store.update(task_id, status="failed", result=str(e))
            self.console.print(
                Text(f"  [{worker.name}] \u2717 Failed: {e}", style=_C_RED))

        # Auto-spawn next queued worker if there's a slot.
        # (Unreachable on cancellation \u2014 we re-raised above.)
        await self._drain_queue()

    async def _drain_queue(self):
        """Spawn queued workers as slots become available."""
        while self._spawn_queue and self.active_count < self.max_workers:
            prompt, name = self._spawn_queue.popleft()
            remaining = len(self._spawn_queue)
            self.console.print(
                Text(f"  [{name}] Starting from queue"
                     + (f" ({remaining} still queued)" if remaining else ""),
                     style=_C_TOOL))
            await self.spawn(prompt, name)

    async def stop(self, worker_id: str) -> bool:
        """Stop a specific worker."""
        worker = self.workers.get(worker_id)
        if not worker:
            return False
        if worker.asyncio_task and not worker.asyncio_task.done():
            worker.asyncio_task.cancel()
            try:
                await worker.asyncio_task
            except asyncio.CancelledError:
                pass
        return True

    async def stop_all(self):
        """Stop all running workers and drop any queued (not-yet-started) ones."""
        # Drop pending queued workers first so nothing spawns after we stop.
        self._spawn_queue.clear()
        tasks = []
        for worker in self.workers.values():
            if worker.asyncio_task and not worker.asyncio_task.done():
                worker.asyncio_task.cancel()
                tasks.append(worker.asyncio_task)
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    def status(self) -> list[dict]:
        """Get status of all workers."""
        result = []
        for w in self.workers.values():
            result.append({
                "id": w.id,
                "name": w.name,
                "status": w.status,
                "prompt": w.prompt,
                "result": w.result,
                "current_tool": w.current_tool,
            })
        return result

    def notify_file_written(self, writer_name: str, file_path: str, line_count: int):
        """Broadcast file write notification to all other running workers."""
        import os
        filename = os.path.basename(file_path)
        msg_content = f"[team] {writer_name} wrote {filename} ({line_count} lines)"
        # Track in shared stats for /clean
        if self.stats and hasattr(self.stats, 'record_file_created'):
            self.stats.record_file_created(file_path)
        self.files_changed.append({
            "path": file_path,
            "worker": writer_name,
            "lines": line_count,
        })
        for w in self.workers.values():
            if w.name != writer_name and w.status == "running":
                w.inbox.append(Message(
                    from_name="team",
                    to_name=w.name,
                    content=msg_content,
                ))

    def format_file_summary(self) -> str:
        """Format a summary of files changed by workers."""
        if not self.files_changed:
            return ""
        import os
        lines = ["  Worker File Summary:"]
        for fc in self.files_changed:
            filename = os.path.basename(fc["path"])
            lines.append(f"    + {filename} ({fc['lines']} lines, by {fc['worker']})")
        return "\n".join(lines)

    def get_worker(self, worker_id: str) -> Worker | None:
        return self.workers.get(worker_id)

    def get_worker_result(self, name: str) -> str | None:
        """Return the FULL stored result for a worker by NAME, or None.

        Reads from the durable ``worker_results`` map first (survives the
        ephemeral lead-message queue being consumed by the status display),
        then falls back to the live worker's result.
        """
        if name in self.worker_results:
            return self.worker_results[name]
        worker = self._find_worker_by_name(name)
        return worker.result if worker else None


class _PrefixedConsole:
    """Wraps a Console to prefix all output with a worker name.

    Delegates all attribute access to the underlying Console so it
    can be used as a drop-in replacement.
    """

    def __init__(self, console: Console, worker_name: str):
        self._console = console
        self._prefix = f"  [{worker_name}] "
        self._worker_name = worker_name

    def print(self, *args, **kwargs):
        """Intercept print calls to add worker prefix."""
        if args and isinstance(args[0], Text):
            # Prefix Text objects
            prefixed = Text(self._prefix, style=_C_DIM)
            prefixed.append_text(args[0])
            self._console.print(prefixed, **kwargs)
        elif args and isinstance(args[0], str):
            self._console.print(f"{self._prefix}{args[0]}", **kwargs)
        else:
            self._console.print(*args, **kwargs)

    @property
    def width(self):
        return self._console.width

    def __enter__(self):
        return self._console.__enter__()

    def __exit__(self, *args):
        return self._console.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._console, name)
