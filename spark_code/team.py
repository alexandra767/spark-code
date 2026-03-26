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

MAX_WORKERS = 6

WORKER_SYSTEM_PROMPT = """You are a Spark Code worker agent running in the background.
You are completing a specific task assigned by the lead agent.
Focus on the task and use the tools available to complete it.
Be thorough but efficient — run tests if you write code.
When done, provide a brief summary of what you accomplished.

You have access to these tools:
- read_file: Read file contents. Parameters: file_path (required), offset, limit
- write_file: Create or overwrite files. Parameters: file_path (required), content (required)
- edit_file: Find & replace in files. Parameters: file_path (required), old_string (required), new_string (required)
- bash: Run shell commands. Parameters: command (required), timeout
- glob: Find files by pattern. Parameters: pattern (required), path
- grep: Search file contents. Parameters: pattern (required), path, glob, include
- list_dir: List directory contents. Parameters: path (required)
- web_search: Search the web. Parameters: query (required)
- web_fetch: Fetch web pages. Parameters: url (required)
- send_message: Send a message to another worker or the lead. Parameters: to (required), message (required)
  - Use to="lead" to message the main session
  - Use to="worker-1" to message a specific worker
  - Use to="broadcast" to message all workers

Guidelines:
- Always provide required parameters for every tool call
- Read files before editing them
- Use glob/grep to find files before reading
- If something fails, try a different approach
- Use send_message to coordinate with other workers when needed
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
    context_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    current_tool: str = ""


class TeamManager:
    """Manages background worker agents with messaging."""

    def __init__(self, model: ModelClient, tools: ToolRegistry,
                 console: Console, task_store: TaskStore):
        self.model = model
        self.tools = tools
        self.console = console
        self.task_store = task_store
        self.workers: dict[str, Worker] = {}
        self._counter = 0
        # Lead agent's inbox — messages from workers to "lead"
        self.lead_inbox: deque[Message] = deque()

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
                    w.inbox.append(msg)
                    # Queue message injection — will be picked up between agent rounds
                    if w.agent:
                        w.agent.context.add_user(
                            f"[Message from {from_name}]: {message}")
                    count += 1
            self.lead_inbox.append(msg)  # Lead also sees broadcasts
            return f"Message broadcast to {count} worker(s) and the lead."

        # Find target worker
        target = self._find_worker_by_name(to_name)
        if not target:
            return f"Error: Worker '{to_name}' not found. Active workers: {self._active_worker_names()}"

        target.inbox.append(msg)
        # Inject message into the target worker's context
        if target.agent:
            target.agent.context.add_user(
                f"[Message from {from_name}]: {message}")
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

    async def spawn(self, prompt: str, name: str = "") -> Worker | None:
        """Spawn a new worker agent with the given task."""
        # Reject duplicate names if that worker is still running
        if name:
            existing = self.get_worker(name)
            if existing and existing.status == "running":
                self.console.print(
                    Text(f"  Worker '{name}' is already running. "
                         f"Use a different name or wait for it to finish.",
                         style=_C_YELLOW))
                return None

        if self.active_count >= MAX_WORKERS:
            self.console.print(
                Text(f"  Max workers reached ({MAX_WORKERS}). "
                     f"Wait for one to finish or use /team stop.",
                     style=_C_YELLOW))
            return None

        self._counter += 1
        worker_id = str(self._counter)
        worker_name = name or f"worker-{worker_id}"

        # Create task in shared store
        task = self.task_store.create(prompt, assigned_to=worker_name)

        # Build a tool registry for this worker with its own send_message instance
        worker_tools = ToolRegistry()
        for tool in self.tools.all():
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

        # Each worker gets its own context and permissions (trust mode)
        context = Context(
            system_prompt=WORKER_SYSTEM_PROMPT + team_info,
            max_tokens=32768,
        )
        permissions = PermissionManager(mode="trust")

        # Create a prefixed console wrapper for worker output
        worker_console = _PrefixedConsole(self.console, worker_name)

        agent = Agent(
            model=self.model,
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
            result = await worker.agent.run(worker.prompt)
            worker.status = "completed"
            worker.result = result or "(completed with no text output)"

            self.task_store.update(
                task_id, status="completed",
                result=worker.result[:500],
            )

            self.lead_inbox.append(Message(
                from_name=worker.name,
                to_name="lead",
                content=f"[team] {worker.name} completed: {worker.result[:200]}"
            ))

            self.console.print(
                Text(f"  [{worker.name}] \u2713 Completed", style=_C_GREEN))

        except asyncio.CancelledError:
            worker.status = "failed"
            worker.result = "Cancelled"
            self.task_store.update(task_id, status="failed", result="Cancelled")
            self.console.print(
                Text(f"  [{worker.name}] Stopped", style=_C_YELLOW))

        except Exception as e:
            worker.status = "failed"
            worker.result = str(e)
            self.task_store.update(task_id, status="failed", result=str(e)[:500])
            self.console.print(
                Text(f"  [{worker.name}] \u2717 Failed: {e}", style=_C_RED))

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
        """Stop all running workers."""
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
        for w in self.workers.values():
            if w.name != writer_name and w.status == "running":
                w.inbox.append(Message(
                    from_name="team",
                    to_name=w.name,
                    content=msg_content,
                ))

    def get_worker(self, worker_id: str) -> Worker | None:
        return self.workers.get(worker_id)


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
