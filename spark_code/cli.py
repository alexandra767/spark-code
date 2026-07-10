"""Spark Code CLI — entry point."""

import asyncio
import base64
import mimetypes
import os
import subprocess
import sys
import uuid
from typing import Callable

import click
from prompt_toolkit.patch_stdout import patch_stdout
from rich.box import ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as _esc
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import Agent
from .agents_registry import load_agent_defs
from .branches import BranchManager
from .codesearch import CodeSearchTool, index_project, resolve_service_url
from .config import ensure_dirs, get, load_config, resolve_write_roots, set_config
from .context import AGENTIC_PROMPT, SYSTEM_PROMPT, Context, build_skill_prompt_section
from .corpus import export_session
from .custom_tools import BUILTIN_TOOL_NAMES, CustomToolRegistry
from .doctor import render_doctor, run_doctor
from .editor import detect_editor, open_in_editor
from .hooks import HookManager
from .instructions import load_instructions
from .keys import PROVIDER_PRESETS, load_keys, mask, resolve_provider_key, save_key
from .mcp.client import MCPClient
from .mcp.registry import find_mcp_configs
from .memory import Memory
from .model import PROVIDERS, ModelClient, _pick_real_model_name, resolve_real_model_name
from .permissions import (
    DISPLAY_ALIASES,
    NATIVE_MODES,
    PermissionManager,
    resolve_mode_name,
)
from .pinned import PinnedFiles
from .plan_executor import execute_plan
from .plan_mode import PlanState, cycle_mode
from .platform_info import format_platform_prompt
from .preflight import _extract_max_context, context_window_warning, fetch_server_models
from .project_detect import detect_project_type, detect_test_command
from .projectplan import extract_keywords, fetch_rag_context
from .routing import get_utility_client, resolve_utility_for
from .skills.base import SkillRegistry, check_skill_compatibility
from .snippets import SnippetLibrary
from .stats import SessionStats
from .task_store import TaskStore
from .team import TeamManager
from .todos import TodoList, TodoWriteTool
from .tool_cache import ToolCache
from .tools.base import ToolRegistry
from .tools.bash import BashTool
from .tools.edit_file import EditFileTool
from .tools.glob_search import GlobTool
from .tools.grep_search import GrepTool
from .tools.list_dir import ListDirTool
from .tools.rag_search import RagSearchTool
from .tools.read_file import ReadFileTool
from .tools.spawn_worker import SpawnWorkerTool
from .tools.stack_info import StackInfoTool
from .tools.wait_for_workers import WaitForWorkersTool
from .tools.web_fetch import WebFetchTool
from .tools.web_search import WebSearchTool
from .tools.write_file import WriteFileTool
from .ui.hotkeys import TeamStatusMonitor
from .ui.input import (
    RESERVED_COMMAND_NAMES,
    build_status_segments,
    cpr_confirmed_supported,
    create_session,
    format_status_line,
    renderer_cpr_support,
    resolve_statusline_mode,
    terminal_supports_cpr,
    toolbar_mode_segments,
    toolbar_status_segments,
)
from .ui.output import render_error
from .ui.theme import get_theme
from .watcher import FileWatcher


async def _warmup_model(model):
    """Send a tiny request to force model into VRAM."""
    try:
        async for _ in model.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[], stream=True
        ):
            break  # One chunk confirms model loaded
    except Exception:
        pass  # Silent — don't block startup


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    try:
        subprocess.run(
            ["pbcopy"], input=text.encode(), check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def load_spark_md() -> str:
    """Load SPARK.md or .spark/SPARK.md from the project root.

    Returns the file content, or empty string if not found.

    Deprecated for system-prompt/banner wiring — use
    ``spark_code.instructions.load_instructions()`` instead, which merges
    SPARK.md with the CLAUDE.md hierarchy (Claude Code parity). Kept here
    because other code/tests still import it directly.
    """
    for candidate in ("SPARK.md", os.path.join(".spark", "SPARK.md")):
        path = os.path.join(os.getcwd(), candidate)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError:
                pass
    return ""


def _get_git_info() -> str:
    """Get git branch + dirty status for the banner.

    Returns e.g. "main ✓" or "feature/foo *", or "" if not a git repo.
    """
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if branch.returncode != 0:
            return ""
        branch_name = branch.stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=3,
        )
        dirty = bool(status.stdout.strip())
        icon = " *" if dirty else " ✓"
        return f"{branch_name}{icon}"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _make_session_label(context) -> str:
    """Extract a short label from the first user message in the session."""
    import re
    for msg in context.messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Clean up: remove newlines, strip, truncate
                label = content.strip().split("\n")[0][:50]
                # Sanitize for filename: lowercase, replace non-alnum with hyphens
                label = re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')
                return label[:40]  # Keep filename reasonable
    return ""


def _checkpoint_resume_note(round_count: int) -> dict:
    """Build the note appended when resuming a checkpoint via /continue.

    Uses role:"user" (not "system") so it never injects a second, non-leading
    system message — get_messages() already prepends the real system prompt at
    index 0, and a mid-history system role is rejected by strict OpenAI-compatible
    servers. Mirrors how compaction injects its summary as a user message.
    """
    return {
        "role": "user",
        "content": (f"Session resumed from checkpoint at round "
                    f"{round_count}/{Agent.MAX_TOOL_ROUNDS}. "
                    f"Continue where you left off."),
    }


def _restore_checkpoint(context, data: dict) -> None:
    """Restore a checkpoint's messages into ``context`` for /continue.

    Re-establishes ``turn_count`` (from the saved ``round_count``, falling back
    to the message count) so the exit-auto-save guard (``turn_count > 0``) still
    fires in a fresh process — otherwise post-continue work is never written to
    history. Appends a role:"user" resume note. Copies the message list so the
    original checkpoint data is not mutated.
    """
    context.messages = list(data.get("messages") or [])
    context.turn_count = data.get("round_count") or len(context.messages)
    context.messages.append(_checkpoint_resume_note(data.get("round_count", 0)))


def _redacted_config(config: dict) -> dict:
    """Deep-copy config with any api_key values masked, for safe display."""
    import copy

    def _mask(obj):
        if isinstance(obj, dict):
            return {k: ("***redacted***" if "api_key" in k.lower() and v else _mask(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [_mask(v) for v in obj]
        return obj

    return _mask(copy.deepcopy(config))


def _export_corpus_session(config: dict, context, agent, *, error: str | None) -> None:
    """Best-effort session -> training corpus export (Phase 4 Task 6).

    Called from both teardown points (run_interactive's finally, and after
    a headless _one_shot run) so every completed session — interactive or
    scripted — gets one export attempt. Opt-in gating and secret scrubbing
    live in :func:`spark_code.corpus.export_session` itself (strict-coerced
    ``enabled``, "skip if errored", "skip if no messages"); this wrapper
    supplies the metadata each call site has available and swallows any
    exception, because a broken corpus export must never surface as a
    session-ending crash (interactive) or a headless-run failure (exit
    code / JSON contract untouched either way).

    Interruption handling: a Ctrl+C / task-cancel returns via the agent's
    ``if self._cancelled: return`` branch WITHOUT setting a stream error, so
    ``error`` alone can't tell a clean completion from an interrupted one.
    We treat the session as errored (-> skipped) when the last turn was
    cancelled (``agent._cancelled``) OR any earlier turn was interrupted
    (the persistent ``agent._session_interrupted`` flag, set in the cancel
    branch and never reset within a session) — an interrupted session may
    hold partial/orphaned turns and must never be exported as "clean".
    """
    try:
        from datetime import datetime, timezone
        interrupted = (getattr(agent, "_cancelled", False)
                       or getattr(agent, "_session_interrupted", False))
        effective_error = error or ("interrupted" if interrupted else None)
        export_session(
            context,
            get(config, "corpus", "dir", default="~/training-corpus/spark-code"),
            meta={
                "enabled": get(config, "corpus", "export_enabled", default=False),
                "error": effective_error,
                "model": get(config, "model", "name", default=""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "files_changed": list(getattr(agent, "_verify_written_paths", []) or []),
                "session_id": uuid.uuid4().hex,
            },
        )
    except Exception:
        pass


def _sessions_dir(create: bool = True) -> str:
    """Return the session-history directory, normalizing a legacy collision.

    ``~/.spark/history`` was historically (mis)used as the prompt-toolkit input
    history FILE, which shadows this DIRECTORY of session JSONs — the reason
    --resume/--continue/history/fork silently never worked. If that stray file
    is present, move it aside (the input history now lives at
    ``~/.spark/input_history``) so the directory can be created.
    """
    d = os.path.expanduser("~/.spark/history")
    if os.path.isfile(d):
        legacy = os.path.expanduser("~/.spark/input_history")
        try:
            if not os.path.exists(legacy):
                os.rename(d, legacy)
            else:
                os.remove(d)
        except OSError:
            pass
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _get_latest_session() -> str:
    """Return the path to the most recent session file, or empty string."""
    history_dir = _sessions_dir(create=False)
    if not os.path.isdir(history_dir):
        return ""
    sessions = sorted(
        [f for f in os.listdir(history_dir) if f.endswith(".json")],
        reverse=True,
    )
    if sessions:
        return os.path.join(history_dir, sessions[0])
    return ""


# How many sessions /history and /resume actually display. _list_sessions
# eagerly parses metadata for exactly this many entries, so the display
# slice and the metadata cap can't drift apart.
_SESSION_DISPLAY_LIMIT = 10


def _load_session_meta(entry: dict) -> dict:
    """Fill label/age/turns/cwd on a _list_sessions() entry (idempotent).

    Metadata needs a full JSON parse of the session file, so it is loaded
    only for entries that are actually displayed or name-matched — never
    for the whole history directory (review amendment 2026-07-06).
    """
    if entry.get("_meta_loaded"):
        return entry
    from datetime import datetime as _dt

    meta = Context.read_metadata(entry["path"])
    age = ""
    ts = meta.get("timestamp", "")
    if ts:
        try:
            session_time = _dt.fromisoformat(ts)
            delta = _dt.now() - session_time
            if delta.days > 0:
                age = f"{delta.days}d ago"
            elif delta.seconds >= 3600:
                age = f"{delta.seconds // 3600}h ago"
            elif delta.seconds >= 60:
                age = f"{delta.seconds // 60}m ago"
            else:
                age = "just now"
        except (ValueError, TypeError):
            pass
    entry["label"] = meta.get("label", "")
    entry["age"] = age
    entry["turns"] = meta.get("turn_count", 0)
    entry["cwd"] = meta.get("cwd", "")
    entry["_meta_loaded"] = True
    return entry


def _list_sessions(meta_limit: int = _SESSION_DISPLAY_LIMIT) -> list[dict]:
    """Return every saved session, newest first.

    Each entry: path, name (filename minus ``.json``), label, age (human
    string like "2h ago", "" if unknown), turns, cwd. Metadata requires a
    full JSON parse, so it is eagerly loaded only for the first
    `meta_limit` entries — the slice /history and /resume actually
    display. Later entries keep placeholder metadata; callers that
    name-match one of them hydrate just that entry via
    _load_session_meta(). This is the single place that reads
    ``~/.spark/history`` off disk for display purposes — /history's
    listing and /resume's picker both consume it so they can't drift out
    of sync with each other.
    """
    history_dir = _sessions_dir()
    if not os.path.isdir(history_dir):
        return []
    files = sorted(
        [f for f in os.listdir(history_dir) if f.endswith(".json")],
        reverse=True,
    )
    sessions = []
    for i, f in enumerate(files):
        entry = {
            "path": os.path.join(history_dir, f),
            "name": f[:-len(".json")],
            "label": "",
            "age": "",
            "turns": 0,
            "cwd": "",
            "_meta_loaded": False,
        }
        if i < meta_limit:
            _load_session_meta(entry)
        sessions.append(entry)
    return sessions


def _resume_session_into(context: Context, path: str) -> bool:
    """Load a saved session file into a live Context.

    Shared by the startup ``--resume``/``--continue`` path, ``/history
    <name>``, and ``/resume`` so there's exactly one seam that knows how to
    swap a session into a running context (``Context.load`` already
    repairs orphaned tool_calls; this just gives that call one name
    instead of three copies).
    """
    return context.load(path)


def _notify_done():
    """Play a notification sound (macOS bell)."""
    try:
        # macOS: use afplay with system sound
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # Fallback: terminal bell
        print("\a", end="", flush=True)


# Shell command prefixes — when user types these, run directly via bash
_SHELL_PREFIXES = (
    "python ", "python3 ", "pip ", "pip3 ",
    "node ", "npm ", "npx ", "yarn ", "bun ",
    "cargo ", "go ", "make ", "cmake ",
    "docker ", "docker-compose ",
    "git ", "gh ", "ls ", "cat ", "cd ", "mkdir ", "rm ", "cp ", "mv ",
    "curl ", "wget ", "chmod ", "brew ",
    "swift ", "swiftc ", "javac ", "java ",
    "gcc ", "g++ ", "clang ",
    "pytest ", "jest ", "ruby ",
    "./",
)


# Prefixes that are also common English verbs — need a natural-language guard
# so "make the tests pass" / "go through the code" aren't run as shell.
_AMBIGUOUS_PREFIXES = {"make", "go", "cat", "cd", "rm", "cp", "mv", "ls"}
_ENGLISH_AFTER_VERB = {
    "the", "a", "an", "me", "my", "this", "that", "it", "them", "us", "some",
    "your", "all", "through", "sure", "then", "again", "everything", "one",
    "another", "yourself", "him", "her", "please", "and", "to", "into", "up",
    "down", "over", "back", "out", "off", "on", "in",
}


def _is_shell_command(text: str) -> bool:
    """Check if input looks like a direct shell command.

    For prefixes that double as English verbs (make/go/cat/...), reject inputs
    that read as a sentence ("make the tests pass") so they go to the model
    instead of being executed literally.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("./"):
        return True
    if not stripped.startswith(_SHELL_PREFIXES):
        return False
    parts = stripped.split()
    verb = parts[0].lower()
    if verb in _AMBIGUOUS_PREFIXES:
        if len(parts) >= 2 and parts[1].lower() in _ENGLISH_AFTER_VERB:
            return False
        if len(parts) > 6:  # real invocations of these verbs are short
            return False
    return True


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg")

# ---------------------------------------------------------------------------
# Provider info — shared by /providers and --setup
# ---------------------------------------------------------------------------

_PROVIDER_INFO = [
    {
        "name": "gemini",
        "label": "Google Gemini",
        "model": "gemini-2.5-flash",
        "env_var": "GEMINI_API_KEY",
        "signup": "https://aistudio.google.com/apikey",
    },
    {
        "name": "openai",
        "label": "OpenAI",
        "model": "gpt-4o-mini",
        "env_var": "OPENAI_API_KEY",
        "signup": "https://platform.openai.com/api-keys",
    },
    {
        "name": "groq",
        "label": "Groq",
        "model": "llama-3.3-70b-versatile",
        "env_var": "GROQ_API_KEY",
        "signup": "https://console.groq.com/keys",
    },
    {
        "name": "deepseek",
        "label": "DeepSeek",
        "model": "deepseek-chat",
        "env_var": "DEEPSEEK_API_KEY",
        "signup": "https://platform.deepseek.com/api_keys",
    },
    {
        "name": "openrouter",
        "label": "OpenRouter",
        "model": "anthropic/claude-sonnet-4",
        "env_var": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/keys",
    },
    {
        "name": "ollama",
        "label": "Ollama (local)",
        "model": "qwen3.5:122b",
        "env_var": "",
        "signup": "https://ollama.ai",
    },
    {
        "name": "sglang",
        "label": "SGLang (fast local)",
        "model": "Qwen3.5-122B-A10B-NVFP4",
        "env_var": "",
        "signup": "https://github.com/sgl-project/sglang",
    },
]


def _is_image_drop(text: str) -> tuple[str, str]:
    """Detect if input is a dragged-in image file path.

    macOS pastes the full path when you drag a file into the terminal.
    Sometimes paths are quoted or escaped.

    Returns (file_path, remaining_text) or ("", "") if not an image.
    """
    # Clean up: strip quotes, unescape spaces
    cleaned = text.strip().strip("'\"")
    cleaned = cleaned.replace("\\ ", " ")

    # Check if it starts with a path-like string
    # Could be absolute (/Users/...) or home-relative (~/...)
    parts = cleaned.split(maxsplit=1)
    candidate = parts[0] if parts else cleaned

    # Also check the full cleaned string (path might have spaces)
    for path_str in [cleaned, candidate]:
        lower = path_str.lower()
        if any(lower.endswith(ext) for ext in _IMAGE_EXTENSIONS):
            expanded = os.path.expanduser(path_str)
            if os.path.isfile(expanded):
                remaining = parts[1] if len(parts) > 1 and path_str == candidate else ""
                return expanded, remaining

    return "", ""


def _detect_file_mentions(text: str) -> list[str]:
    """Extract file paths mentioned in user input for auto-reading.

    Looks for patterns like: fix auth.py, look at src/main.rs, edit config.yaml
    Also honors explicit Claude Code-style `@path` mentions (files or
    directories), which take priority over the looser heuristic below.
    Returns list of existing paths (mentions first, then heuristic matches).
    """
    import re
    found = []

    # `@`-prefixed mentions: @auth.py, @src/app.py, @src (dir). The
    # (?:^|\s) guard requires the "@" to open the token (start of string or
    # preceded by whitespace) so mid-token "@" in emails like
    # "alexandra@gmail.com" is never matched.
    at_pattern = r"(?:^|\s)@([\w./-]+)"
    for token in re.findall(at_pattern, text):
        # "see @app.py." captures "app.py." — "." is in the character class,
        # so sentence punctuation rides along and the exists() check would
        # silently drop the mention. Retry with trailing punctuation stripped.
        if not os.path.exists(token):
            stripped = token.rstrip(".,;:!?)\"'")
            if not stripped or not os.path.exists(stripped):
                continue
            token = stripped
        if token not in found:
            found.append(token)

    # Match common file path patterns (word.ext or path/to/file.ext)
    pattern = r'(?:^|\s)((?:[\w./~-]+/)?[\w.-]+\.(?:py|js|ts|jsx|tsx|rs|go|java|kt|swift|rb|c|cpp|h|hpp|css|html|yaml|yml|toml|json|md|txt|sh|sql|env))\b'
    matches = re.findall(pattern, text)
    for match in matches:
        path = os.path.expanduser(match)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if os.path.isfile(path) and path not in found:
            found.append(path)
    return found


async def _auto_read_context(paths: list[str], console) -> str:
    """Build auto-read context blocks for file/directory mentions.

    Files are read directly. Directories can't be `open()`-ed (that raises
    IsADirectoryError, silently swallowed by the old file-only code path),
    so they're rendered through the `list_dir` tool instead. Returns the
    joined context blocks (empty string if nothing could be read).
    """
    home = os.path.expanduser("~")
    blocks = []
    for fpath in paths:
        display = "~" + fpath[len(home):] if fpath.startswith(home) else fpath
        if os.path.isdir(fpath):
            listing = await ListDirTool().execute(path=fpath)
            blocks.append(f"Directory listing for {display}:\n```\n{listing}\n```")
            console.print(f"  [#4c566a]Auto-listed: {display}[/#4c566a]")
            continue
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                content = f.read(10000)
            blocks.append(f"Contents of {display}:\n```\n{content}\n```")
            console.print(f"  [#4c566a]Auto-read: {display}[/#4c566a]")
        except OSError:
            pass
    return "\n\n".join(blocks)


def _is_error_paste(text: str) -> bool:
    """Detect if pasted text looks like an error/traceback."""
    indicators = [
        "Traceback (most recent call last)",
        "Error:", "error:", "ERROR:",
        "TypeError:", "ValueError:", "KeyError:", "AttributeError:",
        "ImportError:", "ModuleNotFoundError:", "FileNotFoundError:",
        "SyntaxError:", "IndentationError:", "NameError:",
        "npm ERR!", "FAIL ", "FAILED",
        "panic:", "fatal error:",
        "Exception in thread",
        "at Object.<anonymous>",
        "Cannot find module",
        "undefined is not",
    ]
    line_count = text.count("\n")
    # Multi-line text with error indicators
    return line_count >= 2 and any(ind in text for ind in indicators)


def _context_pct_style(pct: float) -> str:
    """Toolbar style class for a context-left percentage (0-100): green when
    there's plenty of room, yellow approaching the limit, red about to
    auto-compact/400."""
    if pct > 40:
        return "class:bottom-toolbar.context-green"
    if pct >= 15:
        return "class:bottom-toolbar.context-yellow"
    return "class:bottom-toolbar.context-red"


def _agent_only_servers(config) -> set:
    """MCP servers marked ``agent_only: true`` in config. Their tools are still
    threaded to sub-agents (via DispatchAgentTool), but NOT registered on the
    main model — so the main model can't call them directly. This exists for
    the browser: a page snapshot is huge, and driving the browser from the
    local 32K-context model floods and overflows it, so only the Gemini
    web-driver sub-agent gets the playwright tools."""
    servers = get(config, "mcp_servers", default={}) or {}
    return {name for name, conf in servers.items()
            if isinstance(conf, dict) and conf.get("agent_only")}


def _register_mcp_tools(registry, mcp_tools, config) -> int:
    """Register MCP tools on the main registry, skipping agent-only servers.
    Returns the number skipped (still available to sub-agents via dispatch)."""
    agent_only = _agent_only_servers(config)
    skipped = 0
    for t in mcp_tools:
        if getattr(t, "server_name", None) in agent_only:
            skipped += 1
            continue
        registry.register(t)
    return skipped


def build_tools(todo_list: TodoList | None = None, model=None,
                config: dict | None = None,
                permissions: "PermissionManager | None" = None,
                plan_state: "PlanState | None" = None,
                console: Console | None = None,
                agent_defs: dict | None = None,
                utility_model=None,
                mcp_tools=None,
                interactive: bool = True) -> ToolRegistry:
    """Register all built-in tools.

    ``todo_list`` is the session-scoped live checklist backing ``todo_write``.
    Callers that want to inspect/share it (e.g. to create it once per session
    alongside the Context) pass their own instance; otherwise a fresh, empty
    one is created here — either way state is never persisted across
    sessions.

    ``model``/``config`` (and optionally ``permissions``) enable the
    ``dispatch_agent`` tool: only registered when a model AND config are
    supplied. Sub-agents build their tool registry from a plain ``build_tools()``
    (no model) so it structurally CANNOT contain ``dispatch_agent`` — that is
    the depth guard against nesting. ``permissions`` is read live at dispatch
    time so a mid-session mode change (Shift+Tab) is inherited by sub-agents.

    ``plan_state`` enables the ``exit_plan_mode`` tool — the approval gate for
    true plan mode. Only registered when a ``plan_state`` is supplied (the
    interactive session), so one-shot runs and sub-agents (which pass none)
    never see it.

    ``agent_defs`` (Phase 5 Task 4) is the ``.spark/agents/*.md`` +
    ``~/.spark/agents/*.md`` custom subagent map (``agents_registry.
    load_agent_defs``) — threaded into ``DispatchAgentTool`` so its
    ``agent_type`` enum/description/dispatch gain the custom names. Omitted
    (``None``/``{}``) leaves dispatch_agent's schema exactly as it was before
    this feature existed. ``utility_model`` (Phase 4 Task 2) is likewise only
    consulted for a custom def whose ``model_hint == "utility"``.

    ``mcp_tools`` (browser/MCP bugfix) are the caller's already-connected MCP
    tool instances (e.g. Playwright browser control) — threaded only into
    ``DispatchAgentTool`` so a sub-agent it later dispatches can actually see
    them (see ``dispatch.run_subagent``'s docstring). This function's OWN
    registry never registers MCP tools directly; the caller (``run_interactive``
    / ``_one_shot``) still registers them onto its own top-level registry
    itself, exactly as before this parameter existed.

    ``interactive`` (App Store submission task) is threaded into
    ``AppStoreSubmitTool`` so a headless run (``_one_shot``, ``interactive=
    False``) additionally refuses to submit for review even when asked to —
    it can only ever produce the no-op summary — while the interactive
    session (``run_interactive``, ``interactive=True``) may proceed given an
    explicit ``confirm="submit"``. Defaults to ``True`` so sub-agents built
    from a bare ``build_tools()`` (no caller override) keep today's behavior.
    ``AppStoreBuildUploadTool`` (build-upload task) receives the same
    ``interactive`` flag for the identical reason, gating its own outward
    step (``altool --upload-app``) on ``confirm="upload"``.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    # Phase 5 Task 1: permissions.write_roots — extra directory trees (e.g. a
    # sibling repo) writes/edits may target on top of cwd + system temp.
    # Resolved once here (expanded/realpath'd, non-existent roots skipped) and
    # threaded into the tool instances, mirroring how CodeSearchTool/
    # DispatchAgentTool receive `config` at construction time.
    write_roots = resolve_write_roots(config) if config else []
    # SECURITY (Phase 5 whole-branch review, Finding 1b): anchor the sandbox
    # root at THIS process's real cwd, captured once, at construction time —
    # not left as None (which would fall back to a live os.getcwd() call
    # inside _validate_path, but ALSO let a model-supplied cwd argument
    # freely override that fallback, since None never wins the `self.cwd or
    # cwd` precedence in write_file.py/edit_file.py). Safe to freeze here:
    # this codebase never calls os.chdir() at runtime (see dispatch.py's
    # _scope_registry_for_worktree docstring), so a captured os.getcwd() and
    # a live one are always the same value for this process's lifetime.
    # Isolated dispatches still get their OWN narrower cwd (the worktree
    # path) via fresh instances in dispatch.py's _scope_registry_for_worktree
    # — this only closes the gap for the non-isolated (trust-mode/main-agent)
    # registry built here.
    registry.register(WriteFileTool(write_roots=write_roots, cwd=os.getcwd()))
    registry.register(EditFileTool(write_roots=write_roots, cwd=os.getcwd()))
    registry.register(BashTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(ListDirTool())
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    registry.register(RagSearchTool(config=config))
    registry.register(StackInfoTool())
    from .tools.appstore_status import AppStoreStatusTool
    registry.register(AppStoreStatusTool())
    from .tools.play_status import PlayStatusTool
    registry.register(PlayStatusTool())
    from .tools.appstore_submit import AppStoreSubmitTool
    registry.register(AppStoreSubmitTool(interactive=interactive))
    from .tools.appstore_build_upload import AppStoreBuildUploadTool
    registry.register(AppStoreBuildUploadTool(interactive=interactive))
    from .tools.play_publish import PlayPublishTool
    registry.register(PlayPublishTool(interactive=interactive))
    # Phase 4 Task 3: always registered (no startup network call — see
    # codesearch.py) unless the user explicitly disabled it; away from the
    # RAG host it just returns a friendly unavailable message per-query.
    if config is None or get(config, "rag", "code_search_enabled", default=True):
        registry.register(CodeSearchTool(config=config))
    registry.register(TodoWriteTool(todo_list if todo_list is not None else TodoList()))
    # Phase 4 Task 4: iOS vision loop — only registered on macOS with Xcode's
    # command line tools installed (`xcrun`/`simctl` on the PATH); absent
    # everywhere else (e.g. this project's own Linux/DGX host) — no error,
    # just no tool. Registered even when `model` is None (sub-agents) since
    # the guard here is purely platform-level; the tool itself degrades to
    # its non-vision message if there's no model to check.
    import platform
    import shutil
    if platform.system() == "Darwin" and shutil.which("xcrun"):
        from .tools.simulator import SimulatorScreenshotTool
        registry.register(SimulatorScreenshotTool(model=model))
    # Phase (phone-screenshot vision): real-device screen capture, described
    # via Gemini vision (spark_code.vision.describe_image) rather than fed
    # directly to the primary model. Unlike SimulatorScreenshotTool above,
    # these are NOT gated on macOS/Xcode — adb works cross-platform and both
    # tools degrade to a clean "no device/binary" message on their own when
    # the underlying tool isn't present, so registering them unconditionally
    # costs nothing and needs no host-capability probe here.
    from .tools.android_screenshot import AndroidScreenshotTool
    from .tools.ios_screenshot import IosScreenshotTool
    from .tools.watch_phone import WatchPhoneTool
    registry.register(AndroidScreenshotTool())
    registry.register(IosScreenshotTool())
    registry.register(WatchPhoneTool())
    from .tools.android_control import AndroidControlTool
    registry.register(AndroidControlTool())
    from .tools.ios_launch import IosLaunchTool
    registry.register(IosLaunchTool())
    if model is not None and config is not None:
        from .dispatch import BrowseWebTool, DispatchAgentTool
        registry.register(DispatchAgentTool(
            model=model, config=config, permissions=permissions,
            agent_defs=agent_defs, utility_model=utility_model,
            mcp_tools=mcp_tools))
        # A correctly-named browser affordance on the main model. Without it
        # the 80B (which can't see the agent_only playwright tools) invents a
        # nonexistent `web_driver` tool instead of dispatch_agent(agent_type=
        # "web-driver"); this routes cleanly to the Gemini web-driver sub-agent.
        registry.register(BrowseWebTool(
            model=model, config=config, permissions=permissions,
            agent_defs=agent_defs, utility_model=utility_model,
            mcp_tools=mcp_tools))
    if plan_state is not None:
        from .plan_mode import ExitPlanModeTool
        registry.register(ExitPlanModeTool(
            plan_state, permissions=permissions, console=console))
    return registry


_SPARK_LOGO = """\
    [bold #ebcb8b]▄▄[/]
   [bold #ebcb8b]▟██[/]
  [bold #ebcb8b]▟██▀[/]
 [bold #ebcb8b]▟████▙[/]
  [bold #ebcb8b]▀██▄[/]
   [bold #ebcb8b]▀██▄[/]
    [bold #ebcb8b]▀▀[/]\
"""


def print_banner(console: Console, config: dict, mcp_count: int = 0,
                 skill_count: int = 0, instruction_sources: list[str] | None = None,
                 project_type: str = "", real_model_name: str | None = None):
    """Print startup banner — two-column layout matching Claude Code."""
    config_name = get(config, "model", "name", default="unknown")
    display_name = get(config, "model", "display_name", default="")
    # A configured display_name wins and suppresses the "(served as <id>)" alias;
    # otherwise fall back to the auto-resolved real name (with the alias shown).
    if display_name:
        model_name = display_name
        show_served_as = False
    else:
        model_name = real_model_name or config_name
        show_served_as = bool(real_model_name and real_model_name != config_name)
    provider = get(config, "model", "provider", default="")
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        cwd = "~" + cwd[len(home):]

    git_info = _get_git_info()

    # Left column: logo + identity
    left = Text()
    left.append("\n")
    for line in _SPARK_LOGO.strip().split("\n"):
        left.append("  ")
        left.append_text(Text.from_markup(line))
        left.append("\n")
    left.append("\n")
    left.append(f"  {model_name}", style="#88c0d0")
    if show_served_as:
        left.append(f"  (served as {config_name})", style="#4c566a")
    if provider:
        left.append(f" · {provider}", style="#4c566a")
    left.append(f"\n  {cwd}", style="#4c566a")
    if git_info:
        left.append(f"  ({git_info})", style="#a3be8c" if "✓" in git_info else "#ebcb8b")

    # Right column: tips and info
    right = Text()
    right.append("Tips for getting started\n", style="bold #eceff4")
    right.append("Run ", style="#8899aa")
    right.append("/help", style="bold #d8dee9")
    right.append(" for available commands\n", style="#8899aa")

    right.append("─" * 35 + "\n", style="#3b4252")

    right.append("Capabilities\n", style="bold #eceff4")
    right.append("Read, write, and edit files\n", style="#8899aa")
    right.append("Run shell commands\n", style="#8899aa")
    right.append("Search code with glob/grep\n", style="#8899aa")
    right.append("Web search and fetch\n", style="#8899aa")
    right.append("Send images with /image\n", style="#8899aa")

    extras = (mcp_count > 0 or skill_count > 0 or instruction_sources or project_type)
    if extras:
        right.append("─" * 35 + "\n", style="#3b4252")
        for name in instruction_sources or []:
            right.append(f"{name} loaded\n", style="#a3be8c")
        if project_type:
            right.append(f"{project_type}\n", style="#88c0d0")
        if mcp_count > 0:
            right.append(f"MCP: {mcp_count} tool(s) loaded\n", style="#a3be8c")
        if skill_count > 0:
            right.append(f"Skills: {skill_count} available\n", style="#a3be8c")

    # Build two-column table
    table = Table(show_header=False, show_edge=False, box=None,
                  padding=(0, 2), expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_row(left, right)

    title = Text()
    title.append(f" Spark Code v{__version__} ", style="bold #ebcb8b")

    console.print(Panel(table, title=title, border_style="#4c566a",
                        box=ROUNDED, padding=(0, 1)))
    console.print()


def _undo_last(console: Console, count: int = 1) -> int:
    """Restore the most recent `count` file snapshots from the undo stack.

    Prints status via `console` exactly as /undo's inline body used to.
    Shared by /undo and /rewind's "last file edits" option so there's one
    place that knows how to pop the undo stack. Returns the number of
    operations actually restored.
    """
    undo_dir = os.path.expanduser("~/.spark/.undo")
    if not os.path.isdir(undo_dir):
        console.print("[#8899aa]Nothing to undo.[/#8899aa]")
        return 0
    undo_files = sorted(os.listdir(undo_dir), reverse=True)
    if not undo_files:
        console.print("[#8899aa]Nothing to undo.[/#8899aa]")
        return 0

    count = min(count, len(undo_files))
    restored = 0
    import json as _json
    for uf in undo_files[:count]:
        undo_meta_path = os.path.join(undo_dir, uf)
        try:
            with open(undo_meta_path, encoding="utf-8") as f:
                undo_data = _json.load(f)
            original_path = undo_data["path"]
            original_content = undo_data["content"]
            with open(original_path, "w", encoding="utf-8") as f:
                f.write(original_content)
            os.remove(undo_meta_path)
            home = os.path.expanduser("~")
            display = "~" + original_path[len(home):] if original_path.startswith(home) else original_path
            console.print(f"[#a3be8c]Restored: {display}[/#a3be8c]")
            restored += 1
        except Exception as e:
            console.print(f"[#bf616a]Undo failed: {e}[/#bf616a]")
    if restored > 1:
        console.print(f"[#a3be8c]Undid {restored} operations[/#a3be8c]")
    return restored


def _get_checkpoints() -> list[tuple[str, str]] | None:
    """Return (real_stash_ref, description) for each spark-checkpoint stash
    entry, newest first, or None if git itself is unavailable.

    Shared by /rollback and /rewind's "checkpoint" option so there's one
    place that knows how to read the checkpoint stash list.
    """
    try:
        result = subprocess.run(
            ["git", "stash", "list"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    checkpoints = []
    for line in result.stdout.strip().split("\n"):
        if line and "spark-checkpoint" in line:
            ref = line.split(":", 1)[0].strip()  # "stash@{2}"
            checkpoints.append((ref, line))
    return checkpoints


def _apply_checkpoint(console: Console, checkpoints: list[tuple[str, str]], idx: int) -> bool:
    """Restore the working tree to the checkpoint at `idx` (0 = most recent).

    `apply` (not `pop`) so the checkpoint survives and can be rolled back to
    again. Shared by /rollback <N> and /rewind's "checkpoint" option.
    """
    if idx < 0 or idx >= len(checkpoints):
        console.print(f"[#bf616a]No checkpoint {idx}. Use /rollback to list.[/#bf616a]")
        return False
    real_ref = checkpoints[idx][0]
    try:
        restore_result = subprocess.run(
            ["git", "stash", "apply", real_ref],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        console.print("[#bf616a]Git not available[/#bf616a]")
        return False
    if restore_result.returncode == 0:
        console.print(f"[#a3be8c]Rolled back to checkpoint {idx}[/#a3be8c]")
        return True
    console.print(f"[#bf616a]{restore_result.stderr.strip()}[/#bf616a]")
    return False


def _rewind_files(console: Console) -> bool:
    """Restore the working tree to the most recent checkpoint.

    Shared by /rewind's "checkpoint" and "both" options.
    """
    checkpoints = _get_checkpoints()
    if checkpoints is None:
        console.print("[#bf616a]Git not available[/#bf616a]")
        return False
    if not checkpoints:
        console.print("[#8899aa]No checkpoints found. Use /checkpoint first.[/#8899aa]")
        return False
    return _apply_checkpoint(console, checkpoints, 0)


def _rewind_conversation(console: Console, context: Context) -> bool:
    """Restore the conversation to its state before the most recent turn.

    Phase 5 Task 7: unlike ``_undo_last``/``_rewind_files`` (which restore
    FILES), this restores the actual message history via
    ``Context.rewind_conversation`` — closing the Phase 1 honest-labels gap
    where /rewind could only ever touch files, never the conversation
    itself. Shared by /rewind's "conversation" and "both" options.
    """
    if context.rewind_conversation(1):
        console.print("[#a3be8c]Rewound conversation to before the last turn[/#a3be8c]")
        return True
    console.print("[#8899aa]No conversation snapshot to rewind to.[/#8899aa]")
    return False


async def _handle_compact(model, context: Context, console: Console,
                          instructions: str | None, utility_model=None):
    """Run the /compact flow: model summary, then compact — Ctrl+C-safe.

    Every sibling sentinel branch in the REPL guards its await with a
    KeyboardInterrupt handler; the __COMPACT__ branch initially didn't, so
    Ctrl+C during the (long) summary request unwound past the REPL loop and
    killed the whole session. Catch it here: cancel cleanly, touch nothing.

    ``utility_model`` (Phase 4 Task 2, already use_for-gated by the caller
    via ``routing.resolve_utility_for`` — same as the auto-compact path in
    Agent._maybe_compact) is preferred for the summary request, silently
    falling back to ``model`` on failure.
    """
    from spark_code.agent import generate_compaction_summary
    from spark_code.routing import call_with_fallback
    try:
        # Summary-request tokens count toward ModelClient lifetime /stats
        # totals — intentional: those tokens were really spent.
        summary = await call_with_fallback(
            utility_model, model,
            lambda m: generate_compaction_summary(m, context, instructions))
    except KeyboardInterrupt:
        console.print("\n[#ebcb8b]Compaction cancelled.[/#ebcb8b]")
        return
    except Exception as e:
        console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
        return
    compact_msg = context.compact(summary=summary)
    if compact_msg:
        console.print(f"  [#ebcb8b]⚡ {compact_msg}[/#ebcb8b]")
    else:
        console.print("[green]Nothing to compact.[/green]")


def handle_slash_command(cmd: str, context: Context, console: Console,
                         config: dict, skills: SkillRegistry,
                         model: ModelClient,
                         permissions: PermissionManager | None = None,
                         team_manager: TeamManager | None = None,
                         task_store: TaskStore | None = None,
                         memory: Memory | None = None,
                         stats: SessionStats | None = None,
                         pinned: PinnedFiles | None = None,
                         snippets: SnippetLibrary | None = None,
                         plan_state: PlanState | None = None) -> str | None:
    """Handle slash commands.
    Returns None if handled (no agent needed), or a prompt string for the agent.
    Returns "__ASYNC__" for commands that schedule async work (team spawn).
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        help_text = """## Commands

**Session**
- `/help` — Show this help
- `/clear` — Clear conversation
- `/compact` — Summarize conversation to save context
- `/tokens` — Show token usage
- `/stats` — Show session statistics
- `/cost` — Show session costs and budget
- `/analytics` — Detailed analytics dashboard
- `/history` — List and resume past sessions
- `/resume [n]` — Pick a past session from a numbered list and resume it
- `/search <query>` — Search past sessions by keyword
- `/export` — Export session as markdown
- `/share [gist]` — Export as shareable HTML (or markdown for gist)
- `/fork` — Save current + start fresh context

**Branching**
- `/branch [name]` — Create/list conversation branches
- `/switch <name>` — Switch to a branch
- `/branches` — List all branches

**Code & Files**
- `/diff` — Show git diff with syntax highlighting
- `/review [path]` — Multi-agent review swarm on the diff (3 lenses + skeptic verify)
- `/undo [N]` — Undo last N file operations (`/undo list` to see stack)
- `/pin <file>` — Pin a file to always stay in context
- `/unpin <file>` — Remove a pinned file
- `/open <file[:line]>` — Open a file in your editor (Cursor/VS Code/Xcode)
- `/watch <cmd>` — Auto-run command on file changes (`/watch off` to stop)
- `/checkpoint` — Create a restorable checkpoint (git stash)
- `/rollback [N]` — Restore from a checkpoint
- `/rewind` — Restore files (undo stack/checkpoint) or the conversation itself (unified menu)
- `/continue` — Resume from last checkpoint
- `/retry` — Re-send the last message
- `/clean` — Delete files created this session
- `/apply <url>` — Apply code from a URL (gist, PR, diff)

**Model & Config**
- `/config` — Show current config / `/config set <key> <value>`
- `/model` — Show model info / `/model list` / `/model <provider>`
- `/providers` — Show API providers with signup URLs
- `/setkey <provider>` — Guided cloud API key setup (masked prompt, e.g. `/setkey openrouter`)
- `/doctor` — Health check: engine, RAG, MCP, editor, keys (read-only, safe anytime)
- `/hooks` — List configured hooks (event, pattern, command, timeout) — read-only, see docs/hooks.md
- `/profile` — Benchmark model (TTFT, tokens/sec)
- `/benchmark` — Measure model speed (TTFT, tok/s)
- `/mode [ask|auto|trust]` — Switch permission mode
- `/trust` / `/auto` / `/ask` — Quick mode switch
- `/yolo` — Toggle agent mode (autonomous + trust all)
- `/loop [every 30m] [--check "cmd"] <goal>` — Autonomous rounds until done (or on a timer); Esc stops

**Team & Planning**
- `/team <prompt>` — Spawn a background worker
- `/team status` — Show worker status
- `/team stop [id]` — Stop a worker (or all)
- `/team msg <name> <msg>` — Send a message to a worker
- `/tasks` — Show the shared task list
- `/messages` — Check messages from workers
- `/plan <prompt>` — Create plan.md / `/plan show` / `/plan go`
- `/projectplan <prompt>` — RAG-researched plan / `/projectplan show` / `/projectplan go`

**Project**
- `/publish [name]` — Create GitHub repo and push
- `/new <name> [desc]` — Scaffold a new project
- `/init` — Generate a CLAUDE.md for this project
- `/run [command]` — Run the project (auto-detect)
- `/git sync|pr|log|stash` — Smart git commands
- `/memory` — View/add memory / `/memory edit`
- `/image <path> [prompt]` — Send an image

**Knowledge Base**
- `/docs <query>` — Search indexed docs (Swift, SwiftUI, HIG, App Store Guidelines, CNN)
- `/index` — Index this project into the RAG service for code_search (semantic "where/how is X handled" search)

**Extensibility**
- `/teach <name> <desc> -- <cmd>` — Create a custom tool
- `/snippet save <name> <prompt>` — Save a reusable prompt
- `/snippet <name>` — Run a saved snippet
- `/quit` or `/exit` — Exit

## Skills"""
        for skill in skills.all():
            help_text += f"\n- `/{skill.name}` — {skill.description}"
        console.print(Markdown(help_text))
        return None

    elif command == "/clear":
        context.clear()
        os.system("clear" if os.name != "nt" else "cls")
        print_banner(console, config,
                     skill_count=len(skills.all()),
                     instruction_sources=load_instructions(os.getcwd()).sources,
                     project_type=detect_project_type(os.getcwd()),
                     real_model_name=getattr(model, "real_model_name", "") or None)
        return None

    elif command == "/compact":
        # Compaction needs a non-streamed model call for the summary, which the
        # async REPL runs (this handler is sync). Defer via a sentinel; any
        # `/compact <instructions>` are folded into the summary request.
        return f"__COMPACT__{args.strip()}"

    elif command == "/config":
        import yaml
        if not args:
            console.print(Markdown(f"```yaml\n{yaml.dump(_redacted_config(config), default_flow_style=False)}```"))
            return None

        sub_parts = args.strip().split(maxsplit=2)
        sub_cmd = sub_parts[0].lower()

        if sub_cmd == "set":
            if len(sub_parts) < 3:
                console.print("[#ebcb8b]Usage: /config set <key> <value>[/#ebcb8b]")
                console.print("[#8899aa]  /config set model.temperature 0.5[/#8899aa]")
                console.print("[#8899aa]  /config set permissions.mode trust[/#8899aa]")
                console.print("[#8899aa]  /config set ui.notification_sound true[/#8899aa]")
                return None
            key_path = sub_parts[1]
            value = sub_parts[2]
            if key_path == "permissions.mode":
                # Accept Claude Code's mode names here too, normalized to
                # the native name BEFORE it's persisted/applied so the
                # saved config and the live permissions object never
                # disagree about which name is canonical.
                resolved_mode = resolve_mode_name(value.strip())
                if resolved_mode is not None:
                    value = resolved_mode
                if value == "plan":
                    # "plan" is a per-session STATE, not a runnable live mode:
                    # permissions.mode="plan" silently behaves like ask, and a
                    # persisted "plan" would recreate that on next launch (startup
                    # always begins with plan_state inactive). Mirror /mode plan
                    # for this session and persist the VALID mode it maps to
                    # (auto) so the saved config never holds an unrunnable value.
                    if permissions is not None:
                        permissions.mode = "auto"
                    if plan_state is not None:
                        plan_state.active = True
                    ok, msg = set_config(config, key_path, "auto")
                    if ok:
                        console.print("[#a3be8c]Entered plan mode for this session (reads free, writes prompt).[/#a3be8c]")
                        console.print("[#8899aa]Note: plan is a per-session state — saved permissions.mode = auto. Use /mode plan to re-enter it next session.[/#8899aa]")
                    else:
                        console.print(f"[#bf616a]{msg}[/#bf616a]")
                    return None
            ok, msg = set_config(config, key_path, value)
            if ok:
                # Apply to the live objects so the change takes effect this
                # session (set_config only mutates the config dict/file).
                if key_path == "permissions.mode" and permissions is not None:
                    permissions.mode = value.strip()
                    if plan_state is not None:
                        plan_state.active = False
                elif key_path == "model.temperature":
                    try:
                        model.temperature = float(value)
                    except (TypeError, ValueError):
                        pass
                console.print(f"[#a3be8c]{msg}[/#a3be8c]")
            else:
                console.print(f"[#bf616a]{msg}[/#bf616a]")
            return None

        elif sub_cmd == "reset":
            from .config import DEFAULT_CONFIG
            for key, val in DEFAULT_CONFIG.items():
                config[key] = val
            console.print("[#a3be8c]Config reset to defaults (in-memory only)[/#a3be8c]")
            return None

        else:
            console.print(Markdown(f"```yaml\n{yaml.dump(_redacted_config(config), default_flow_style=False)}```"))
            return None

    elif command == "/model":
        if not args:
            # Show current model info. Use the name resolved once at startup
            # (avoids a blocking 2s HTTP call on the event loop every /model).
            config_name = get(config, "model", "name")
            display_name = get(config, "model", "display_name", default="")
            real_name = getattr(model, "real_model_name", "") or None
            if display_name:
                console.print(f"Model: {display_name}")
                console.print(f"Wire id: {config_name} [#4c566a](sent to the server)[/#4c566a]")
            elif real_name and real_name != config_name:
                console.print(f"Model: {real_name}")
                console.print(f"Served as: {config_name} [#4c566a](alias)[/#4c566a]")
            else:
                console.print(f"Model: {config_name}")
            console.print(f"Provider: {get(config, 'model', 'provider', default='unknown')}")
            console.print(f"Endpoint: {get(config, 'model', 'endpoint')}")
            console.print(f"Temperature: {get(config, 'model', 'temperature')}")
            console.print(f"Input tokens: {model.total_input_tokens:,}")
            console.print(f"Output tokens: {model.total_output_tokens:,}")
            return None

        sub = args.strip().lower()

        if sub == "list":
            # Show available providers from config
            providers = config.get("providers", {})
            current = get(config, "model", "provider", default="")
            if not providers:
                console.print("[#8899aa]No providers configured in config.yaml[/#8899aa]")
                console.print("[#8899aa]Built-in providers: ollama, gemini, openai[/#8899aa]")
                return None
            console.print("[bold #eceff4]Available providers:[/bold #eceff4]")
            for name, pconf in providers.items():
                marker = " [#a3be8c]← active[/#a3be8c]" if name == current else ""
                model_name = pconf.get("model", "?")
                console.print(f"  [#88c0d0]{name}[/#88c0d0]  {model_name}{marker}")
            console.print("[#8899aa]Switch with: /model <provider>[/#8899aa]")
            return None

        # Switch provider — return signal for main loop
        return f"__MODEL_SWITCH__{sub}"

    elif command == "/providers":
        table = Table(
            title="API Providers",
            border_style="#4c566a",
            show_header=True,
            header_style="bold #88c0d0",
        )
        table.add_column("Provider", style="#88c0d0")
        table.add_column("Default Model", style="#d8dee9")
        table.add_column("API Key", style="#d8dee9")
        table.add_column("Signup URL", style="#5e81ac")
        for p in _PROVIDER_INFO:
            if p["env_var"]:
                key_status = (
                    "[#a3be8c]SET[/#a3be8c]"
                    if os.environ.get(p["env_var"])
                    else "[#bf616a]NOT SET[/#bf616a]"
                )
                key_col = f"{p['env_var']}  {key_status}"
            else:
                key_col = "[#8899aa]none needed[/#8899aa]"
            table.add_row(p["label"], p["model"], key_col, p["signup"])
        console.print(table)
        console.print()
        console.print("[bold #eceff4]To set a key:[/bold #eceff4]")
        console.print("  [#88c0d0]1.[/#88c0d0] Run [bold]/setkey <provider>[/bold] (guided, masked entry — this session)")
        console.print("  [#88c0d0]2.[/#88c0d0] Run [bold]spark --setup[/bold] (interactive wizard)")
        console.print("  [#88c0d0]3.[/#88c0d0] Or add to your shell profile (~/.zshrc):")
        console.print('     [#a3be8c]export GEMINI_API_KEY="your-key-here"[/#a3be8c]')
        console.print("     then restart your terminal or run [#a3be8c]source ~/.zshrc[/#a3be8c]")
        return None

    elif command == "/setkey":
        provider_name = args.strip().lower()

        if not provider_name:
            keys_on_disk = load_keys()
            console.print("[bold #eceff4]Cloud provider presets:[/bold #eceff4]")
            for name, preset in PROVIDER_PRESETS.items():
                have = "  [#a3be8c]key set[/#a3be8c]" if name in keys_on_disk else ""
                console.print(
                    f"  [#88c0d0]{name}[/#88c0d0]  {preset['model']}  "
                    f"[#8899aa]({preset['key_env']})[/#8899aa]{have}"
                )
            console.print()
            console.print("[#8899aa]Usage: /setkey <provider>   e.g. /setkey openrouter[/#8899aa]")
            return None

        if provider_name not in PROVIDER_PRESETS:
            console.print(f"[#bf616a]Unknown provider: {provider_name}[/#bf616a]")
            console.print(f"[#8899aa]Available: {', '.join(PROVIDER_PRESETS.keys())}[/#8899aa]")
            return None

        preset = PROVIDER_PRESETS[provider_name]
        console.print(f"  Get a key at: [#5e81ac]{preset['doc_url']}[/#5e81ac]")

        # Suspend any active Esc watcher for the duration of the prompt — same
        # reason as permissions._prompt_user: without this, the watcher's
        # background thread holds stdin in cbreak and drains the same fd the
        # masked Prompt.ask needs to read from.
        from spark_code.ui.esc_watcher import pause_all
        try:
            with pause_all():
                entered = Prompt.ask(f"  Paste your {preset['key_env']}", password=True)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[#8899aa]Cancelled — nothing saved.[/#8899aa]")
            return None

        value = (entered or "").strip()
        if not value:
            console.print("[#ebcb8b]  No key entered — nothing saved.[/#ebcb8b]")
            return None

        try:
            path = save_key(provider_name, value)
        except OSError as e:
            console.print(f"[#bf616a]  Could not save key: {e}[/#bf616a]")
            return None

        console.print(f"[#a3be8c]  Saved {mask(value)} to {path} (chmod 600)[/#a3be8c]")

        # New provider → seed a config block from the preset (endpoint + model
        # only; the key itself lives in the keys file, never in config.yaml —
        # resolve_provider_key picks it up from there at model-build time).
        # An EXISTING provider block (hers, or a prior /setkey) is left alone.
        providers = config.setdefault("providers", {})
        if provider_name not in providers:
            ok_endpoint, msg_endpoint = set_config(
                config, f"providers.{provider_name}.endpoint", preset["endpoint"], scope="global")
            ok_model, msg_model = set_config(
                config, f"providers.{provider_name}.model", preset["model"], scope="global")
            if ok_endpoint and ok_model:
                console.print(
                    f"[#a3be8c]  Added '{provider_name}' provider "
                    f"({preset['model']} @ {preset['endpoint']}) to global config[/#a3be8c]"
                )
            else:
                console.print(
                    f"[#ebcb8b]  Key saved, but couldn't write the provider block "
                    f"({msg_endpoint or msg_model}) — add it manually to "
                    f"~/.spark/config.yaml under 'providers:'[/#ebcb8b]"
                )

        console.print(f"[#8899aa]  Use it: /model {provider_name}[/#8899aa]")
        return None

    elif command == "/tokens":
        tokens = context.estimate_tokens()
        max_tokens = context.max_tokens
        pct = tokens / max_tokens * 100 if max_tokens else 0
        console.print(f"Context: ~{tokens:,} / {max_tokens:,} tokens ({pct:.0f}%)")
        console.print(f"Turns: {context.turn_count}")
        console.print(f"API usage: {model.total_input_tokens:,} in / {model.total_output_tokens:,} out")
        return None

    elif command in ("/stats", "/status"):
        if not stats:
            console.print("[#8899aa]No stats available[/#8899aa]")
            return None
        table = Table(title="Session Statistics", border_style="#4c566a",
                      show_header=True, header_style="bold #88c0d0")
        table.add_column("Metric", style="#d8dee9")
        table.add_column("Value", style="#eceff4")

        table.add_row("Duration", stats.format_duration())
        table.add_row("Turns", str(context.turn_count))

        # Generation speed
        speed_str = stats.format_speed()
        if speed_str:
            table.add_row("Generation speed", speed_str)

        # Token counts
        if stats.input_tokens > 0 or stats.output_tokens > 0:
            table.add_row("Tokens in / out",
                           f"{stats.input_tokens:,} / {stats.output_tokens:,}")
        elif hasattr(model, 'total_input_tokens'):
            table.add_row("Tokens in / out",
                           f"{model.total_input_tokens:,} / {model.total_output_tokens:,}")

        # Cost
        cost_str = stats.format_cost()
        if cost_str:
            provider_name = get(config, "model", "provider", default="")
            table.add_row("Session cost", f"{cost_str} ({provider_name})")
        else:
            provider_name = get(config, "model", "provider", default="local")
            table.add_row("Session cost", f"$0.00 ({provider_name})")

        # Tool breakdown
        table.add_row("Total tool calls", str(stats.total_tool_calls))
        if stats.tool_calls:
            for tool_name, count in sorted(stats.tool_calls.items(),
                                            key=lambda x: -x[1]):
                table.add_row(f"  {tool_name}", str(count))

        # Files
        created = len(stats.files_created) if hasattr(stats, 'files_created') else 0
        table.add_row("Files",
                       f"{created} created, "
                       f"{len(stats.files_read)} read, "
                       f"{len(stats.files_edited)} edited")
        table.add_row("Commands run", str(stats.commands_run))

        # Workers
        if team_manager:
            workers = team_manager.workers
            total = len(workers)
            if total > 0:
                completed = sum(1 for w in workers.values() if w.status == "completed")
                failed = sum(1 for w in workers.values() if w.status == "failed")
                worker_str = f"{total} spawned, {completed} completed"
                if failed:
                    worker_str += f", {failed} failed"
                table.add_row("Workers", worker_str)

        if pinned and pinned.count > 0:
            table.add_row("Pinned files", str(pinned.count))
        console.print(table)
        return None

    elif command == "/doctor":
        # run_doctor is a coroutine and this handler is sync (see the
        # __DOCTOR__ branch in the async REPL loop) — deferred via a
        # sentinel, same pattern as /index and /compact.
        return "__DOCTOR__"

    elif command == "/hooks":
        # Read-only listing (Task 6) — reconstructs a HookManager straight
        # from config so this always reflects what's actually configured,
        # never a live agent's possibly-stale instance. No hook runs.
        hm = HookManager(config)
        if hm.count == 0:
            console.print(
                "[#8899aa]No hooks configured. See docs/hooks.md for "
                "recipes and the `hooks:` config shape.[/#8899aa]"
            )
            return None
        table = Table(
            title="Configured Hooks",
            border_style="#4c566a",
            show_header=True,
            header_style="bold #88c0d0",
        )
        table.add_column("Event", style="#88c0d0")
        table.add_column("Pattern", style="#d8dee9")
        table.add_column("Command", style="#d8dee9")
        table.add_column("Timeout", style="#8899aa", justify="right")
        for event in hm.get_events():
            for hook in hm.get_hooks(event):
                # Wrap pattern/command in Text() so a config-supplied value
                # containing Rich markup (e.g. `echo [bold]x[/bold]`) renders
                # VERBATIM — /hooks is an audit of a possibly-untrusted
                # .spark/config.yaml, so its own display must not let a hook
                # command consume/alter its markup. (event is a fixed key.)
                table.add_row(
                    event,
                    Text(hook.pattern, style="#d8dee9"),
                    Text(hook.command, style="#d8dee9"),
                    f"{hook.timeout}s",
                )
        console.print(table)
        return None

    elif command == "/diff":
        # Run git diff and show with syntax highlighting
        diff_args = args.strip() if args else ""
        git_cmd = ["git", "diff"] + (diff_args.split() if diff_args else [])
        try:
            result = subprocess.run(
                git_cmd, capture_output=True, text=True, timeout=10,
            )
            diff_output = result.stdout
            if not diff_output:
                console.print("[#8899aa]No changes to show.[/#8899aa]")
                return None
            syntax = Syntax(diff_output, "diff", theme="nord-darker",
                            line_numbers=False, background_color="#2e3440")
            console.print(Panel(syntax, title="[bold #88c0d0] git diff [/bold #88c0d0]",
                                border_style="#4c566a", box=ROUNDED, padding=(0, 1)))
        except subprocess.TimeoutExpired:
            console.print("[#bf616a]git diff timed out[/#bf616a]")
        except FileNotFoundError:
            console.print("[#bf616a]git not found[/#bf616a]")
        return None

    elif command == "/memory":
        if memory is None:
            console.print("[#bf616a]Memory system not available[/#bf616a]")
            return None

        if not args:
            # Show global + project memory
            global_mem = memory.load_global()
            project_mem = memory.load_project()
            if global_mem:
                console.print(Panel(
                    Markdown(global_mem),
                    title="[bold #88c0d0] Global Memory [/bold #88c0d0]",
                    border_style="#4c566a", box=ROUNDED, padding=(1, 2),
                ))
            if project_mem:
                console.print(Panel(
                    Markdown(project_mem),
                    title="[bold #88c0d0] Project Memory [/bold #88c0d0]",
                    border_style="#4c566a", box=ROUNDED, padding=(1, 2),
                ))
            if not global_mem and not project_mem:
                console.print("[#8899aa]No memory entries yet. Use /memory add <entry>[/#8899aa]")
            return None

        sub_parts = args.strip().split(maxsplit=1)
        sub_cmd = sub_parts[0].lower()

        if sub_cmd == "add":
            entry = sub_parts[1] if len(sub_parts) > 1 else ""
            if not entry:
                console.print("[#ebcb8b]Usage: /memory add <entry>[/#ebcb8b]")
                return None
            memory.append_project(entry)
            console.print("[#a3be8c]Added to project memory[/#a3be8c]")
            return None

        elif sub_cmd == "edit":
            return (
                "Read the project memory file at .spark/memory/MEMORY.md "
                "(create it if it doesn't exist). Show its contents to the user "
                "and ask what they'd like to change. Then edit it accordingly."
            )

        else:
            # Treat as add
            memory.append_project(args.strip())
            console.print("[#a3be8c]Added to project memory[/#a3be8c]")
            return None

    elif command == "/yolo":
        # Toggle agent mode — swap system prompt and set trust
        is_agentic = context.system_prompt.startswith("You are Spark Code, a fully autonomous")
        if is_agentic:
            # Switch back to normal mode
            new_prompt = context.system_prompt.replace(AGENTIC_PROMPT, SYSTEM_PROMPT)
            context.system_prompt = new_prompt
            if permissions:
                permissions.mode = "auto"
            console.print("[#8899aa]Agent mode off — back to normal[/#8899aa]")
        else:
            # Switch to agent mode
            new_prompt = context.system_prompt.replace(SYSTEM_PROMPT, AGENTIC_PROMPT)
            context.system_prompt = new_prompt
            if permissions:
                permissions.mode = "trust"
            console.print("[bold #ebcb8b]⚡ Agent mode on[/bold #ebcb8b] [#8899aa]— autonomous execution, all tools trusted[/#8899aa]")
        return None

    elif command == "/image":
        if not args:
            console.print("[#ebcb8b]Usage: /image <file_path> \\[prompt][/#ebcb8b]")
            console.print("[#8899aa]Example: /image ~/Desktop/screenshot.png what's wrong with this UI?[/#8899aa]")
            return None
        # Parse: first token is path, rest is prompt. Use shlex so dragged-in
        # paths with escaped/quoted spaces (macOS) stay intact.
        import shlex
        try:
            tokens = shlex.split(args)
        except ValueError:
            tokens = args.split()
        img_path = os.path.expanduser(tokens[0]) if tokens else ""
        img_prompt = " ".join(tokens[1:]) if len(tokens) > 1 else "Describe this image."
        if not os.path.exists(img_path):
            # Fallback: a raw (unescaped) path with spaces — grow the prefix
            # until an existing file is found, treating the remainder as prompt.
            found = False
            raw = args.strip()
            for i in range(len(tokens), 0, -1):
                cand = os.path.expanduser(" ".join(tokens[:i]))
                if os.path.exists(cand):
                    img_path = cand
                    img_prompt = " ".join(tokens[i:]) or "Describe this image."
                    found = True
                    break
            if not found:
                console.print(f"[#bf616a]File not found: {_esc(img_path or raw)}[/#bf616a]")
                return None
        # Read and encode image. os.path.exists() above is True for a
        # directory too (e.g. `/image ~/Desktop`), so guard the open with an
        # explicit isfile check + OSError catch — mirrors the drag-drop path —
        # instead of letting IsADirectoryError escape as a raw traceback.
        if not os.path.isfile(img_path):
            console.print(f"[#bf616a]Not a file: {_esc(img_path)}[/#bf616a]")
            return None
        mime_type = mimetypes.guess_type(img_path)[0] or "image/png"
        try:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
        except OSError as e:
            console.print(f"[#bf616a]Could not read image: {_esc(str(e))}[/#bf616a]")
            return None
        size_kb = len(img_data) * 3 / 4 / 1024
        console.print(f"  [#88c0d0]Image[/#88c0d0] [#d8dee9]{os.path.basename(img_path)}[/#d8dee9] [#4c566a]({size_kb:.0f} KB, {mime_type})[/#4c566a]")
        # Snapshot BEFORE the turn's user message enters context (Phase 5
        # Task 7) — agent.run_without_user_add() (called by the caller once
        # it sees __IMAGE_SENT__) intentionally does not snapshot itself
        # since the message is already added by the time it runs.
        context.snapshot()
        # Store image in context and return prompt for agent
        context.add_user_with_image(img_prompt, img_data, mime_type)
        return "__IMAGE_SENT__"  # Signal to skip add_user in agent

    elif command in ("/mode", "/trust", "/auto", "/ask"):
        if permissions is None:
            console.print("[#bf616a]Permissions not available[/#bf616a]")
            return None
        valid_modes = {"ask", "auto", "trust"}
        if command == "/mode":
            target_raw = args.strip() if args else ""
            # Resolve Claude Code's mode names too (default/acceptEdits/
            # bypassPermissions/plan), case-insensitively.
            target = resolve_mode_name(target_raw)
            if target == "plan":
                # Enter plan mode: reads run free, writes are blocked by the
                # plan gate (drop to auto so reads never prompt either).
                if plan_state is not None:
                    plan_state.active = True
                permissions.mode = "auto"
                console.print("[#a3be8c]Switched to plan mode[/#a3be8c]")
                return None
            if target in valid_modes:
                new_mode = target
            else:
                display = "plan" if (plan_state and plan_state.active) else permissions.mode
                console.print(f"[#88c0d0]Current mode:[/#88c0d0] [#d8dee9]{display}[/#d8dee9]")
                console.print("[#8899aa]Usage: /mode <ask|auto|trust|plan>  or  shift+tab to cycle[/#8899aa]")
                console.print("[#8899aa]  ask   — confirm every tool call  (Claude Code: default)[/#8899aa]")
                console.print("[#8899aa]  auto  — allow reads, ask for writes  (Claude Code: acceptEdits)[/#8899aa]")
                console.print("[#8899aa]  trust — allow all tool calls  (Claude Code: bypassPermissions)[/#8899aa]")
                console.print("[#8899aa]  plan  — plan before executing (reads free, writes prompt)[/#8899aa]")
                return None
        else:
            new_mode = command[1:]  # /trust -> trust, /auto -> auto, /ask -> ask
        # Any explicit mode switch exits plan mode.
        if plan_state is not None:
            plan_state.active = False
        permissions.mode = new_mode
        config["permissions"]["mode"] = new_mode
        if team_manager is not None:
            team_manager.worker_permission_mode = new_mode
        console.print(f"[#a3be8c]Switched to {new_mode} mode[/#a3be8c]")
        return None

    elif command in ("/quit", "/exit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        sys.exit(0)

    elif command == "/team":
        if not team_manager:
            console.print("[#bf616a]Team system not available[/#bf616a]")
            return None

        if not args:
            console.print("[#ebcb8b]Usage: /team <prompt> — spawn a worker[/#ebcb8b]")
            console.print("[#8899aa]  /team status     — show all workers[/#8899aa]")
            console.print("[#8899aa]  /team stop \\[id]  — stop a worker[/#8899aa]")
            console.print("[#8899aa]  /team msg <name> <message> — message a worker[/#8899aa]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        if sub_cmd == "status":
            workers = team_manager.status()
            if not workers:
                console.print("[#8899aa]No workers spawned yet.[/#8899aa]")
                return None
            for w in workers:
                status_icon = {
                    "running": "[#ebcb8b]⟳ running[/#ebcb8b]",
                    "completed": "[#a3be8c]✓ completed[/#a3be8c]",
                    "failed": "[#bf616a]✗ failed[/#bf616a]",
                }.get(w["status"], w["status"])
                console.print(
                    f"  [#88c0d0]#{w['id']}[/#88c0d0] "
                    f"[#d8dee9]{_esc(str(w['name']))}[/#d8dee9]  "
                    f"{status_icon}  "
                    f"[#8899aa]{_esc(str(w['prompt'])[:60])}[/#8899aa]"
                )
            return None

        elif sub_cmd == "stop":
            stop_id = sub[1] if len(sub) > 1 else ""
            return f"__TEAM_STOP__{stop_id}"

        elif sub_cmd == "msg":
            # /team msg worker-1 Hey, how's it going?
            msg_rest = sub[1] if len(sub) > 1 else ""
            msg_parts = msg_rest.split(maxsplit=1)
            if len(msg_parts) < 2:
                console.print("[#ebcb8b]Usage: /team msg <worker-name> <message>[/#ebcb8b]")
                return None
            target_name = msg_parts[0]
            msg_content = msg_parts[1]
            result = team_manager.deliver_message("lead", target_name, msg_content)
            console.print(f"[#a3be8c]{result}[/#a3be8c]")
            return None

        else:
            # Spawn a worker — the full args is the prompt
            return f"__TEAM_SPAWN__{args}"

    elif command == "/messages":
        if not team_manager:
            console.print("[#bf616a]Team system not available[/#bf616a]")
            return None
        msgs = team_manager.get_lead_messages()
        if not msgs:
            console.print("[#8899aa]No new messages.[/#8899aa]")
            return None
        for m in msgs:
            console.print(
                f"  [#5e81ac]\\[{_esc(str(m.from_name))}][/#5e81ac] "
                f"[#d8dee9]{_esc(str(m.content))}[/#d8dee9]"
            )
        return None

    elif command == "/tasks":
        if not task_store:
            console.print("[#bf616a]Task store not available[/#bf616a]")
            return None
        tasks = task_store.list()
        if not tasks:
            console.print("[#8899aa]No tasks yet. Spawn a worker with /team <prompt>[/#8899aa]")
            return None
        for t in tasks:
            status_icon = {
                "pending": "[#ebcb8b]○[/#ebcb8b]",
                "in_progress": "[#ebcb8b]⟳[/#ebcb8b]",
                "completed": "[#a3be8c]✓[/#a3be8c]",
                "failed": "[#bf616a]✗[/#bf616a]",
            }.get(t.status, " ")
            assigned = f"  [#4c566a]({t.assigned_to})[/#4c566a]" if t.assigned_to else ""
            console.print(
                f"  [#88c0d0]#{t.id}[/#88c0d0]  "
                f"{status_icon} {t.status:<12}  "
                f"[#d8dee9]{_esc(str(t.description)[:60])}[/#d8dee9]"
                f"{assigned}"
            )
        return None

    elif command == "/plan":
        if not args:
            console.print("[#ebcb8b]Usage: /plan <prompt> — create a plan[/#ebcb8b]")
            console.print("[#8899aa]  /plan show    — show current plan[/#8899aa]")
            console.print("[#8899aa]  /plan copy    — copy plan to clipboard[/#8899aa]")
            console.print("[#8899aa]  /plan go      — execute the approved plan[/#8899aa]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        plan_path = os.path.join(os.getcwd(), "plan.md")

        if sub_cmd == "show":
            if not os.path.exists(plan_path):
                console.print("[#8899aa]No plan.md found. Create one with /plan <prompt>[/#8899aa]")
                return None
            with open(plan_path) as f:
                content = f.read()
            console.print()
            console.print(Panel(
                Markdown(content),
                title="[bold #88c0d0] plan.md [/bold #88c0d0]",
                border_style="#4c566a",
                box=ROUNDED,
                padding=(1, 2),
            ))
            _copy_to_clipboard(content)
            console.print()
            console.print("[#a3be8c]  ✓ Copied to clipboard  ·  /plan go to execute  ·  /plan <prompt> to redo[/#a3be8c]")
            return None

        elif sub_cmd == "copy":
            if not os.path.exists(plan_path):
                console.print("[#8899aa]No plan.md found. Create one with /plan <prompt>[/#8899aa]")
                return None
            with open(plan_path) as f:
                content = f.read()
            if _copy_to_clipboard(content):
                console.print("[#a3be8c]✓ Plan copied to clipboard[/#a3be8c]")
            else:
                console.print("[#bf616a]Could not copy — pbcopy not available[/#bf616a]")
            return None

        elif sub_cmd == "go":
            if not os.path.exists(plan_path):
                console.print("[#bf616a]No plan.md found. Create one first with /plan <prompt>[/#bf616a]")
                return None
            with open(plan_path) as f:
                plan_content = f.read()
            # System-driven execution — return signal for main loop
            return f"__PLAN_EXECUTE__{plan_content}"

        else:
            # /plan <prompt> — create a plan
            plan_prompt = args
            create_plan_prompt = (
                f"The user wants you to create a plan before executing. "
                f"Their request: {plan_prompt}\n\n"
                "RULES:\n"
                "1. Use read-only tools to explore the codebase first (read_file, glob, grep, list_dir)\n"
                "2. Do NOT create project files yet — only create plan.md\n"
                "3. Create a detailed implementation plan\n"
                "4. You MUST use the write_file tool to save the plan to 'plan.md' in the current directory\n"
                "   This is the ONE file you must write. Do not skip this step.\n\n"
                "The plan.md should include:\n"
                "- Summary of what will be done\n"
                "- Numbered steps with clear descriptions\n"
                "- Files that will be modified or created\n"
                "- A '## Parallelization' section. On its own line, list the step\n"
                "  numbers that are INDEPENDENT and safe to run in parallel using\n"
                "  the exact marker format: 'Parallel: 1, 2' (only truly\n"
                "  independent steps; omit steps that depend on earlier ones).\n"
                "- Any risks or considerations\n\n"
                "After writing plan.md, tell the user: Review the plan with /plan show, then /plan go to execute."
            )
            return create_plan_prompt

    elif command == "/projectplan":
        if not args:
            console.print("[#ebcb8b]Usage: /projectplan <prompt> — create a RAG-researched plan[/#ebcb8b]")
            console.print("[#8899aa]  /projectplan show    — show current plan[/#8899aa]")
            console.print("[#8899aa]  /projectplan copy    — copy plan to clipboard[/#8899aa]")
            console.print("[#8899aa]  /projectplan go      — execute the approved plan[/#8899aa]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        pp_path = os.path.join(os.getcwd(), "projectplan.md")

        if sub_cmd == "show":
            if not os.path.exists(pp_path):
                console.print("[#8899aa]No projectplan.md found. Create one with /projectplan <prompt>[/#8899aa]")
                return None
            with open(pp_path) as f:
                content = f.read()
            console.print()
            console.print(Panel(
                Markdown(content),
                title="[bold #88c0d0] projectplan.md [/bold #88c0d0]",
                border_style="#4c566a",
                box=ROUNDED,
                padding=(1, 2),
            ))
            _copy_to_clipboard(content)
            console.print()
            console.print("[#a3be8c]  ✓ Copied to clipboard  ·  /projectplan go to execute  ·  /projectplan <prompt> to redo[/#a3be8c]")
            return None

        elif sub_cmd == "copy":
            if not os.path.exists(pp_path):
                console.print("[#8899aa]No projectplan.md found. Create one with /projectplan <prompt>[/#8899aa]")
                return None
            with open(pp_path) as f:
                content = f.read()
            if _copy_to_clipboard(content):
                console.print("[#a3be8c]✓ Project plan copied to clipboard[/#a3be8c]")
            else:
                console.print("[#bf616a]Could not copy — pbcopy not available[/#bf616a]")
            return None

        elif sub_cmd == "go":
            if not os.path.exists(pp_path):
                console.print("[#bf616a]No projectplan.md found. Create one first with /projectplan <prompt>[/#bf616a]")
                return None
            with open(pp_path) as f:
                plan_content = f.read()
            return f"__PLAN_EXECUTE__{plan_content}"

        else:
            # /projectplan <prompt> — research RAG, then create plan
            plan_prompt = args
            project_type = detect_project_type(os.getcwd())

            # Extract keywords and fetch RAG context
            keywords = extract_keywords(plan_prompt)
            console.print(f"[#88c0d0]▸ Researching docs for: {', '.join(keywords) or plan_prompt}[/#88c0d0]")

            # fetch_rag_context is async; this handler is sync and runs inside
            # the REPL's event loop, so drive the coroutine on a worker thread.
            _rag = fetch_rag_context(keywords, project_type, prompt=plan_prompt)
            if asyncio.iscoroutine(_rag):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                    rag_context = _ex.submit(asyncio.run, _rag).result()
            else:
                rag_context = _rag

            if rag_context:
                ref_count = rag_context.count("[Ref ")
                console.print(f"[#a3be8c]  ✓ Found {ref_count} reference(s) from knowledge base[/#a3be8c]")
            else:
                console.print("[#ebcb8b]  ⚠ No RAG results (service down or no matches)[/#ebcb8b]")
            # Discover MCP tools
            mcp_section = ""
            mcp_configs = find_mcp_configs()
            if mcp_configs:
                mcp_lines = ["## Available MCP Tools\n",
                             "You MUST incorporate these tools into the plan steps where applicable. "
                             "For each tool, add a step or sub-step that uses it.\n"]
                for server_name, server_conf in mcp_configs.items():
                    desc = server_conf.get("description", "")
                    tools_list = server_conf.get("tools", [])
                    mcp_lines.append(f"**{server_name}**" + (f" — {desc}" if desc else ""))
                    if tools_list:
                        for t in tools_list:
                            if isinstance(t, dict):
                                mcp_lines.append(f"  - `{t.get('name', '?')}`: {t.get('description', '')}")
                            else:
                                mcp_lines.append(f"  - `{t}`")
                    mcp_lines.append("")
                mcp_section = "\n".join(mcp_lines) + "\n"
                console.print(f"[#a3be8c]  ✓ Found {len(mcp_configs)} MCP server(s)[/#a3be8c]")

            # Detect if directory is empty (new project)
            is_empty_dir = not any(
                f for f in os.listdir(os.getcwd())
                if not f.startswith(".") and f not in ("projectplan.md", "plan.md")
            )

            console.print(f"[#88c0d0]▸ {'Writing' if is_empty_dir else 'Exploring codebase and writing'} projectplan.md...[/#88c0d0]")
            console.print("[#4c566a]  (This may take a minute with large models. Ctrl+C to cancel)[/#4c566a]")

            rag_section = ""
            if rag_context:
                rag_section = (
                    "Pre-researched documentation — include as '## Reference Material' in the plan. "
                    "Tag steps with [see Ref N].\n\n"
                    f"{rag_context}\n\n---\n\n"
                )

            # Build MCP instruction if tools are available
            mcp_instruction = ""
            if mcp_section:
                mcp_instruction = (
                    "## MCP Tools section is REQUIRED. Add a dedicated step for each "
                    "MCP tool that applies to this project. For example, if an image generation "
                    "tool is available and the project involves images/photos, add a step that "
                    "uses it.\n\n"
                )

            if is_empty_dir:
                # Lean prompt for new projects — no codebase to explore
                create_plan_prompt = (
                    f"Create projectplan.md for a NEW project.\n"
                    f"Request: {plan_prompt}\n\n"
                    f"{rag_section}"
                    f"{mcp_section}"
                    f"{mcp_instruction}"
                    "This is an EMPTY directory — do NOT explore files. "
                    "Write projectplan.md IMMEDIATELY using write_file.\n\n"
                    "Format: ## Reference Material (if docs above), ---, "
                    "## Summary, ## Steps (numbered, tag with [see Ref N]), "
                    "## Parallelization (list independent step numbers as 'Parallel: 1, 2'), "
                    "## Files, ## Risks\n\n"
                    "Keep the plan concise — max 8 steps. Write the file NOW."
                )
            else:
                # Full prompt for existing projects
                create_plan_prompt = (
                    f"Create projectplan.md for this project.\n"
                    f"Request: {plan_prompt}\n"
                    f"Project type: {project_type or 'unknown'}\n\n"
                    f"{rag_section}"
                    f"{mcp_section}"
                    f"{mcp_instruction}"
                    "RULES:\n"
                    "1. Explore the codebase first (glob, grep, read_file)\n"
                    "2. Only create projectplan.md — no other files\n"
                    "3. Use write_file to save projectplan.md\n\n"
                    "Format: ## Reference Material (if docs above), ---, "
                    "## Summary, ## Steps (numbered, tag with [see Ref N]), "
                    "## Parallelization (list independent step numbers as 'Parallel: 1, 2'), "
                    "## Files, ## Risks\n\n"
                    "Do NOT use rag_search — docs are pre-researched.\n"
                    "After writing, say: Review with /projectplan show, then /projectplan go to execute."
                )
            return create_plan_prompt

    elif command == "/publish":
        # Create a GitHub repo and push the project
        raw_args = args.strip() if args else ""

        # Parse --private flag first
        visibility = "--public"
        if "--private" in raw_args:
            visibility = "--private"
            raw_args = raw_args.replace("--private", "").strip()

        # Auto-detect repo name from current directory if not provided
        repo_name = raw_args
        if not repo_name:
            import re
            repo_name = os.path.basename(os.getcwd())
            # Sanitize: lowercase, replace spaces/underscores with hyphens, strip non-alnum
            repo_name = repo_name.lower().replace(" ", "-").replace("_", "-")
            repo_name = re.sub(r'[^a-z0-9\-]', '', repo_name)
            repo_name = re.sub(r'-+', '-', repo_name).strip('-')
            console.print(f"[#88c0d0]Publishing as [bold]\"{repo_name}\"[/bold] on GitHub ({visibility.lstrip('-')})[/#88c0d0]")

        return (
            f"Create a GitHub repository called \"{repo_name}\" and push the current project. Steps:\n\n"
            f"1. Read the source files in this directory to understand the project\n"
            f"2. If no README.md exists, use write_file to create a short README.md with:\n"
            f"   - Project name and one-line description\n"
            f"   - How to install and run it\n"
            f"   - Keep it concise (under 30 lines)\n"
            f"3. If no .gitignore exists, use write_file to create one appropriate for the languages used\n"
            f"   (e.g. Python: __pycache__/, .venv/, .env; Node: node_modules/, dist/; etc.)\n"
            f"4. Check if this directory already has a git repo. If not, run `git init`\n"
            f"5. Stage all project files with `git add .`\n"
            f"6. Commit with a descriptive message based on what the project does\n"
            f"7. Create the GitHub repo: `gh repo create {repo_name} {visibility} --source=. --push`\n"
            f"8. Show the repo URL when done\n\n"
            f"Use write_file for creating README.md and .gitignore. Use bash for git commands.\n"
            f"Do not ask questions — just do it."
        )

    elif command == "/new":
        # Scaffold a new project
        raw_args = args.strip() if args else ""
        if not raw_args:
            console.print("[#ebcb8b]Usage: /new <project-name> \\[description][/#ebcb8b]")
            console.print("[#8899aa]  Scaffolds a new project with git[/#8899aa]")
            console.print("[#8899aa]  Example: /new my-app[/#8899aa]")
            console.print("[#8899aa]  Example: /new weather-api a FastAPI weather service[/#8899aa]")
            return None

        parts = raw_args.split(None, 1)
        project_name = parts[0]
        description = parts[1] if len(parts) > 1 else ""

        desc_hint = ""
        if description:
            desc_hint = (
                f"The user described this project as: \"{description}\"\n"
                f"Use this to pick the right language, framework, and project structure.\n\n"
            )

        console.print(f"[#88c0d0]Scaffolding [bold]\"{project_name}\"[/bold]...[/#88c0d0]")

        return (
            f"Create a new project called \"{project_name}\". Steps:\n\n"
            f"{desc_hint}"
            f"1. Create the directory: `mkdir -p {project_name}`\n"
            f"2. Run `git init` inside it\n"
            f"3. Use write_file to create these files inside {project_name}/:\n"
            f"   - A main entry point file (e.g. main.py, index.js, main.go — pick based on the description)\n"
            f"   - A .gitignore appropriate for the language\n"
            f"   - A short README.md with the project name and description\n"
            f"4. If the language needs a dependency file (requirements.txt, package.json, go.mod), create it\n"
            f"5. Show the file tree when done\n"
            f"6. End with: \"Use `/publish` to push to GitHub when ready.\"\n\n"
            f"Use write_file for all file creation. Use bash only for mkdir and git init.\n"
            f"Do not ask questions — just do it."
        )

    elif command == "/init":
        # Generate a CLAUDE.md for this project (Claude Code parity). Use the
        # loader's nearest-ancestor discovery, not just $CWD/CLAUDE.md: running
        # /init from a subdirectory of a repo that already has a root CLAUDE.md
        # would otherwise write a new subdir CLAUDE.md that permanently SHADOWS
        # the repo one the loader actually reads.
        from spark_code.instructions import _find_project_claude_md
        existing_claude_md = _find_project_claude_md(os.getcwd())
        if existing_claude_md:
            console.print(
                f"[#8899aa]CLAUDE.md already exists at {existing_claude_md}. "
                "Remove or edit it directly if you want to regenerate it.[/#8899aa]"
            )
            return None
        return (
            "Analyze this project and create a CLAUDE.md file in the project root.\n"
            "Include: what the project is, build/test/run commands you can verify from\n"
            "project files, code style conventions you can observe, and any directory\n"
            "layout worth knowing. Keep it under 60 lines. Write the file with write_file."
        )

    elif command == "/run":
        # Run/launch the project in the current directory
        run_cmd = args.strip() if args else ""

        if run_cmd:
            # User specified a command: /run python app.py
            return f"__RUN_CMD__{run_cmd}"

        # Auto-detect: ask the agent to figure out how to run the project
        return (
            "Look at the files in the current directory and figure out how to "
            "run this project. Check for:\n"
            "- package.json (npm start / npm run dev)\n"
            "- pyproject.toml or setup.py (python -m or entry point)\n"
            "- main.py, app.py, server.py, index.js, etc.\n"
            "- Makefile (make run)\n"
            "- docker-compose.yml (docker compose up)\n\n"
            "Then run it using the bash tool. If it's a web server, "
            "note the URL. If it's a CLI app, show the output."
        )

    elif command == "/history":
        sessions = _list_sessions()
        if not sessions:
            console.print("[#8899aa]No saved sessions yet.[/#8899aa]")
            return None
        if args.strip():
            # Resume a specific session
            target = args.strip()
            matches = [s for s in sessions if target in os.path.basename(s["path"])]
            if not matches:
                console.print(f"[#bf616a]No session matching '{target}'[/#bf616a]")
                return None
            chosen = _load_session_meta(matches[0])
            if _resume_session_into(context, chosen["path"]):
                label = chosen["label"]
                label_display = f"  [#d8dee9]{label}[/#d8dee9]" if label else ""
                console.print(f"[#a3be8c]Resumed session: {os.path.basename(chosen['path'])}{label_display}[/#a3be8c]")
            else:
                console.print("[#bf616a]Failed to load session[/#bf616a]")
            return None
        # List recent sessions with metadata
        console.print("[bold #eceff4]Recent sessions:[/bold #eceff4]")
        for sess in sessions[:_SESSION_DISPLAY_LIMIT]:
            cwd = sess["cwd"]
            home = os.path.expanduser("~")
            if cwd.startswith(home):
                cwd = "~" + cwd[len(home):]

            # Build display line
            parts = []
            if sess["age"]:
                parts.append(f"[#8899aa]{sess['age']:<8}[/#8899aa]")
            if sess["turns"]:
                parts.append(f"[#8899aa]{sess['turns']} turns[/#8899aa]")
            if sess["label"]:
                parts.append(f"[#d8dee9]{_esc(str(sess['label']))}[/#d8dee9]")
            if cwd:
                parts.append(f"[#4c566a]{_esc(cwd)}[/#4c566a]")

            detail = "  ·  ".join(parts) if parts else ""
            console.print(f"  [#88c0d0]{sess['name']}[/#88c0d0]  {detail}")
        console.print("[#8899aa]Use /history <name> to resume  ·  spark --resume to continue last[/#8899aa]")
        return None

    elif command == "/resume":
        sessions = _list_sessions()
        if not sessions:
            console.print("[#8899aa]No saved sessions yet.[/#8899aa]")
            return None
        shown = sessions[:_SESSION_DISPLAY_LIMIT]

        def _finish_resume(chosen: dict) -> None:
            _load_session_meta(chosen)
            if _resume_session_into(context, chosen["path"]):
                label_display = f"  [#d8dee9]{chosen['label']}[/#d8dee9]" if chosen["label"] else ""
                console.print(f"[#a3be8c]Resumed session: {chosen['name']}{label_display}[/#a3be8c]")
                print_banner(
                    console, config,
                    skill_count=len(skills.all()),
                    instruction_sources=load_instructions(os.getcwd()).sources,
                    project_type=detect_project_type(os.getcwd()),
                    real_model_name=getattr(model, "real_model_name", "") or None,
                )
            else:
                console.print("[#bf616a]Failed to load session[/#bf616a]")

        if args.strip():
            # /resume <n> or /resume <name> — skip the picker
            target = args.strip()
            if target.isdigit():
                idx = int(target) - 1
                if idx < 0 or idx >= len(shown):
                    console.print(f"[#bf616a]No session #{target}. Use /resume to list.[/#bf616a]")
                    return None
                _finish_resume(shown[idx])
            else:
                matches = [s for s in sessions if target in s["name"]]
                if not matches:
                    console.print(f"[#bf616a]No session matching '{target}'[/#bf616a]")
                    return None
                _finish_resume(matches[0])
            return None

        # No args — show a numbered picker and prompt for a choice.
        console.print("[bold #eceff4]Pick a session to resume:[/bold #eceff4]")
        for i, sess in enumerate(shown, 1):
            parts = []
            if sess["age"]:
                parts.append(f"[#8899aa]{sess['age']:<8}[/#8899aa]")
            if sess["turns"]:
                parts.append(f"[#8899aa]{sess['turns']} turns[/#8899aa]")
            if sess["label"]:
                parts.append(f"[#d8dee9]{_esc(str(sess['label']))}[/#d8dee9]")
            detail = "  ·  ".join(parts) if parts else ""
            console.print(f"  [#88c0d0]{i}.[/#88c0d0] {sess['name']}  {detail}")

        try:
            answer = Prompt.ask(
                "[#8899aa]Resume which session? (number, blank to cancel)[/#8899aa]",
                default="",
            )
        except (KeyboardInterrupt, EOFError):
            console.print()
            return None
        answer = answer.strip()
        if not answer:
            console.print("[#8899aa]Cancelled.[/#8899aa]")
            return None
        if not answer.isdigit() or not (1 <= int(answer) <= len(shown)):
            console.print(f"[#bf616a]Invalid selection: {answer}[/#bf616a]")
            return None
        _finish_resume(shown[int(answer) - 1])
        return None

    elif command == "/undo":
        if args and args.strip() == "list":
            undo_dir = os.path.expanduser("~/.spark/.undo")
            if not os.path.isdir(undo_dir):
                console.print("[#8899aa]Nothing to undo.[/#8899aa]")
                return None
            undo_files = sorted(os.listdir(undo_dir), reverse=True)
            if not undo_files:
                console.print("[#8899aa]Nothing to undo.[/#8899aa]")
                return None
            # Show undo stack
            import json as _json
            console.print("[bold #eceff4]Undo stack:[/bold #eceff4]")
            for i, uf in enumerate(undo_files[:20], 1):
                try:
                    with open(os.path.join(undo_dir, uf), encoding="utf-8") as f:
                        data = _json.load(f)
                    home = os.path.expanduser("~")
                    p = data["path"]
                    display = "~" + p[len(home):] if p.startswith(home) else p
                    console.print(f"  [#88c0d0]{i}.[/#88c0d0] [#d8dee9]{display}[/#d8dee9]")
                except Exception:
                    console.print(f"  [#88c0d0]{i}.[/#88c0d0] [#8899aa](corrupt entry)[/#8899aa]")
            return None

        # Parse count: /undo 3 → undo last 3
        count = 1
        if args and args.strip().isdigit():
            count = int(args.strip())
        _undo_last(console, count)
        return None

    elif command == "/pin":
        if not pinned:
            console.print("[#bf616a]Pin system not available[/#bf616a]")
            return None
        if not args:
            files = pinned.list()
            if not files:
                console.print("[#8899aa]No pinned files. Use /pin <file_path>[/#8899aa]")
            else:
                console.print("[bold #eceff4]Pinned files:[/bold #eceff4]")
                home = os.path.expanduser("~")
                for f in files:
                    display = "~" + f[len(home):] if f.startswith(home) else f
                    console.print(f"  [#88c0d0]{display}[/#88c0d0]")
            return None
        ok, msg = pinned.pin(args.strip())
        style = "#a3be8c" if ok else "#bf616a"
        console.print(f"[{style}]{msg}[/{style}]")
        return None

    elif command == "/unpin":
        if not pinned:
            console.print("[#bf616a]Pin system not available[/#bf616a]")
            return None
        if not args:
            console.print("[#ebcb8b]Usage: /unpin <file_path>[/#ebcb8b]")
            return None
        ok, msg = pinned.unpin(args.strip())
        style = "#a3be8c" if ok else "#bf616a"
        console.print(f"[{style}]{msg}[/{style}]")
        return None

    elif command == "/git":
        if not args:
            console.print("[#ebcb8b]Usage: /git <command>[/#ebcb8b]")
            console.print("[#8899aa]  /git sync   — pull, then push[/#8899aa]")
            console.print("[#8899aa]  /git pr     — create a PR with AI-generated description[/#8899aa]")
            console.print("[#8899aa]  /git stash  — stash changes[/#8899aa]")
            console.print("[#8899aa]  /git log    — show recent commits[/#8899aa]")
            return None
        sub = args.strip().lower()
        if sub == "sync":
            return "__RUN_CMD__git pull --rebase && git push"
        elif sub == "stash":
            return "__RUN_CMD__git stash"
        elif sub == "log":
            return "__RUN_CMD__git log --oneline -15"
        elif sub == "pr":
            return (
                "Create a pull request for the current branch. Steps:\n"
                "1. Run `git branch --show-current` to get branch name\n"
                "2. Run `git log main..HEAD --oneline` to see commits\n"
                "3. Run `git diff main...HEAD --stat` to see changed files\n"
                "4. Generate a concise PR title and description based on the changes\n"
                "5. Run `gh pr create --title \"<title>\" --body \"<description>\"` to create the PR\n"
                "6. Show the PR URL when done"
            )
        else:
            # Pass through as git command
            return f"__RUN_CMD__git {args.strip()}"

    elif command == "/fork":
        # Save current session and start fresh
        from datetime import datetime as _dt
        try:
            history_dir = _sessions_dir()
            os.makedirs(history_dir, exist_ok=True)
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            label = _make_session_label(context)
            suffix = f"_{label}" if label else ""
            save_path = os.path.join(history_dir, f"{ts}{suffix}_fork.json")
            context.save(save_path, label=f"{label} (forked)", cwd=os.getcwd())
            console.print("[#a3be8c]Session saved as fork. Starting fresh context.[/#a3be8c]")
            console.print(f"[#8899aa]Resume the fork with /history {ts}[/#8899aa]")
            context.clear()
        except Exception as e:
            console.print(f"[#bf616a]Fork failed: {e}[/#bf616a]")
        return None

    elif command == "/snippet":
        if not snippets:
            console.print("[#bf616a]Snippet system not available[/#bf616a]")
            return None
        if not args:
            all_snippets = snippets.list()
            if not all_snippets:
                console.print("[#8899aa]No snippets. Use /snippet save <name> <prompt>[/#8899aa]")
            else:
                console.print("[bold #eceff4]Saved snippets:[/bold #eceff4]")
                for name, prompt in all_snippets.items():
                    preview = prompt[:60].replace("\n", " ")
                    console.print(f"  [#88c0d0]{_esc(name)}[/#88c0d0]  [#8899aa]{_esc(preview)}[/#8899aa]")
            return None
        sub_parts = args.strip().split(maxsplit=2)
        if sub_parts[0].lower() == "save":
            if len(sub_parts) < 3:
                console.print("[#ebcb8b]Usage: /snippet save <name> <prompt>[/#ebcb8b]")
                return None
            msg = snippets.add(sub_parts[1], sub_parts[2])
            console.print(f"[#a3be8c]{msg}[/#a3be8c]")
            return None
        elif sub_parts[0].lower() == "remove":
            if len(sub_parts) < 2:
                console.print("[#ebcb8b]Usage: /snippet remove <name>[/#ebcb8b]")
                return None
            msg = snippets.remove(sub_parts[1])
            console.print(f"[#8899aa]{msg}[/#8899aa]")
            return None
        else:
            # Run a snippet by name
            prompt = snippets.get(sub_parts[0])
            if prompt:
                return prompt
            console.print(f"[#bf616a]Snippet not found: {sub_parts[0]}[/#bf616a]")
            return None

    elif command == "/export":
        # Export session as markdown
        if context.turn_count == 0:
            console.print("[#8899aa]Nothing to export — no conversation yet.[/#8899aa]")
            return None
        lines = ["# Spark Code Session Export\n"]
        for msg in context.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and isinstance(content, str):
                lines.append(f"## User\n\n{content}\n")
            elif role == "assistant" and isinstance(content, str) and content:
                lines.append(f"## Assistant\n\n{content}\n")
            elif role == "tool":
                name = msg.get("name", "tool")
                result_preview = (content[:200] + "...") if len(content or "") > 200 else content
                lines.append(f"**Tool: {name}**\n```\n{result_preview}\n```\n")
        export_text = "\n".join(lines)
        export_path = os.path.join(os.getcwd(), "session_export.md")
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(export_text)
            console.print(f"[#a3be8c]Exported to {export_path}[/#a3be8c]")
            _copy_to_clipboard(export_text)
            console.print("[#8899aa]Also copied to clipboard[/#8899aa]")
        except OSError as e:
            console.print(f"[#bf616a]Export failed: {e}[/#bf616a]")
        return None

    elif command == "/cost":
        # Show running cost information
        in_tokens = model.total_input_tokens
        out_tokens = model.total_output_tokens
        cost = model.estimated_cost
        provider = get(config, "model", "provider", default="ollama")
        console.print("[bold #eceff4]Session Cost[/bold #eceff4]")
        console.print(f"  [#88c0d0]Provider:[/#88c0d0] [#d8dee9]{provider}[/#d8dee9]")
        console.print(f"  [#88c0d0]Input tokens:[/#88c0d0] [#d8dee9]{in_tokens:,}[/#d8dee9]")
        console.print(f"  [#88c0d0]Output tokens:[/#88c0d0] [#d8dee9]{out_tokens:,}[/#d8dee9]")
        if cost > 0:
            console.print(f"  [#88c0d0]Estimated cost:[/#88c0d0] [#a3be8c]${cost:.4f}[/#a3be8c]")
            budget = get(config, "budget_limit", default=0)
            if budget and cost > float(budget):
                console.print(f"  [#bf616a]Over budget limit (${budget})![/#bf616a]")
            elif budget:
                remaining = float(budget) - cost
                console.print(f"  [#8899aa]Budget remaining: ${remaining:.4f}[/#8899aa]")
        else:
            console.print("  [#8899aa]Local model — no cost[/#8899aa]")
        return None

    elif command == "/watch":
        if not args:
            console.print("[#ebcb8b]Usage: /watch <command> — auto-run on file changes[/#ebcb8b]")
            console.print("[#8899aa]  /watch pytest       — run tests on change[/#8899aa]")
            console.print("[#8899aa]  /watch npm test     — run JS tests on change[/#8899aa]")
            console.print("[#8899aa]  /watch off          — stop watching[/#8899aa]")
            return None
        return f"__WATCH__{args.strip()}"

    elif command == "/loop":
        from .loop import build_loop_sentinel
        sentinel = build_loop_sentinel(args)
        if sentinel is None:
            console.print("[#ebcb8b]Usage: /loop \\[every <interval>] [--check \"<cmd>\"] <goal>[/#ebcb8b]")
            console.print("[#8899aa]  /loop fix the failing tests            — rounds until done[/#8899aa]")
            console.print("[#8899aa]  /loop --check \"pytest -q\" fix tests    — done when check passes[/#8899aa]")
            console.print("[#8899aa]  /loop every 30m keep the build green   — repeat on a timer[/#8899aa]")
            console.print("[#8899aa]  /loop status · /loop stop · Esc stops a running loop[/#8899aa]")
            return None
        return sentinel

    elif command == "/checkpoint":
        # Create a git stash checkpoint
        try:
            result = subprocess.run(
                ["git", "stash", "push", "-m",
                 f"spark-checkpoint-{int(__import__('time').time())}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if "No local changes" in output:
                    console.print("[#8899aa]No changes to checkpoint.[/#8899aa]")
                else:
                    console.print(f"[#a3be8c]Checkpoint created: {output}[/#a3be8c]")
                    # Re-apply the changes so the working tree is unchanged while
                    # the stash entry REMAINS as the checkpoint. `pop` would have
                    # deleted the very checkpoint we just made.
                    apply_result = subprocess.run(
                        ["git", "stash", "apply", "--index"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if apply_result.returncode == 0:
                        console.print("[#8899aa]Working state preserved. Use /rollback to restore.[/#8899aa]")
                    else:
                        # The stash push SUCCEEDED (changes are safely stashed)
                        # but re-applying to the working tree failed — do NOT
                        # claim the tree is preserved; point at the recovery path.
                        err = apply_result.stderr.strip() or "git stash apply failed"
                        console.print(f"[#bf616a]Could not restore working tree: {err}[/#bf616a]")
                        console.print("[#ebcb8b]Your changes are saved in the stash — run /rollback (or `git stash apply`) to recover them.[/#ebcb8b]")
            else:
                console.print(f"[#bf616a]Checkpoint failed: {result.stderr.strip()}[/#bf616a]")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            console.print("[#bf616a]Git not available[/#bf616a]")
        return None

    elif command == "/rollback":
        # Show and restore from git stash checkpoints
        checkpoints = _get_checkpoints()
        if checkpoints is None:
            console.print("[#bf616a]Git not available[/#bf616a]")
            return None
        if not checkpoints:
            console.print("[#8899aa]No checkpoints found. Use /checkpoint first.[/#8899aa]")
            return None

        if args and args.strip().isdigit():
            idx = int(args.strip())
            _apply_checkpoint(console, checkpoints, idx)
        else:
            console.print("[bold #eceff4]Checkpoints:[/bold #eceff4]")
            for i, (_ref, stash) in enumerate(checkpoints[:10]):
                console.print(f"  [#88c0d0]{i}.[/#88c0d0] [#d8dee9]{stash}[/#d8dee9]")
            console.print("[#8899aa]Use /rollback <number> to restore[/#8899aa]")
        return None

    elif command == "/rewind":
        # Unified restore: the undo stack (last write/edit snapshots), the
        # latest /checkpoint git stash, the conversation itself (per-turn
        # snapshot ring — Phase 5 Task 7), or all three. Delegates to
        # /undo's, /rollback's, and Context.rewind_conversation's own logic
        # via the shared _undo_last/_rewind_files/_rewind_conversation
        # helpers so there's exactly one implementation of each restore
        # action. The Phase 1 honest labels (plan amendment 2026-07-06) said
        # each option must do exactly what it claims — "conversation" now
        # really does restore the message history, not just files.
        alias = {"undo": "1", "checkpoint": "2", "both": "3", "conversation": "4"}
        choice = args.strip().lower() if args else ""
        choice = alias.get(choice, choice)
        if choice and choice not in ("1", "2", "3", "4"):
            console.print(f"[#bf616a]Unknown /rewind option: {args.strip()}[/#bf616a]")
            console.print("[#8899aa]Use 1 (last file edits), 2 (checkpoint), 3 (both), or 4 (conversation)[/#8899aa]")
            return None

        if not choice:
            console.print("[bold #eceff4]Rewind:[/bold #eceff4]")
            console.print("  [#88c0d0]1.[/#88c0d0] last file edits (undo stack)")
            console.print("  [#88c0d0]2.[/#88c0d0] checkpoint (git stash)")
            console.print("  [#88c0d0]3.[/#88c0d0] both (files + conversation)")
            console.print("  [#88c0d0]4.[/#88c0d0] conversation (last turn)")
            try:
                choice = Prompt.ask(
                    "[#8899aa]Rewind what?[/#8899aa]",
                    choices=["1", "2", "3", "4"],
                    default="1",
                )
            except (KeyboardInterrupt, EOFError):
                console.print()
                return None

        if choice == "1":
            _undo_last(console, 1)
        elif choice == "2":
            _rewind_files(console)
        elif choice == "3":
            _undo_last(console, 1)
            _rewind_files(console)
            _rewind_conversation(console, context)
        else:  # "4"
            _rewind_conversation(console, context)
        return None

    elif command == "/profile":
        # Model performance benchmark
        return "__PROFILE__"

    elif command == "/benchmark":
        return "__BENCHMARK__"

    elif command == "/apply":
        if not args:
            console.print("[#ebcb8b]Usage: /apply <url> — apply code from a URL[/#ebcb8b]")
            console.print("[#8899aa]  Supports: GitHub gist, PR diff, raw file URLs[/#8899aa]")
            return None
        url = args.strip()
        return (
            f"Fetch the content from this URL and apply it to the local codebase:\n"
            f"URL: {url}\n\n"
            f"Steps:\n"
            f"1. Use web_fetch to get the content from the URL\n"
            f"2. If it's a diff/patch, apply it using edit_file\n"
            f"3. If it's a full file, write it using write_file\n"
            f"4. If it's a gist with multiple files, create each one\n"
            f"5. Show what was applied"
        )

    elif command == "/teach":
        if not args:
            console.print("[#ebcb8b]Usage: /teach <name> <description> -- <command>[/#ebcb8b]")
            console.print("[#8899aa]  Example: /teach deploy \"Deploy to production\" -- ssh prod 'cd /app && git pull'[/#8899aa]")
            console.print("[#8899aa]  Example: /teach lint \"Run linter\" -- ruff check --fix .[/#8899aa]")
            return None
        return f"__TEACH__{args.strip()}"

    elif command == "/branch":
        return f"__BRANCH__{args.strip() if args else ''}"

    elif command == "/switch":
        if not args:
            console.print("[#ebcb8b]Usage: /switch <branch-name>[/#ebcb8b]")
            return None
        return f"__SWITCH__{args.strip()}"

    elif command == "/branches":
        return "__BRANCHES__"

    elif command == "/share":
        if context.turn_count == 0:
            console.print("[#8899aa]Nothing to share — no conversation yet.[/#8899aa]")
            return None
        return f"__SHARE__{args.strip() if args else ''}"

    elif command == "/search":
        if not args:
            console.print("[#ebcb8b]Usage: /search <query> — search past sessions[/#ebcb8b]")
            return None
        return f"__SEARCH__{args.strip()}"

    elif command == "/analytics":
        return "__ANALYTICS__"

    elif command == "/retry":
        return "__RETRY__"

    elif command == "/continue":
        return "__CONTINUE__"

    elif command == "/docs":
        if not args.strip():
            console.print("[#ebcb8b]Usage: /docs <query> — search indexed knowledge base[/#ebcb8b]")
            console.print("[#8899aa]Searches: Swift docs, SwiftUI, Apple HIG, App Store Guidelines, CNN, and more[/#8899aa]")
            return None
        return f"Use rag_search to find information about: {args.strip()}. Show the most relevant results with source citations."

    elif command == "/index":
        # Builds/refreshes this project's code_search collection. Deferred via
        # a sentinel — index_project is a coroutine and this handler is sync
        # (see the __INDEX__ branch in the async REPL loop).
        return "__INDEX__"

    elif command == "/clean":
        return "__CLEAN__"

    elif command == "/review":
        # The /review COMMAND supersedes the builtin `review` skill: it runs the
        # multi-agent swarm (async, non-streamed) rather than injecting a prompt.
        # Deferred via a sentinel because this handler is sync. Optional arg is a
        # path target for the diff.
        return f"__REVIEW__{args.strip()}"

    elif command == "/open":
        # Hand a file off to the host editor (Phase 3 Task 4). Never blocks
        # or raises — every failure path (no file, no editor, launch error)
        # is a single dim console note, same message so users always land
        # on the same fix ("set ui.editor").
        no_editor_note = "[#8899aa]No editor detected — set ui.editor[/#8899aa]"
        target = args.strip()
        if not target:
            console.print("[#ebcb8b]Usage: /open <file[:line]>[/#ebcb8b]")
            return None

        file_part, sep, line_part = target.rpartition(":")
        if sep and line_part.isdigit():
            raw_path, line = file_part, int(line_part)
        else:
            raw_path, line = target, None

        path = os.path.abspath(os.path.expanduser(raw_path))
        if not os.path.exists(path):
            console.print(f"[#bf616a]No such file: {path}[/#bf616a]")
            return None

        editor = _resolve_editor(config)
        if not editor or not open_in_editor(editor, path, line):
            console.print(no_editor_note)
        return None

    # Check skills (skip /plan since we handle it above)
    skill = skills.get(command)
    if skill:
        if skill.requires_args and not args.strip():
            console.print(f"[#ebcb8b]Usage: {command} <prompt>[/#ebcb8b]")
            console.print(f"[#8899aa]{skill.description}[/#8899aa]")
            return None
        # Pre-flight compatibility check for skills that depend on external services
        # (e.g. kuiper daemon for /app-gap-research). Warn but don't block — user can
        # always start the service and retry, or proceed knowing it will likely fail.
        ok, reason = check_skill_compatibility(skill)
        if not ok:
            console.print(f"[#ebcb8b]⚠ {reason}[/#ebcb8b]")
            console.print("[#8899aa]Skill will run anyway. Start the dependency first if it needs one.[/#8899aa]")
        return skill.get_prompt(args)

    console.print(f"[yellow]Unknown command: {command}. Type /help for available commands.[/yellow]")
    return None


def _model_arg_candidates(config: dict) -> list[str]:
    """/model argument candidates: the literal "list" subcommand plus the
    configured provider names — /model accepts either. Without "list" here,
    wiring provider names as /model's value provider would shadow the real
    "/model list" subcommand in autocomplete (review fast-follow 2026-07-07):
    the argument branch returns early, so the whole-phrase match that used to
    suggest "/model list" is never reached once a provider is registered.
    """
    return ["list"] + list(config.get("providers", {}).keys())


def _mode_arg_candidates() -> list[str]:
    """/mode argument candidates, derived from the permissions constants —
    the single source of truth — rather than a hand-typed duplicate that
    could drift when a mode is added or renamed: native names first, then
    the Claude Code alias spellings that aren't already a native name.
    DISPLAY_ALIASES supplies the original camelCase forms ("acceptEdits")
    that MODE_ALIASES' lowercased keys destroy.
    """
    return sorted(NATIVE_MODES) + sorted(
        a for a in DISPLAY_ALIASES if a.lower() not in NATIVE_MODES
    )


def _resolve_editor(config: dict) -> str | None:
    """Resolve the effective host editor from ``ui.editor`` (Phase 3 Task 4).

    A config override ("cursor"/"code"/"xed"/"none") wins outright —
    ``detect_editor`` is never consulted in that case, matching its own
    "config override wins (caller passes it)" contract. "auto" (the
    default) runs real detection via the real environment/PATH. "none"
    (or anything unrecognized) resolves to None, which disables both
    ``/open`` and OSC 8 file-link styling.
    """
    editor_cfg = get(config, "ui", "editor", default="auto")
    if editor_cfg == "auto":
        return detect_editor()
    if editor_cfg in ("cursor", "code", "xed"):
        return editor_cfg
    return None


def _prompt_with_patched_stdout(session):
    """Run ``session.prompt()`` with ``patch_stdout`` scoped to just this call.

    This runs inside ``run_in_executor``'s worker thread (the same thread that
    calls ``session.prompt()``) — prompt_toolkit's ``StdoutProxy`` captures the
    active ``AppSession``/output at construction time and the ``PromptSession``
    it wraps must run in that same scope, so entering ``patch_stdout`` anywhere
    else (e.g. the event-loop thread, around the ``run_in_executor`` call
    instead of inside it) would be relying on incidental context-propagation
    behavior of the executor rather than a guaranteed pairing.

    While this is active, anything written to ``sys.stdout``/``sys.stderr``
    (e.g. the file watcher's or team monitor's background prints, which run as
    ordinary asyncio tasks independent of the input loop) is drawn ABOVE the
    prompt instead of mangling the input line. This must never wrap
    generation — Rich ``Live`` (StreamingRenderer, started only inside
    ``_run_with_notify``) owns the screen there, and the two scopes are
    disjoint by construction: the main loop calls this helper only while
    idling for input, never while a generation task is running.
    """
    with patch_stdout(raw=True):
        return session.prompt()


def _loop_escape_action(engine, agent) -> None:
    """Esc during a /loop: stop the loop AND cancel the in-flight round.

    ``engine.request_stop()`` only flips a flag the loop checks BETWEEN rounds,
    so on its own Esc appears dead until the current round finishes (up to
    ``round_timeout_s`` — as long as 15 min). Also calling ``agent.cancel()`` —
    the same clean-cancel Ctrl+C uses — makes the running generation return its
    partial promptly; the round then ends and the loop honors the stop flag.
    Extracted to module scope so the Esc wiring is unit-testable.
    """
    engine.request_stop()
    agent.cancel()


async def run_interactive(config: dict, resume_session: str = "",
                          continue_prompt: str = ""):
    """Run interactive CLI session.

    resume_session: path to a session JSON to resume on startup
    continue_prompt: if set, resume last session and send this prompt
    """
    theme_name = get(config, "ui", "theme", default="dark")
    console = Console(theme=get_theme(theme_name))
    ensure_dirs()

    # Initialize skills
    skills = SkillRegistry()
    skills.load_all()

    # Phase 5 Task 4: custom subagent definitions (.spark/agents/*.md +
    # ~/.spark/agents/*.md). A malformed file is skipped with a logged
    # warning inside load_agent_defs — never raises, so this can never fail
    # startup the way an unguarded skill-index build once did.
    agent_defs = load_agent_defs(os.getcwd())

    # Initialize pinned files and snippets
    pinned = PinnedFiles()
    snippet_lib = SnippetLibrary()

    # Initialize MCP
    mcp_client = MCPClient()
    mcp_configs = get(config, "mcp_servers", default={})
    mcp_tools = []
    if mcp_configs:
        console.print("[dim]Connecting to MCP servers...[/dim]")
        mcp_tools = await mcp_client.connect_all(mcp_configs)

    # Initialize memory
    memory = Memory(
        global_path=get(config, "memory", "global_path", default="~/.spark/memory"),
        bridge_budget_chars=get(config, "memory", "bridge_budget_chars", default=4000),
    )
    memory_context = memory.load_all()
    agentic = config.get("_agentic", False)
    system_prompt = AGENTIC_PROMPT if agentic else SYSTEM_PROMPT

    # Load unified project instructions: SPARK.md + the CLAUDE.md hierarchy
    # (project + global), Claude Code parity — see spark_code/instructions.py.
    instructions = load_instructions(os.getcwd())
    if instructions.text:
        system_prompt += f"\n\n{instructions.text}"

    if memory_context:
        system_prompt += f"\n\n{memory_context}"

    # Smart project detection
    project_type = detect_project_type(os.getcwd())
    if project_type:
        system_prompt += f"\n\nThis is a {project_type}."

    # 80B-friendly skills (Task 8): ONLY a compact index (name + one-line
    # description, capped) ever enters the system prompt — never full skill
    # bodies. A skill's full prompt enters context only on invocation, via
    # the slash-command fallback below or agent.py's auto-expand. Skills
    # whose names collide with real commands (e.g. the builtin `review`
    # skill vs. the /review swarm command) are excluded — at the CLI those
    # names resolve to the command, so advertising the skill would teach the
    # model a name that expands to something else.
    system_prompt += build_skill_prompt_section(
        skills.build_index(exclude=RESERVED_COMMAND_NAMES))

    # Verification habit (Task 6): detect the project's test command once —
    # instructions text checked first (an explicit CLAUDE.md/SPARK.md
    # instruction wins), falling back to project-type markers. None → the
    # Agent's nudge is silently off (nothing detectable to verify with).
    test_command = detect_test_command(os.getcwd(), instructions.text)

    # Single /v1/models fetch feeds both the real-model-name banner display
    # right below and the context-window preflight check further down —
    # startup used to hit /v1/models twice (once per consumer); now once.
    # timeout=5.0 preserves the preflight check's original budget (the
    # stricter of the two consumers' needs; the banner tolerates it).
    server_models = await fetch_server_models(
        get(config, "model", "endpoint", default=""),
        get(config, "model", "api_key", default=""),
        timeout=5.0,
    )

    # Resolve real model name (unmasks vLLM --served-model-name aliasing).
    # Pass the configured name so the correct entry is picked on multi-model
    # servers (Ollama lists every pulled model).
    real_model_name = _pick_real_model_name(
        server_models or [], get(config, "model", "name", default=""))

    # Print banner
    print_banner(console, config, mcp_count=len(mcp_tools),
                 skill_count=len(skills.all()),
                 instruction_sources=instructions.sources,
                 project_type=project_type,
                 real_model_name=real_model_name)

    # Initialize components
    model = ModelClient(
        endpoint=get(config, "model", "endpoint"),
        model=get(config, "model", "name"),
        temperature=get(config, "model", "temperature", default=0.7),
        max_tokens=get(config, "model", "max_tokens", default=8192),
        api_key=get(config, "model", "api_key", default=""),
        provider=get(config, "model", "provider", default="ollama"),
        timeout=float(get(config, "model", "timeout", default=300)),
        real_model_name=real_model_name or "",
        supports_vision=bool(get(config, "model", "vision", default=False)),
        display_name=get(config, "model", "display_name", default=""),
    )

    # Opt-in provider failover: if the config declares a `fallback.chain`,
    # wrap the model in a FallbackChain that auto-switches providers when one
    # is unreachable. Dormant (no behavior change) when unconfigured.
    _fb_cfg = config.get("fallback", {}) or {}
    if _fb_cfg.get("chain"):
        from .fallback import FallbackChain

        def _model_factory(name, pconf):
            # Phase 5 Task 2: a failover provider set up via /setkey (key in
            # ~/.spark/keys, no api_key in its config block) must still work
            # when the chain switches to it — this is precisely the "DGX is
            # down" path the feature exists for.
            return ModelClient(
                endpoint=pconf.get("endpoint", "http://localhost:11434"),
                model=pconf.get("model", "unknown"),
                temperature=pconf.get("temperature", 0.7),
                max_tokens=pconf.get("max_tokens", 8192),
                api_key=resolve_provider_key(name, pconf, load_keys()),
                provider=name,
                timeout=float(pconf.get("timeout", 300)),
                supports_vision=bool(pconf.get("vision", False)),
                display_name=pconf.get("display_name", ""),
            )

        await model.close()  # the chain manages its own per-provider clients
        model = FallbackChain(config.get("providers", {}), _fb_cfg, _model_factory)
        console.print(f"  [#8899aa]Fallback chain: {' → '.join(_fb_cfg['chain'])}[/#8899aa]")

    # Startup connection check (non-blocking)
    try:
        ok, msg = await model.ping()
        if ok:
            console.print(f"  [#a3be8c]{msg}[/#a3be8c]")
            # Preload model into VRAM in background
            asyncio.create_task(_warmup_model(model))

            # Context-window drift check: does the configured context_window
            # match what the live server actually reports for this model?
            # Reuses the /v1/models payload fetched once above — no second
            # request.
            server_max = _extract_max_context(
                server_models, get(config, "model", "name", default=""))
            warning = context_window_warning(
                get(config, "model", "context_window", default=32768), server_max)
            if warning:
                console.print(f"  [#ebcb8b]⚠ {warning}[/#ebcb8b]")
        else:
            console.print(f"  [#ebcb8b]{msg}[/#ebcb8b]")
            console.print("  [#8899aa]Session will start anyway — requests may fail until server is available[/#8899aa]")
    except Exception:
        pass  # Don't block startup

    if agentic:
        console.print("  [bold #ebcb8b]⚡ Agent mode[/bold #ebcb8b] [#8899aa]— fully autonomous, trust all tools[/#8899aa]")
    console.print()

    # Initialize session stats
    session_stats = SessionStats()
    session_stats.set_cost_rates(
        input_rate=get(config, "model", "cost_per_million_input", default=0),
        output_rate=get(config, "model", "cost_per_million_output", default=0),
    )

    platform_prompt = format_platform_prompt(os.getcwd())
    provider_prompt = get(config, "model", "system_prompt", default="")

    context = Context(
        system_prompt=system_prompt,
        max_tokens=get(config, "model", "context_window", default=32768),
        platform_prompt=platform_prompt,
        provider_prompt=provider_prompt,
    )
    # Build permissions BEFORE tools so dispatch_agent can read the lead's
    # live mode (Shift+Tab changes .mode in place → sub-agents inherit it).
    permissions = PermissionManager(
        mode=get(config, "permissions", "mode", default="ask"),
        always_allow=get(config, "permissions", "always_allow", default=[]),
    )
    # Shared plan-mode state — one object held by the CLI (Shift+Tab, slash
    # handlers, toolbar) AND the Agent's tool-execution enforcement, so they can
    # never disagree. Replaces the old loose config["_plan_mode"] dict flag.
    plan_state = PlanState()
    todo_list = TodoList()
    # Dual-model routing (Phase 4 Task 2): built here (rather than further
    # below, its original spot) so build_tools can hand it to
    # DispatchAgentTool for a custom agent def's model_hint=="utility"
    # (Phase 5 Task 4). Same client instance is reused below for compaction —
    # see the "Initialize tool cache" section for the rest of its lifecycle
    # commentary.
    utility_model = get_utility_client(config)
    tools = build_tools(todo_list=todo_list, model=model, config=config,
                        permissions=permissions, plan_state=plan_state,
                        console=console, agent_defs=agent_defs,
                        utility_model=resolve_utility_for(config, "dispatch", utility_model),
                        mcp_tools=mcp_tools, interactive=True)

    # Register MCP tools on the main model — but skip agent-only servers (e.g.
    # the browser), whose tools go only to sub-agents so their huge outputs
    # never flood the local model's context.
    _register_mcp_tools(tools, mcp_tools, config)

    # Progress tracking for toolbar
    current_tool = {"name": "", "detail": ""}

    def _on_tool_start(tool_name, args):
        """Update toolbar with current tool being executed."""
        detail = ""
        if tool_name == "read_file":
            detail = args.get("file_path", "")
        elif tool_name == "bash":
            cmd = args.get("command", "")
            detail = cmd[:40] + "..." if len(cmd) > 40 else cmd
        elif tool_name in ("glob", "grep"):
            detail = args.get("pattern", "")
        elif tool_name in ("write_file", "edit_file"):
            detail = args.get("file_path", "")
        home = os.path.expanduser("~")
        if detail.startswith(home):
            detail = "~" + detail[len(home):]
        current_tool["name"] = tool_name
        current_tool["detail"] = detail

    # Initialize tool cache
    tool_cache = ToolCache(ttl=120.0, max_entries=300)

    # Initialize hooks
    hook_manager = HookManager(config)

    # Initialize branch manager
    branch_manager = BranchManager()

    # Initialize custom tools. Reserve every LIVE built-in tool name (plus the
    # static safety-net list) so a /teach tool can never shadow a real tool and
    # inherit its always_allow grant — `tools` here holds only built-ins (custom
    # tools are registered just below), so its names are exactly the reserved set.
    reserved = set(tools.names()) | set(BUILTIN_TOOL_NAMES)
    custom_tool_registry = CustomToolRegistry(reserved_names=reserved)
    for ct in custom_tool_registry.all():
        tools.register(ct)

    # Initialize file watcher (lazy — started by /watch)
    file_watcher = None
    # Most recent /loop engine this session (for /loop status); the loop
    # itself owns the prompt while running, so there's only ever one live.
    last_loop_engine = None

    # Dual-model routing (Phase 4 Task 2): an optional cheaper utility model
    # (default: the real 30B on Ollama, see routing.DEFAULT_UTILITY_MODEL)
    # for compaction summaries and the /review swarm — built ABOVE (before
    # build_tools, so DispatchAgentTool can see it too) and reused for the
    # rest of the session. Its lifecycle is independent of the primary
    # `model` — a /model switch below only rebinds `model`, and session
    # teardown closes each separately. Building a ModelClient just sets up
    # an httpx.AsyncClient (no request, no VRAM load), so an unused or
    # use_for-gated-off utility model costs nothing even when GPU memory is
    # tight next to a big primary model — see get_utility_client.

    agent = Agent(model, context, tools, permissions, console,
                  stats=session_stats, on_tool_start=_on_tool_start,
                  tool_cache=tool_cache, hooks=hook_manager,
                  result_budgets=get(config, "tools", "result_budgets", default=None),
                  plan_state=plan_state, test_command=test_command,
                  skills=skills, editor=_resolve_editor(config),
                  diff_in_editor=get(config, "ui", "diff_in_editor", default=False),
                  utility_model=resolve_utility_for(config, "compaction", utility_model),
                  agent_defs=agent_defs)

    # Initialize team system — optionally use a faster model for workers
    task_store = TaskStore()
    worker_model_obj = None
    worker_model_name = get(config, "model", "worker_model", default="")
    if worker_model_name:
        # If worker_endpoint is set in the active profile, send worker traffic there
        # (e.g. a separate vLLM serving a code-specialist on a different port).
        # Otherwise workers share the main endpoint.
        worker_endpoint = (
            get(config, "model", "worker_endpoint", default="")
            or get(config, "model", "endpoint", default="")
        )
        worker_model_obj = ModelClient(
            endpoint=worker_endpoint,
            model=worker_model_name,
            temperature=get(config, "model", "temperature", default=0.3),
            max_tokens=get(config, "model", "max_tokens", default=8192),
            api_key=get(config, "model", "api_key", default=""),
            provider=get(config, "model", "provider", default="ollama"),
            display_name=get(config, "model", "display_name", default=""),
        )
        # Show the friendly name only when workers run the SAME wire model as the
        # main client (so a distinct worker_model still shows its own real id).
        _disp = get(config, "model", "display_name", default="")
        _worker_disp = (_disp if _disp and worker_model_name == get(config, "model", "name", default="")
                        else worker_model_name)
        console.print(f"  [#88c0d0]Workers will use: {_worker_disp}[/#88c0d0]")
    team_manager = TeamManager(model, tools, console, task_store,
                               stats=session_stats, worker_model=worker_model_obj,
                               worker_permission_mode=permissions.mode)

    # Give the lead agent the ability to spawn workers
    spawn_tool = SpawnWorkerTool()
    spawn_tool.set_team_manager(team_manager)
    tools.register(spawn_tool)

    # Register wait_for_workers tool (only useful when team is available)
    tools.register(WaitForWorkersTool(team=team_manager))

    # Resume session if requested
    if resume_session:
        if _resume_session_into(context, resume_session):
            meta = Context.read_metadata(resume_session)
            label = meta.get("label", "")
            turns = meta.get("turn_count", 0)
            label_str = f" — {label}" if label else ""
            console.print(f"  [#a3be8c]Resumed session ({turns} turns{label_str})[/#a3be8c]")
            console.print()
        else:
            console.print(f"  [#ebcb8b]Could not load session: {resume_session}[/#ebcb8b]")

    # Callbacks for the prompt_toolkit footer (always visible below input).
    # One-source refactor (Task 3 review): the formatting for both toolbar
    # lines now lives in ui/input.py's toolbar_status_segments /
    # toolbar_mode_segments (thin wrappers over build_status_segments, the
    # same source the inline non-CPR fallback renders from), golden-tested
    # byte-identical to the closures they replaced. These closures only
    # gather live state.
    def _display_mode() -> str:
        """The mode name shown to the user — plan wins over the underlying
        permission mode while plan mode is active. Single copy (review fix:
        previously duplicated verbatim in mode_callback, mode_switch and the
        inline status fallback)."""
        return "plan" if plan_state.active else permissions.mode

    def _context_pct_and_style() -> tuple[float | None, str]:
        """Context-left percentage + its color style — server-reported usage
        (real tokenizer count) when the model has reported it, falling back
        to the char-based estimate otherwise (context.context_left handles
        both, and a zero/missing window safely). Colored by how much room is
        left so a near-full context is visible before it 400s. None when no
        percentage is computable (callers then show a raw count or nothing).
        """
        tokens = context.estimate_tokens()
        max_tok = context.max_tokens
        if max_tok > 0 and tokens > 0:
            pct = context.context_left(max_tok) * 100
            return pct, _context_pct_style(pct)
        return None, "class:bottom-toolbar.context"

    def status_callback():
        """Line 1: model + turns + speed/cost + context %."""
        context_pct, context_style = _context_pct_and_style()
        return toolbar_status_segments(
            model_name=(get(config, "model", "display_name", default="")
                        or get(config, "model", "name", default="")),
            provider_name=get(config, "model", "provider", default=""),
            turns=context.turn_count,
            speed_str=session_stats.format_speed() if session_stats else "",
            cost_str=session_stats.format_cost() if session_stats else "",
            context_pct=context_pct,
            context_style=context_style,
            context_tokens=context.estimate_tokens(),
        )

    def mode_switch():
        """Cycle modes: ask → auto → plan → ask (Shift+Tab).

        Trust is not in the cycle (matches Claude Code: bypassPermissions is
        never landed on via Shift+Tab) — it's reached only via /trust,
        --trust, or /mode trust. If the current mode is trust when
        Shift+Tab is pressed, it's not in the cycle so it restarts at "auto".

        The plan/permission bookkeeping lives in cycle_mode (shared, tested);
        entering plan drops permissions to "auto" (reads free, writes blocked by
        the plan gate). Persist the non-plan mode into config so it survives.
        """
        next_mode = cycle_mode(_display_mode(), plan_state, permissions)
        if next_mode != "plan":
            config["permissions"]["mode"] = next_mode

    def mode_callback():
        """Line 2: ⏵⏵ mode on · shift+tab to switch · ctrl+t team."""
        return toolbar_mode_segments(
            _display_mode(),
            team_active=team_manager.active_count,
            team_total=len(team_manager.workers),
        )

    def team_callback():
        """Line 3+: live worker status (only shown when workers exist)."""
        if not team_manager.workers:
            return None

        # Count by status
        running = sum(1 for w in team_manager.workers.values() if w.status == "running")
        done = sum(1 for w in team_manager.workers.values() if w.status == "completed")
        failed = sum(1 for w in team_manager.workers.values() if w.status == "failed")

        parts = []
        parts.append(("class:bottom-toolbar.team", "  ▸ Team: "))

        status_parts = []
        if running:
            status_parts.append(("class:bottom-toolbar.worker-running", f"{running} running"))
        if done:
            status_parts.append(("class:bottom-toolbar.worker-done", f"{done} done"))
        if failed:
            status_parts.append(("class:bottom-toolbar.worker-failed", f"{failed} failed"))

        for i, (style, text) in enumerate(status_parts):
            if i > 0:
                parts.append(("class:bottom-toolbar.info", " · "))
            parts.append((style, text))

        return parts

    def team_display():
        """Ctrl+T handler — print detailed team status to console."""
        workers = team_manager.status()
        if not workers:
            console.print("\n[#8899aa]  No workers. Use /team <prompt> to spawn one.[/#8899aa]")
            return

        console.print()
        console.print("[bold #88c0d0]  ▸ Team Members[/bold #88c0d0]")
        console.print("[#4c566a]  ─────────────────────────────────────────[/#4c566a]")

        for w in workers:
            if w["status"] == "running":
                icon = "[#ebcb8b]⟳[/#ebcb8b]"
                status_text = "[#ebcb8b]running[/#ebcb8b]"
            elif w["status"] == "completed":
                icon = "[#a3be8c]✓[/#a3be8c]"
                status_text = "[#a3be8c]completed[/#a3be8c]"
            else:
                icon = "[#bf616a]✗[/#bf616a]"
                status_text = "[#bf616a]failed[/#bf616a]"

            console.print(
                f"  {icon} [bold #d8dee9]{_esc(str(w['name']))}[/bold #d8dee9]  "
                f"{status_text}"
            )
            console.print(
                f"    [#8899aa]Task: {_esc(str(w['prompt'])[:70])}[/#8899aa]"
            )
            if w["result"] and w["status"] != "running":
                result_preview = w["result"][:100].replace("\n", " ")
                console.print(
                    f"    [#4c566a]Result: {_esc(result_preview)}[/#4c566a]"
                )

        # Messages
        msgs = team_manager.get_lead_messages()
        if msgs:
            console.print()
            console.print("[bold #5e81ac]  ▸ Messages[/bold #5e81ac]")
            console.print("[#4c566a]  ─────────────────────────────────────────[/#4c566a]")
            for m in msgs:
                console.print(
                    f"  [#5e81ac]\\[{_esc(str(m.from_name))}][/#5e81ac] "
                    f"[#d8dee9]{_esc(str(m.content)[:80])}[/#d8dee9]"
                )

        console.print("[#4c566a]  ─────────────────────────────────────────[/#4c566a]")
        console.print()

    # Auto-display team status during agent execution + Ctrl+T signal handler
    team_monitor = TeamStatusMonitor(
        team_manager.status, console, display_fn=team_display
    )

    def _resume_names_provider() -> list[str]:
        # _list_sessions is defined earlier in this same module, so this
        # reference is already lazy by construction (looked up at call time,
        # not at lambda-creation time) — no separate import needed to dodge
        # a cycle, unlike a provider sourced from another module.
        return [s["name"] for s in _list_sessions()[:_SESSION_DISPLAY_LIMIT]]

    # Argument-aware autocomplete (Phase 3 Task 1): providers are called
    # lazily per keystroke by SlashCommandCompleter, which already swallows
    # exceptions — so a stale config or a session-listing hiccup can only
    # cost a missing suggestion, never a crashed prompt.
    value_providers: dict[str, Callable[[], list[str]]] = {
        "/model": lambda: _model_arg_candidates(config),
        "/providers": lambda: list(config.get("providers", {}).keys()),
        "/setkey": lambda: list(PROVIDER_PRESETS.keys()),
        "/mode": _mode_arg_candidates,
        "/resume": _resume_names_provider,
        "/rewind": lambda: ["undo", "checkpoint", "both", "conversation"],
    }

    # Non-CPR terminal fallback (Phase 3 Task 3): `cpr_state["supported"]`
    # starts at the pre-flight heuristic's guess and is flipped to False by
    # create_session's neutered CPR-warning callback the moment prompt_toolkit
    # itself confirms CPR isn't answered (its own ~2s timeout) — see
    # ui/input.py's terminal_supports_cpr investigation notes for why the
    # pre-flight guess alone can't catch a pty like pexpect's that presents
    # as CPR-capable but never answers.
    statusline_mode = get(config, "ui", "statusline", default="auto")
    cpr_state = {"supported": terminal_supports_cpr()}

    # Input session with history, autocomplete, and status footer
    session = create_session(
        skill_names=skills.names(),
        status_callback=status_callback,
        mode_callback=mode_callback,
        team_callback=team_callback,
        team_display_callback=team_display,
        mode_switch_callback=mode_switch,
        value_providers=value_providers,
        statusline=statusline_mode,
        cpr_state=cpr_state,
    )

    def _inline_status_text() -> str:
        """The one-line "mode + ctx%" fallback printed after each turn when
        the toolbar isn't rendering (ui.statusline: inline, or auto while CPR
        is not positively confirmed) — built from build_status_segments, the
        SAME source the toolbar lines render through (via the toolbar_*
        wrappers), per this task's "one source" requirement."""
        context_pct, context_style = _context_pct_and_style()
        segments = build_status_segments(
            _display_mode(),
            model_name=get(config, "model", "name", default=""),
            provider_name=get(config, "model", "provider", default=""),
            turns=context.turn_count,
            context_pct=context_pct,
            context_style=context_style,
        )
        return format_status_line(segments)

    def _should_show_inline_status() -> bool:
        """Route ui.statusline through the LIVE CPR verdict (review fix).

        The renderer's cpr_support enum is the primary signal: prompt_
        toolkit's 2s CPR timer is a background task of the prompt
        Application, so a sub-2s turn cancels it before it can conclude —
        the neutered-callback latch alone would never flip in a fast
        scripted non-CPR pty (the exact target scenario). Pessimistic rule:
        inline prints while the verdict is NOT_SUPPORTED **or still
        UNKNOWN**; it stops only once SUPPORTED has actually been observed.
        The cpr_state latch (pre-flight heuristic + neutered callback)
        stays authoritative as a secondary signal.
        """
        live = renderer_cpr_support(session)
        if live is False:
            # A definitive NOT_SUPPORTED verdict — keep the shared latch in
            # sync (same latch the neutered warning callback flips).
            cpr_state["supported"] = False
        confirmed = cpr_confirmed_supported(
            live, latched_unsupported=cpr_state["supported"] is False
        )
        _, use_inline = resolve_statusline_mode(statusline_mode, confirmed)
        return use_inline

    # Notification sound config
    notify_enabled = get(config, "ui", "notification_sound", default=True)
    import time as _time

    async def _run_with_notify(coro):
        """Run a coroutine as a cancellable task. Esc or Ctrl+C cancels generation."""
        import signal as _signal

        from spark_code.ui.esc_watcher import EscWatcher

        start = _time.monotonic()
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)

        prev_handler = _signal.getsignal(_signal.SIGINT)
        interrupted = {"v": False}

        def _cancel():
            """Stop generation: signal the agent then cancel the task.

            Shared by the SIGINT handler (Ctrl+C) and the Esc watcher so both
            interrupts follow the exact same path — closing the stream so the
            engine stops generating (the Phase 0 GPU-stop fix). Idempotent.
            """
            interrupted["v"] = True
            agent.cancel()
            task.cancel()
            loop.call_soon_threadsafe(lambda: None)

        def _cancel_on_sigint(sig, frame):
            _cancel()

        _signal.signal(_signal.SIGINT, _cancel_on_sigint)

        # Ensure terminal ISIG is set so Ctrl+C generates SIGINT.
        # prompt_toolkit may leave it cleared after session.prompt()
        # runs in a thread via run_in_executor.
        try:
            import termios as _termios
            _fd = sys.stdin.fileno()
            _attrs = _termios.tcgetattr(_fd)
            if not (_attrs[3] & _termios.ISIG):
                _attrs[3] |= _termios.ISIG
                _termios.tcsetattr(_fd, _termios.TCSANOW, _attrs)
        except Exception:
            pass

        # Watch the SAME fd the ISIG block above mutates (sys.stdin), not a
        # bare 0 — they must agree or the watcher could cbreak a different fd.
        try:
            _esc_fd = sys.stdin.fileno()
        except Exception:
            _esc_fd = 0

        try:
            # Esc watcher starts AFTER the ISIG mutation above and restores
            # termios attrs on __exit__ (innermost scope — before the finally
            # below restores the SIGINT handler). cbreak keeps ISIG on, so
            # Ctrl+C still delivers SIGINT. The watcher thread never touches
            # agent/task directly — it marshals _cancel onto the event loop.
            with EscWatcher(lambda: loop.call_soon_threadsafe(_cancel), fd=_esc_fd):
                result = await task
            # The agent loop handles cancellation gracefully (closes the stream,
            # keeps the partial), so `await task` returns normally instead of
            # raising — surface the interrupt here for Esc AND Ctrl+C alike.
            if interrupted["v"]:
                console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                return ""
            if notify_enabled and (_time.monotonic() - start) > 5.0:
                _notify_done()
            return result
        except asyncio.CancelledError:
            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
            return ""
        finally:
            _signal.signal(_signal.SIGINT, prev_handler)
            # Inline status fallback (Phase 3 Task 3, review fix): printed
            # once after each COMPLETED turn (this finally runs for normal,
            # interrupted and failed generations alike — every _run_with_
            # notify call site is a turn) rather than at the top of the
            # input loop, so blank-line/slash-command iterations don't
            # re-print it. Ordering vs patch_stdout (Task 2) is trivially
            # safe: no prompt is being rendered at this moment — the line
            # lands in scrollback between the turn's output and the next
            # prompt redraw.
            if _should_show_inline_status():
                console.print(f"[dim]{_esc(_inline_status_text())}[/dim]")

    # Handle --continue prompt (send first prompt automatically)
    if continue_prompt:
        try:
            team_monitor.start()
            await _run_with_notify(agent.run(continue_prompt))
        except KeyboardInterrupt:
            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
        except Exception as e:
            console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
        finally:
            team_monitor.stop()

    last_user_message = ""

    try:
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _prompt_with_patched_stdout(session),
                )
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("\n[dim]Use /quit to exit[/dim]")
                continue

            user_input = user_input.strip()
            if not user_input:
                continue

            # Detect dragged-in image files
            img_path, img_remaining = _is_image_drop(user_input)
            if img_path:
                img_prompt = img_remaining.strip() if img_remaining else "Describe this image and help with anything you see."
                mime_type = mimetypes.guess_type(img_path)[0] or "image/png"
                try:
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    size_kb = len(img_data) * 3 / 4 / 1024
                    console.print(f"  [#88c0d0]Image[/#88c0d0] [#d8dee9]{os.path.basename(img_path)}[/#d8dee9] [#4c566a]({size_kb:.0f} KB, {mime_type})[/#4c566a]")
                    # Snapshot BEFORE the turn's user message enters context
                    # (Phase 5 Task 7) — see the /image handler above for why
                    # this can't live inside run_without_user_add() itself.
                    context.snapshot()
                    context.add_user_with_image(img_prompt, img_data, mime_type)
                    try:
                        team_monitor.start()
                        await _run_with_notify(agent.run_without_user_add())
                    except KeyboardInterrupt:
                        console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                    except Exception as e:
                        console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                    finally:
                        team_monitor.stop()
                except OSError as e:
                    console.print(f"[#bf616a]Could not read image: {e}[/#bf616a]")
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                try:
                    result = handle_slash_command(
                        user_input, context, console, config, skills, model,
                        permissions=permissions,
                        team_manager=team_manager,
                        task_store=task_store,
                        memory=memory,
                        stats=session_stats,
                        pinned=pinned,
                        snippets=snippet_lib,
                        plan_state=plan_state,
                    )
                    if result is None:
                        continue
                    if result == "__IMAGE_SENT__":
                        # Image already added to context — run agent without add_user
                        try:
                            team_monitor.start()
                            await _run_with_notify(agent.run_without_user_add())
                        except KeyboardInterrupt:
                            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                        except Exception as e:
                            console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                        finally:
                            team_monitor.stop()
                    elif result.startswith("__COMPACT__"):
                        instructions = result[len("__COMPACT__"):].strip() or None
                        # Ctrl+C-safe (sibling-branch pattern) — see _handle_compact.
                        await _handle_compact(
                            model, context, console, instructions,
                            utility_model=resolve_utility_for(
                                config, "compaction", utility_model))
                    elif result.startswith("__REVIEW__"):
                        from .review import run_review
                        review_target = result[len("__REVIEW__"):].strip()
                        lead_mode = getattr(permissions, "mode", "ask") or "ask"
                        try:
                            await run_review(model, config, console, lead_mode,
                                             target=review_target,
                                             utility_model=utility_model)
                        except KeyboardInterrupt:
                            console.print("\n[#ebcb8b]Review cancelled.[/#ebcb8b]")
                        except Exception as e:
                            console.print(f"\n[#bf616a]Review error: {e}[/#bf616a]")
                    elif result.startswith("__TEAM_SPAWN__"):
                        prompt = result[len("__TEAM_SPAWN__"):]
                        try:
                            await team_manager.spawn(prompt)
                        except Exception as e:
                            console.print(f"\n[#bf616a]Error spawning worker: {e}[/#bf616a]")
                    elif result.startswith("__PLAN_EXECUTE__"):
                        plan_content = result[len("__PLAN_EXECUTE__"):]
                        try:
                            team_monitor.start()
                            await execute_plan(
                                plan_content, team_manager, agent, console
                            )
                        except KeyboardInterrupt:
                            console.print("\n[#ebcb8b]Plan interrupted[/#ebcb8b]")
                        except Exception as e:
                            console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                        finally:
                            team_monitor.stop()
                    elif result.startswith("__RUN_CMD__"):
                        run_command = result[len("__RUN_CMD__"):]
                        run_prompt = (
                            f"Run this command in the current directory:\n\n"
                            f"```\n{run_command}\n```\n\n"
                            f"Use the bash tool. Show the output."
                        )
                        try:
                            team_monitor.start()
                            await _run_with_notify(agent.run(run_prompt))
                        except KeyboardInterrupt:
                            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                        except Exception as e:
                            console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                        finally:
                            team_monitor.stop()
                    elif result == "__INDEX__":
                        index_service_url = resolve_service_url(config)
                        console.print(
                            f"[#88c0d0]▸ Indexing project for code_search "
                            f"({index_service_url})...[/#88c0d0]")
                        try:
                            outcome = await index_project(
                                os.getcwd(), service_url=index_service_url)
                        except Exception as e:
                            console.print(f"[#bf616a]Index error: {e}[/#bf616a]")
                        else:
                            if outcome.get("error"):
                                console.print(f"[#ebcb8b]  ⚠ {outcome['error']}[/#ebcb8b]")
                            else:
                                console.print(
                                    f"[#a3be8c]  ✓ Indexed {outcome['indexed']} file(s), "
                                    f"skipped {outcome['skipped']} — "
                                    f"collection {outcome['collection']}[/#a3be8c]")
                    elif result == "__DOCTOR__":
                        # Read-only: run_doctor never raises per-check (every
                        # check is individually try/excepted — see doctor.py),
                        # but this outer try/except is defense-in-depth so a
                        # genuine bug still can't crash the REPL.
                        console.print("[#88c0d0]▸ Running health checks...[/#88c0d0]")
                        try:
                            checks = await run_doctor(config)
                        except Exception as e:
                            console.print(f"[#bf616a]Doctor error: {e}[/#bf616a]")
                        else:
                            render_doctor(console, checks)
                    elif result.startswith("__TEAM_STOP__"):
                        stop_id = result[len("__TEAM_STOP__"):]
                        try:
                            if stop_id:
                                ok = await team_manager.stop(stop_id)
                                if not ok:
                                    console.print(f"[#ebcb8b]Worker #{stop_id} not found[/#ebcb8b]")
                            else:
                                await team_manager.stop_all()
                                console.print("[#a3be8c]All workers stopped[/#a3be8c]")
                        except Exception as e:
                            console.print(f"\n[#bf616a]Error stopping worker: {e}[/#bf616a]")
                    elif result.startswith("__WATCH__"):
                        watch_cmd = result[len("__WATCH__"):]
                        if watch_cmd.lower() == "off":
                            if file_watcher and file_watcher.is_running:
                                await file_watcher.stop()
                                file_watcher = None
                            else:
                                console.print("[#8899aa]No watcher running.[/#8899aa]")
                        else:
                            if file_watcher and file_watcher.is_running:
                                await file_watcher.stop()
                            file_watcher = FileWatcher(watch_cmd, console)
                            await file_watcher.start()

                    elif result.startswith("__LOOP__"):
                        from .loop import LoopEngine, parse_loop_args
                        from .ui.esc_watcher import EscWatcher
                        loop_arg = result[len("__LOOP__"):]
                        if loop_arg == "stop":
                            console.print("[#8899aa]No loop is running (a running loop owns the prompt — stop it with Esc).[/#8899aa]")
                        elif loop_arg == "status":
                            if last_loop_engine is not None:
                                console.print(f"[#88c0d0]{last_loop_engine.status_line()}[/#88c0d0]")
                                console.print(f"[#8899aa]log: {last_loop_engine.state.log_path}[/#8899aa]")
                            else:
                                console.print("[#8899aa]No loop has run this session.[/#8899aa]")
                        else:
                            try:
                                loop_cfg = parse_loop_args(loop_arg)
                            except ValueError as e:
                                console.print(f"[#bf616a]{e}[/#bf616a]")
                                continue
                            engine = LoopEngine(loop_cfg, console, runner=agent.run)
                            last_loop_engine = engine
                            # /yolo recipe for the loop's duration: agentic prompt
                            # + trusted tools, restored in finally NO MATTER WHAT.
                            prev_prompt = context.system_prompt
                            prev_mode = permissions.mode if permissions else None
                            if SYSTEM_PROMPT in context.system_prompt:
                                context.system_prompt = context.system_prompt.replace(
                                    SYSTEM_PROMPT, AGENTIC_PROMPT)
                            if permissions:
                                permissions.mode = "trust"
                            console.print("[bold #ebcb8b]🔁 Loop started[/bold #ebcb8b] [#8899aa]— Esc to stop (ends the current round); checkpoint before every round[/#8899aa]")
                            try:
                                with EscWatcher(on_escape=lambda: _loop_escape_action(engine, agent)):
                                    final_status = await engine.run()
                            except KeyboardInterrupt:
                                engine.request_stop()
                                final_status = "stopped"
                            except Exception as e:
                                console.print(f"[#bf616a]Loop error: {e}[/#bf616a]")
                                final_status = engine.state.status
                            finally:
                                context.system_prompt = prev_prompt
                                if permissions and prev_mode is not None:
                                    permissions.mode = prev_mode
                            color = "#a3be8c" if final_status == "done" else "#ebcb8b"
                            console.print(f"[{color}]Loop ended: {final_status} after {engine.state.round_n} round(s)[/{color}]")
                            console.print(f"[#8899aa]log: {engine.state.log_path} · /rollback lists round checkpoints[/#8899aa]")

                    elif result == "__PROFILE__":
                        # Model performance benchmark
                        import time as _t
                        console.print("[#88c0d0]Running benchmark...[/#88c0d0]")
                        bench_prompt = "Write a Python function that checks if a number is prime. Include type hints."
                        bench_ctx = Context(system_prompt="You are a coding assistant.", max_tokens=4096)
                        bench_ctx.add_user(bench_prompt)
                        start_t = _t.monotonic()
                        first_token_t = None
                        total_chars = 0
                        try:
                            async for chunk in model.chat(bench_ctx.get_messages(), tools=None, stream=True):
                                if chunk["type"] == "text":
                                    if first_token_t is None:
                                        first_token_t = _t.monotonic()
                                    total_chars += len(chunk["content"])
                        except Exception as e:
                            console.print(f"  [#bf616a]Benchmark failed: {_esc(str(e))}[/#bf616a]")
                            continue
                        end_t = _t.monotonic()
                        ttft = (first_token_t - start_t) if first_token_t else 0
                        total_time = end_t - start_t
                        est_tokens = total_chars / 4
                        tps = est_tokens / total_time if total_time > 0 else 0
                        console.print(f"  [#88c0d0]TTFT:[/#88c0d0] [#d8dee9]{ttft:.2f}s[/#d8dee9]")
                        console.print(f"  [#88c0d0]Total time:[/#88c0d0] [#d8dee9]{total_time:.2f}s[/#d8dee9]")
                        console.print(f"  [#88c0d0]Output:[/#88c0d0] [#d8dee9]~{int(est_tokens)} tokens ({total_chars} chars)[/#d8dee9]")
                        console.print(f"  [#88c0d0]Speed:[/#88c0d0] [#a3be8c]{tps:.1f} tokens/sec[/#a3be8c]")
                        console.print(f"  [#88c0d0]Model:[/#88c0d0] [#d8dee9]{_esc(str(get(config, 'model', 'display_name', default='') or get(config, 'model', 'name')))}[/#d8dee9]")

                    elif result == "__BENCHMARK__":
                        import time as _btime
                        model_name = get(config, "model", "name", default="unknown")
                        console.print(f"  [#88c0d0]Benchmarking {model_name}...[/#88c0d0]")
                        bench_prompt = "Write a Python function that checks if a number is prime."
                        bench_msgs = [{"role": "user", "content": bench_prompt}]
                        first_token_time = None
                        token_count = 0
                        start = _btime.monotonic()
                        try:
                            async for chunk in model.chat(
                                messages=bench_msgs, tools=[], stream=True
                            ):
                                if chunk["type"] == "text":
                                    if first_token_time is None:
                                        first_token_time = _btime.monotonic()
                                    token_count += max(1, len(chunk["content"].split()))
                                elif chunk["type"] == "done":
                                    break
                            end = _btime.monotonic()
                            ttft = (first_token_time - start) if first_token_time else 0
                            total_time = end - start
                            gen_time = (end - first_token_time) if first_token_time else 0
                            speed = token_count / gen_time if gen_time > 0 else 0
                            console.print(f"  [#a3be8c]Time to first token: {ttft:.1f}s[/#a3be8c]")
                            console.print(f"  [#a3be8c]Generation speed: {speed:.1f} tok/s[/#a3be8c]")
                            console.print(f"  [#a3be8c]Total: {token_count} tokens in {total_time:.1f}s[/#a3be8c]")
                        except (KeyboardInterrupt, asyncio.CancelledError):
                            console.print("\n  [#ebcb8b]Benchmark cancelled.[/#ebcb8b]")
                        except Exception as e:
                            console.print(f"  [#bf616a]Benchmark failed: {e}[/#bf616a]")
                        continue

                    elif result.startswith("__TEACH__"):
                        teach_args = result[len("__TEACH__"):]
                        # Parse: name "description" -- command
                        if " -- " in teach_args:
                            before, command_str = teach_args.split(" -- ", 1)
                            parts = before.strip().split(maxsplit=1)
                            tool_name = parts[0] if parts else ""
                            tool_desc = parts[1].strip('"\'') if len(parts) > 1 else f"Custom tool: {tool_name}"
                        else:
                            parts = teach_args.strip().split(maxsplit=2)
                            tool_name = parts[0] if parts else ""
                            tool_desc = parts[1].strip('"\'') if len(parts) > 1 else f"Custom tool: {tool_name}"
                            command_str = parts[2] if len(parts) > 2 else ""

                        if not tool_name or not command_str:
                            console.print("[#ebcb8b]Usage: /teach <name> <description> -- <command>[/#ebcb8b]")
                        else:
                            try:
                                ct = custom_tool_registry.add(tool_name, tool_desc, command_str)
                            except ValueError as e:
                                # e.g. name collides with a built-in tool
                                console.print(f"[#bf616a]Can't teach '{_esc(tool_name)}': {_esc(str(e))}[/#bf616a]")
                            else:
                                tools.register(ct)
                                console.print(f"[#a3be8c]Taught: {_esc(tool_name)} — {_esc(tool_desc)}[/#a3be8c]")
                                console.print(f"[#8899aa]Command: {_esc(command_str)}[/#8899aa]")

                    elif result.startswith("__BRANCH__"):
                        branch_args = result[len("__BRANCH__"):]
                        if not branch_args:
                            # List branches
                            branches = branch_manager.list_branches()
                            if not branches:
                                console.print("[#8899aa]No branches yet. Use /branch <name>[/#8899aa]")
                            else:
                                console.print("[bold #eceff4]Branches:[/bold #eceff4]")
                                for b in branches:
                                    marker = " [#a3be8c]<- current[/#a3be8c]" if b["current"] else ""
                                    console.print(
                                        f"  [#88c0d0]{b['name']}[/#88c0d0]  "
                                        f"[#8899aa]{b['turns']} turns[/#8899aa]{marker}")
                        else:
                            try:
                                msg = branch_manager.create_branch(branch_args, context, os.getcwd())
                            except ValueError as e:
                                console.print(f"[#bf616a]{_esc(str(e))}[/#bf616a]")
                            else:
                                console.print(f"[#a3be8c]{msg}[/#a3be8c]")

                    elif result.startswith("__SWITCH__"):
                        branch_name = result[len("__SWITCH__"):]
                        # Save current branch first
                        branch_manager.save_branch(branch_manager.current, context, os.getcwd())
                        ok, msg = branch_manager.switch_branch(branch_name, context)
                        style = "#a3be8c" if ok else "#bf616a"
                        console.print(f"[{style}]{msg}[/{style}]")

                    elif result == "__BRANCHES__":
                        branches = branch_manager.list_branches()
                        if not branches:
                            console.print("[#8899aa]No branches. Use /branch <name> to create one.[/#8899aa]")
                        else:
                            console.print("[bold #eceff4]Conversation branches:[/bold #eceff4]")
                            for b in branches:
                                marker = " [#a3be8c]<- current[/#a3be8c]" if b["current"] else ""
                                parent = f" [#4c566a](from {b['parent']})[/#4c566a]" if b.get("parent") else ""
                                console.print(
                                    f"  [#88c0d0]{b['name']}[/#88c0d0]  "
                                    f"[#8899aa]{b['turns']} turns[/#8899aa]"
                                    f"{parent}{marker}")
                            console.print("[#8899aa]Use /switch <name> to switch branches[/#8899aa]")

                    elif result.startswith("__SHARE__"):
                        share_format = result[len("__SHARE__"):].strip() or "html"
                        lines = ["<!DOCTYPE html><html><head>",
                                 "<meta charset='utf-8'>",
                                 "<title>Spark Code Session</title>",
                                 "<style>",
                                 "body{font-family:monospace;background:#2e3440;color:#d8dee9;max-width:800px;margin:0 auto;padding:20px}",
                                 ".user{color:#88c0d0;font-weight:bold}.assistant{color:#d8dee9}",
                                 ".tool{color:#a3be8c;font-style:italic}pre{background:#3b4252;padding:10px;border-radius:4px;overflow-x:auto}",
                                 "h1{color:#ebcb8b}h2{color:#88c0d0;border-bottom:1px solid #4c566a}",
                                 "</style></head><body>",
                                 "<h1>Spark Code Session</h1>"]
                        import html as _html
                        for msg in context.messages:
                            role = msg.get("role", "")
                            content = msg.get("content", "") or ""
                            if role == "user" and isinstance(content, str):
                                lines.append(f"<h2 class='user'>User</h2><p>{_html.escape(content[:2000])}</p>")
                            elif role == "assistant" and isinstance(content, str) and content:
                                lines.append(f"<h2 class='assistant'>Assistant</h2><p>{_html.escape(content[:2000])}</p>")
                            elif role == "tool":
                                name = msg.get("name", "tool")
                                preview = (content[:500] + "...") if len(content) > 500 else content
                                lines.append(f"<p class='tool'>Tool: {_html.escape(str(name))}</p><pre>{_html.escape(preview)}</pre>")
                        lines.append("</body></html>")
                        html_content = "\n".join(lines)

                        if share_format == "gist":
                            # Export as markdown for gist
                            md_lines = ["# Spark Code Session\n"]
                            for msg in context.messages:
                                role = msg.get("role", "")
                                content = msg.get("content", "") or ""
                                if role == "user" and isinstance(content, str):
                                    md_lines.append(f"## User\n\n{content[:2000]}\n")
                                elif role == "assistant" and isinstance(content, str):
                                    md_lines.append(f"## Assistant\n\n{content[:2000]}\n")
                            md_text = "\n".join(md_lines)
                            share_path = os.path.join(os.getcwd(), "session_share.md")
                            with open(share_path, "w") as f:
                                f.write(md_text)
                            console.print(f"[#a3be8c]Saved to {share_path}[/#a3be8c]")
                            console.print("[#8899aa]Create a gist: gh gist create session_share.md[/#8899aa]")
                        else:
                            share_path = os.path.join(os.getcwd(), "session_share.html")
                            with open(share_path, "w") as f:
                                f.write(html_content)
                            console.print(f"[#a3be8c]Saved to {share_path}[/#a3be8c]")

                    elif result.startswith("__SEARCH__"):
                        query = result[len("__SEARCH__"):].lower()
                        history_dir = _sessions_dir()
                        if not os.path.isdir(history_dir):
                            console.print("[#8899aa]No saved sessions.[/#8899aa]")
                        else:
                            import json as _json
                            matches = []
                            for f in sorted(os.listdir(history_dir), reverse=True):
                                if not f.endswith(".json"):
                                    continue
                                path = os.path.join(history_dir, f)
                                try:
                                    with open(path, encoding="utf-8") as fh:
                                        data = _json.load(fh)
                                    for msg in data.get("messages", []):
                                        content = msg.get("content", "")
                                        if isinstance(content, str) and query in content.lower():
                                            label = data.get("label", f.replace(".json", ""))
                                            # Extract matching snippet
                                            idx = content.lower().index(query)
                                            start = max(0, idx - 40)
                                            end = min(len(content), idx + len(query) + 40)
                                            snippet = content[start:end].replace("\n", " ")
                                            matches.append((label, snippet, f))
                                            break
                                except Exception:
                                    pass
                                if len(matches) >= 10:
                                    break

                            if not matches:
                                console.print(f"[#8899aa]No sessions matching '{_esc(query)}'[/#8899aa]")
                            else:
                                console.print(f"[bold #eceff4]Sessions matching '{_esc(query)}':[/bold #eceff4]")
                                for label, snippet, fname in matches:
                                    console.print(f"  [#88c0d0]{_esc(str(label))}[/#88c0d0]")
                                    console.print(f"    [#8899aa]...{_esc(str(snippet))}...[/#8899aa]")
                                console.print("[#8899aa]Use /history <name> to resume[/#8899aa]")

                    elif result == "__ANALYTICS__":
                        stats = session_stats
                        if not stats:
                            console.print("[#8899aa]No stats available.[/#8899aa]")
                        else:
                            # Generate analytics report
                            console.print()
                            console.print("[bold #eceff4]Session Analytics[/bold #eceff4]")
                            console.print("[#4c566a]─────────────────────────────────[/#4c566a]")
                            console.print(f"  [#88c0d0]Duration:[/#88c0d0] {stats.format_duration()}")
                            console.print(f"  [#88c0d0]Turns:[/#88c0d0] {context.turn_count}")
                            console.print(f"  [#88c0d0]Total tool calls:[/#88c0d0] {stats.total_tool_calls}")
                            console.print()

                            # Tool usage breakdown
                            if stats.tool_calls:
                                console.print("  [bold #eceff4]Tool Usage[/bold #eceff4]")
                                max_count = max(stats.tool_calls.values())
                                for name, count in sorted(stats.tool_calls.items(), key=lambda x: -x[1]):
                                    bar_len = int(count / max_count * 20) if max_count else 0
                                    bar = "[#a3be8c]" + "█" * bar_len + "[/#a3be8c]" + "[#4c566a]" + "░" * (20 - bar_len) + "[/#4c566a]"
                                    console.print(f"    {name:<15} {bar} {count}")

                            # Files touched
                            all_files = stats.files_read | stats.files_written | stats.files_edited
                            if all_files:
                                console.print()
                                console.print("  [bold #eceff4]Files Touched[/bold #eceff4]")
                                home = os.path.expanduser("~")
                                for fpath in sorted(all_files)[:15]:
                                    display = "~" + fpath[len(home):] if fpath.startswith(home) else fpath
                                    tags = []
                                    if fpath in stats.files_read:
                                        tags.append("[#88c0d0]R[/#88c0d0]")
                                    if fpath in stats.files_written:
                                        tags.append("[#a3be8c]W[/#a3be8c]")
                                    if fpath in stats.files_edited:
                                        tags.append("[#ebcb8b]E[/#ebcb8b]")
                                    console.print(f"    {''.join(tags)} {display}")

                            # Token usage
                            console.print()
                            console.print("  [bold #eceff4]Tokens[/bold #eceff4]")
                            console.print(f"    Input:  {model.total_input_tokens:,}")
                            console.print(f"    Output: {model.total_output_tokens:,}")
                            cost = model.estimated_cost
                            if cost > 0:
                                console.print(f"    Cost:   ${cost:.4f}")

                            # Cache stats
                            if tool_cache:
                                cs = tool_cache.stats
                                console.print()
                                console.print("  [bold #eceff4]Cache[/bold #eceff4]")
                                console.print(f"    Entries: {cs['entries']}  Hits: {cs['hits']}  Misses: {cs['misses']}  Rate: {cs['hit_rate']}")

                    elif result == "__RETRY__":
                        if not last_user_message:
                            console.print("  [#ebcb8b]No previous message to retry.[/#ebcb8b]")
                        else:
                            console.print(f"  [#88c0d0]Retrying: {_esc(last_user_message[:60])}...[/#88c0d0]")
                            try:
                                team_monitor.start()
                                await _run_with_notify(agent.run(last_user_message))
                            except KeyboardInterrupt:
                                console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                            except Exception as e:
                                console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                            finally:
                                team_monitor.stop()
                        continue

                    elif result == "__CONTINUE__":
                        from spark_code.agent import CHECKPOINT_DIR, load_checkpoint
                        checkpoint_path = str(CHECKPOINT_DIR / "latest.json")
                        data = load_checkpoint(checkpoint_path)
                        if not data:
                            console.print("  [#ebcb8b]No checkpoint found.[/#ebcb8b]")
                        else:
                            saved_cwd = data.get("cwd", "")
                            if saved_cwd and saved_cwd != os.getcwd():
                                console.print(f"  [#ebcb8b]Note: CWD changed. Checkpoint was in {saved_cwd}[/#ebcb8b]")
                            _restore_checkpoint(context, data)
                            console.print(f"  [#a3be8c]Checkpoint restored ({len(data['messages'])} messages). Resuming...[/#a3be8c]")
                            try:
                                team_monitor.start()
                                await _run_with_notify(agent.run_without_user_add())
                            except KeyboardInterrupt:
                                console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                            except Exception as e:
                                console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                            finally:
                                team_monitor.stop()

                    elif result == "__CLEAN__":
                        if not session_stats or not session_stats.files_created:
                            console.print("  [#ebcb8b]No files were created this session.[/#ebcb8b]")
                        else:
                            existing = []
                            for f in sorted(session_stats.files_created):
                                if os.path.exists(f):
                                    try:
                                        lines = len(open(f).readlines())
                                    except Exception:
                                        lines = 0
                                    existing.append((f, lines))
                            if not existing:
                                console.print("  [#ebcb8b]All created files have already been deleted.[/#ebcb8b]")
                            else:
                                console.print("  [#88c0d0]Files created this session:[/#88c0d0]")
                                for path, lines in existing:
                                    short = path.replace(os.getcwd() + "/", "")
                                    console.print(f"    {short} ({lines} lines)")
                                console.print()
                                try:
                                    answer = await session.prompt_async("  Delete all? [y/N/select] ")
                                except (KeyboardInterrupt, EOFError):
                                    console.print("  Cancelled.")
                                    continue
                                answer = answer.strip().lower()
                                if answer == "y":
                                    for path, _ in existing:
                                        try:
                                            os.remove(path)
                                            console.print(f"  [#a3be8c]Deleted {os.path.basename(path)}[/#a3be8c]")
                                        except Exception as e:
                                            console.print(f"  [#bf616a]Error deleting {path}: {e}[/#bf616a]")
                                elif answer == "select":
                                    for path, lines in existing:
                                        short = path.replace(os.getcwd() + "/", "")
                                        try:
                                            ans = await session.prompt_async(f"  Delete {short}? [y/N] ")
                                        except (KeyboardInterrupt, EOFError):
                                            console.print("  Cancelled.")
                                            break
                                        if ans.strip().lower() == "y":
                                            try:
                                                os.remove(path)
                                                console.print(f"  [#a3be8c]Deleted {os.path.basename(path)}[/#a3be8c]")
                                            except Exception as e:
                                                console.print(f"  [#bf616a]Error: {e}[/#bf616a]")
                                else:
                                    console.print("  Cancelled.")
                        continue

                    elif result.startswith("__MODEL_SWITCH__"):
                        provider_name = result[len("__MODEL_SWITCH__"):]
                        providers = config.get("providers", {})
                        if provider_name not in providers:
                            # Check built-in providers
                            if provider_name in PROVIDERS:
                                console.print(f"[#ebcb8b]Provider '{provider_name}' is built-in but not configured in your config.yaml[/#ebcb8b]")
                                console.print("[#8899aa]Add it to your config.yaml under 'providers:' to use it[/#8899aa]")
                            else:
                                console.print(f"[#bf616a]Unknown provider: {provider_name}[/#bf616a]")
                                if providers:
                                    console.print(f"[#8899aa]Available: {', '.join(providers.keys())}[/#8899aa]")
                            continue

                        pconf = providers[provider_name]
                        # Phase 5 Task 2: a provider added via /setkey has no
                        # api_key in its config block at all (the key lives in
                        # ~/.spark/keys) — resolve_provider_key applies the same
                        # precedence config.resolve_provider does at startup
                        # (explicit ${ENV}-resolved config key > keys file > "").
                        resolved_key = resolve_provider_key(provider_name, pconf, load_keys())
                        # Resolve the underlying model name for the new endpoint.
                        new_real = await resolve_real_model_name(
                            endpoint=pconf.get("endpoint", ""),
                            api_key=resolved_key,
                            timeout=1.5,
                            configured_model=pconf.get("model", ""),
                        )
                        # Close old client
                        await model.close()
                        # Create new client
                        model = ModelClient(
                            endpoint=pconf.get("endpoint", "http://localhost:11434"),
                            model=pconf.get("model", "unknown"),
                            temperature=pconf.get("temperature", 0.7),
                            max_tokens=pconf.get("max_tokens", 8192),
                            api_key=resolved_key,
                            provider=provider_name,
                            timeout=float(pconf.get("timeout", 300)),
                            real_model_name=new_real or "",
                            supports_vision=bool(pconf.get("vision", False)),
                        )
                        # Update config
                        config["model"]["name"] = pconf.get("model", "unknown")
                        config["model"]["endpoint"] = pconf.get("endpoint", "http://localhost:11434")
                        config["model"]["provider"] = provider_name
                        config["model"]["vision"] = bool(pconf.get("vision", False))
                        config["model"]["api_key"] = resolved_key
                        # display_name follows the NEW provider: adopt its configured
                        # display_name if it has one, otherwise DROP the stale key so
                        # the toolbar/banner show this provider's real model name
                        # instead of the previous provider's friendly alias (e.g. the
                        # qwen3.5:122b alias lingering after a switch).
                        new_display = pconf.get("display_name")
                        if new_display:
                            config["model"]["display_name"] = new_display
                        else:
                            config["model"].pop("display_name", None)

                        # Update every holder of the PRIMARY model client — the
                        # agent AND the team manager (which kept the now-closed
                        # client, so /team after a switch would hit a closed
                        # httpx client). Deliberately does NOT touch
                        # agent.utility_model/worker_model_obj — Task 2's utility
                        # client (like the pre-existing worker model) has its own
                        # lifecycle, independent of the primary model switching
                        # providers.
                        agent.model = model
                        if team_manager is not None:
                            team_manager.model = model
                        console.print(f"[#a3be8c]Switched to {provider_name} ({pconf.get('model', '?')})[/#a3be8c]")
                        console.print("[#8899aa]Conversation context preserved[/#8899aa]")
                    else:
                        # Skill returned a prompt — send to agent
                        is_projectplan = "projectplan.md" in result
                        user_input = result
                        try:
                            team_monitor.start()
                            await _run_with_notify(agent.run(user_input))
                        except KeyboardInterrupt:
                            console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                        except Exception as e:
                            console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                        finally:
                            team_monitor.stop()
                        # Post-run status for /projectplan
                        if is_projectplan:
                            pp_path = os.path.join(os.getcwd(), "projectplan.md")
                            if os.path.exists(pp_path):
                                console.print("\n[#a3be8c]▸ projectplan.md created successfully[/#a3be8c]")
                                console.print("[#88c0d0]  /projectplan show  to review  ·  /projectplan go  to execute[/#88c0d0]")
                            else:
                                console.print("\n[#bf616a]▸ projectplan.md was NOT created — model may have failed[/#bf616a]")
                                console.print("[#8899aa]  Try again or use /projectplan <prompt> with a simpler request[/#8899aa]")
                except Exception as _cmd_exc:
                    # Any uncaught error in a slash handler or its sentinel
                    # dispatch is contained here so one bad command can't crash
                    # the whole REPL. KeyboardInterrupt / SystemExit (e.g. /exit)
                    # are BaseException, not Exception, so they still propagate.
                    console.print(f"\n[#bf616a]Command error: {_esc(str(_cmd_exc))}[/#bf616a]")
                    continue
            elif _is_shell_command(user_input):
                # Direct shell command — run via agent with explicit instruction
                run_prompt = (
                    f"Run this command now with the bash tool:\n\n"
                    f"```\n{user_input}\n```\n\n"
                    f"Just run it and show the output. Do not ask questions."
                )
                try:
                    team_monitor.start()
                    await _run_with_notify(agent.run(run_prompt))
                except KeyboardInterrupt:
                    console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                except Exception as e:
                    console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                finally:
                    team_monitor.stop()
            else:
                # Refresh pinned files before each turn and rewrite the single
                # pinned region in the system prompt (idempotent — no duplicate/
                # stale blocks). Runs when there's a region to update or clear.
                if pinned.count > 0:
                    pinned.refresh()
                if pinned.count > 0 or PinnedFiles.PIN_START in context.system_prompt:
                    context.system_prompt = pinned.apply_to_prompt(context.system_prompt)

                last_user_message = user_input

                # Auto-detect error pastes and enhance the prompt
                if _is_error_paste(user_input):
                    user_input = (
                        f"The user pasted an error. Diagnose it and suggest a fix.\n\n"
                        f"```\n{user_input}\n```"
                    )

                # Auto-read files (and list directories) mentioned in the prompt
                mentioned = _detect_file_mentions(user_input)
                if mentioned:
                    extra_context = await _auto_read_context(mentioned[:3], console)
                    if extra_context:
                        user_input += "\n\n" + extra_context

                # Plan mode is now ENFORCED at tool-execution time (writes are
                # blocked, exit_plan_mode is the approval gate) plus a transient
                # per-request system nudge — no prompt-wrapping of the user's
                # message. See spark_code.plan_mode / Agent._plan_denies.
                # Run agent
                try:
                    team_monitor.start()
                    await _run_with_notify(agent.run(user_input))
                except KeyboardInterrupt:
                    console.print("\n[#ebcb8b]Interrupted[/#ebcb8b]")
                except Exception as e:
                    console.print(f"\n[#bf616a]Error: {e}[/#bf616a]")
                finally:
                    team_monitor.stop()


            # Context limit warning
            tokens = context.estimate_tokens()
            if context.max_tokens > 0:
                pct = tokens / context.max_tokens * 100
                if pct > 75 and pct <= 80:
                    console.print(f"[#ebcb8b]Context usage: {pct:.0f}% — consider /compact[/#ebcb8b]")

            # Auto-compact if getting large — routed through the agent's guarded
            # path (real-usage trigger, unknown-window guard, anti-thrash gain +
            # cooldown guards, model summary with mechanical fallback) so this
            # site can never disagree with the in-loop round-boundary trigger.
            # The old inline check here also fired on max_tokens=0 (unknown
            # window) — the shared path fixes that edge too.
            try:
                compact_msg = await agent._maybe_compact()
            except KeyboardInterrupt:
                console.print("\n[#ebcb8b]Compaction cancelled.[/#ebcb8b]")
                compact_msg = None
            if compact_msg:
                console.print(f"  [#ebcb8b]⚡ {compact_msg}[/#ebcb8b]")

            # Auto-review (Task 7, OFF by default): when review.auto is enabled
            # AND the turn just landed a successful write/edit (agent._verify_dirty
            # — the same signal the verification nudge uses), run the review swarm
            # scoped to the TURN's written files (agent._verify_written_paths) so
            # pre-existing dirty files aren't swept in. Both are consumed so a
            # later non-agent iteration (e.g. /help) can't re-trigger on the
            # same diff.
            if (getattr(agent, "_verify_dirty", False)
                    and get(config, "review", "auto", default=False)):
                from .review import maybe_auto_review
                lead_mode = getattr(permissions, "mode", "ask") or "ask"
                wrote = agent._verify_dirty
                changed = list(getattr(agent, "_verify_written_paths", []) or [])
                agent._verify_dirty = False  # consume
                agent._verify_written_paths = []
                try:
                    await maybe_auto_review(model, config, console, lead_mode,
                                            wrote_this_turn=wrote,
                                            changed_files=changed,
                                            utility_model=utility_model)
                except KeyboardInterrupt:
                    console.print("\n[#ebcb8b]Auto-review cancelled.[/#ebcb8b]")
                except Exception as e:
                    console.print(f"[#8899aa]Auto-review skipped: {e}[/#8899aa]")

    finally:
        # Auto-save conversation history with label and cwd
        if context.turn_count > 0:
            try:
                from datetime import datetime
                history_dir = _sessions_dir()
                os.makedirs(history_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                label = _make_session_label(context)
                suffix = f"_{label}" if label else ""
                context.save(
                    os.path.join(history_dir, f"{ts}{suffix}.json"),
                    label=label,
                    cwd=os.getcwd(),
                )
            except Exception:
                pass  # Don't crash on save failure

        # Session -> training corpus export (Phase 4 Task 6, opt-in,
        # default OFF — see _export_corpus_session/corpus.py for the full
        # privacy rationale). Gated the same way as the history auto-save
        # above (turn_count > 0 — nothing happened, nothing to export).
        # agent._last_stream_error is reset at the top of every Agent.run()
        # call, so at this teardown point it reflects only the LAST
        # completed turn — the closest thing to a whole-session "did this
        # end cleanly" signal a REPL loop has (there's no single
        # whole-session error state to check otherwise).
        if context.turn_count > 0:
            _export_corpus_session(
                config, context, agent,
                error=getattr(agent, "_last_stream_error", None),
            )

        # Stop the file watcher so its poll task is cancelled cleanly.
        if file_watcher is not None and getattr(file_watcher, "is_running", False):
            try:
                await file_watcher.stop()
            except Exception:
                pass
        await team_manager.stop_all()
        await model.close()
        # Own lifecycle (Phase 4 Task 2): never touched by a primary /model
        # switch, closed only here at real session end — same treatment as
        # worker_model_obj would ideally get (pre-existing gap, not this
        # task's to fix), but explicit for the client this task introduces.
        if utility_model is not None:
            try:
                await utility_model.close()
            except Exception:
                pass
        await mcp_client.disconnect_all()
        # Remove any diff-in-editor temp files (Task 5) created this
        # session. Also atexit-registered in spark_code.editor as a
        # backstop (e.g. one-shot mode, or an exit path that skips this
        # finally) — calling it here just means interactive sessions don't
        # wait on process exit for cleanup.
        from .editor import cleanup_diff_temp_dirs
        cleanup_diff_temp_dirs()
        console.print("[dim]Session ended.[/dim]")


def _run_setup():
    """Interactive setup wizard for first-time configuration."""
    import yaml as _yaml

    from .config import GLOBAL_CONFIG_DIR, GLOBAL_CONFIG_FILE

    console = Console()
    console.print()
    console.print("[bold #ebcb8b]  Spark Code Setup[/bold #ebcb8b]")
    console.print("[#4c566a]  ─────────────────────────────────────────[/#4c566a]")
    console.print()

    # Provider presets — built from shared _PROVIDER_INFO with setup-specific extras
    _SETUP_EXTRAS = {
        "gemini": {
            "label": "Google Gemini (recommended — fast, free tier)",
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
            "context_window": 1000000,
        },
        "openai": {
            "label": "OpenAI (GPT-4o, GPT-4o-mini)",
            "endpoint": "https://api.openai.com/v1",
            "context_window": 128000,
        },
        "groq": {
            "label": "Groq (fast Llama 3.3 inference, free tier)",
            "endpoint": "https://api.groq.com/openai/v1",
            "context_window": 128000,
        },
        "deepseek": {
            "label": "DeepSeek (cheap, strong at code)",
            "endpoint": "https://api.deepseek.com/v1",
            "context_window": 64000,
        },
        "openrouter": {
            "label": "OpenRouter (100+ models, one API key)",
            "endpoint": "https://openrouter.ai/api/v1",
            "context_window": 200000,
        },
        "ollama": {
            "label": "Ollama (local, no API key needed)",
            "endpoint": "http://localhost:11434",
            "context_window": 32768,
        },
        "sglang": {
            "label": "SGLang (fast local inference server)",
            "endpoint": "http://localhost:30000",
            "context_window": 32768,
        },
    }
    presets = {}
    for i, p in enumerate(_PROVIDER_INFO, 1):
        # Fall back to the provider's own fields if no setup-specific extras
        # exist, so a provider added to _PROVIDER_INFO can't crash setup.
        presets[str(i)] = {**p, **_SETUP_EXTRAS.get(p["name"], {})}

    console.print("  Choose a provider:\n")
    for key, preset in presets.items():
        console.print(f"    [#88c0d0]{key}[/#88c0d0]  {preset['label']}")
    console.print()

    choice = input(f"  Enter number (1-{len(presets)}): ").strip()
    if choice not in presets:
        console.print("[#bf616a]  Invalid choice. Run 'spark --setup' again.[/#bf616a]")
        return

    preset = presets[choice]
    console.print(f"\n  [#a3be8c]Selected: {preset['label']}[/#a3be8c]\n")

    # Get API key
    api_key_value = ""
    env_var = preset["env_var"]
    if env_var:
        console.print(f"  Get your API key at: [#5e81ac]{preset['signup']}[/#5e81ac]\n")
        api_key_value = input(f"  Paste your {env_var}: ").strip()
        if not api_key_value:
            console.print("[#ebcb8b]  No key entered. You can set it later in ~/.zshrc[/#ebcb8b]")
    else:
        console.print("  [#8899aa]No API key needed for local Ollama.[/#8899aa]")

    # Optional: custom model
    console.print(f"\n  Default model: [#88c0d0]{preset['model']}[/#88c0d0]")
    custom_model = input("  Custom model (or Enter to keep default): ").strip()
    if custom_model:
        preset["model"] = custom_model

    # Build config
    GLOBAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing config if any
    existing = {}
    if GLOBAL_CONFIG_FILE.exists():
        with open(GLOBAL_CONFIG_FILE) as f:
            existing = _yaml.safe_load(f) or {}

    # Set up provider
    if "providers" not in existing:
        existing["providers"] = {}

    provider_conf = {
        "endpoint": preset["endpoint"],
        "model": preset["model"],
        "context_window": preset["context_window"],
        "max_tokens": 8192,
        "temperature": 0.7,
    }
    if env_var:
        provider_conf["api_key"] = f"${{{env_var}}}"

    existing["providers"][preset["name"]] = provider_conf
    existing["active_provider"] = preset["name"]

    # Write config
    with open(GLOBAL_CONFIG_FILE, "w") as f:
        _yaml.dump(existing, f, default_flow_style=False)

    console.print(f"\n  [#a3be8c]✓ Config saved to {GLOBAL_CONFIG_FILE}[/#a3be8c]")

    # Set env var
    if env_var and api_key_value:
        shell_rc = os.path.expanduser("~/.zshrc")
        export_line = f'export {env_var}="{api_key_value}"'
        # Check if already set
        try:
            with open(shell_rc, encoding="utf-8") as f:
                rc_content = f.read()
            if env_var not in rc_content:
                with open(shell_rc, "a", encoding="utf-8") as f:
                    f.write(f"\n# Spark Code — {preset['name']}\n{export_line}\n")
                console.print(f"  [#a3be8c]✓ {env_var} added to {shell_rc}[/#a3be8c]")
            else:
                console.print(f"  [#8899aa]{env_var} already in {shell_rc}[/#8899aa]")
        except OSError:
            console.print(f"  [#ebcb8b]Add this to your {shell_rc}:[/#ebcb8b]")
            console.print(f"    {export_line}")

        # Set for current session too
        os.environ[env_var] = api_key_value

    console.print("\n  [bold #a3be8c]Setup complete![/bold #a3be8c]")
    console.print("  Run [bold]spark[/bold] to start coding.\n")
    if env_var and api_key_value:
        console.print("  [#8899aa]Run 'source ~/.zshrc' or open a new terminal first.[/#8899aa]\n")


@click.command()
@click.option("--endpoint", "-e", help="Model API endpoint URL")
@click.option("--model", "-m", "model_name", help="Model name")
@click.option("--provider", help="Provider: ollama, gemini, openai")
@click.option("--prompt", "-p", "prompt_flag", default="",
              help="One-shot prompt (headless print mode; Claude Code parity) "
                   "— same as passing PROMPT positionally")
@click.option("--trust", is_flag=True, help="Trust mode (allow all tool calls)")
@click.option("--auto", "auto_mode", is_flag=True, help="Auto mode (allow reads, ask for writes)")
@click.option("--yolo", is_flag=True, help="Full agent mode (trust + autonomous execution)")
@click.option("--resume", "-r", is_flag=True, help="Resume the most recent session")
@click.option("--continue", "-c", "continue_prompt", default="", help="Resume last session and send a prompt")
@click.option("--setup", is_flag=True, help="Run interactive setup wizard")
@click.option("--version", "-v", is_flag=True, help="Show version")
@click.option("--output", "output_format", type=click.Choice(["text", "json"]),
              default="text",
              help="One-shot output format. 'json' prints a single machine-"
                   "readable JSON object to stdout (headless contract) instead "
                   "of streaming Rich text.")
@click.option("--max-rounds", "max_rounds", type=int, default=None,
              help="Override the agent's tool-round cap for this one-shot run.")
@click.argument("prompt", nargs=-1, required=False)
def main(endpoint, model_name, provider, prompt_flag, trust, auto_mode, yolo,
         resume, continue_prompt, setup, version, output_format, max_rounds,
         prompt):
    """Spark Code — Your local AI coding assistant."""
    if version:
        click.echo(f"Spark Code v{__version__}")
        return

    if setup:
        _run_setup()
        return

    # -p/--prompt (headless print mode, Phase 4 Task 1) is just another way
    # to supply the one-shot prompt. NOTE: -p was --provider's short flag
    # until Phase 4; the plan's headless contract (`spark -p "<prompt>"`,
    # mirroring Claude Code) claims it — use --provider long-form to pick a
    # provider.
    if prompt_flag:
        prompt = (prompt_flag, *prompt)

    # Load config with provider selection
    config = load_config(os.getcwd(), provider=provider)

    # Legacy-usage advisory: -p was --provider's short flag until Phase 4, so
    # `spark -p llm` used to select a provider. If the -p value exactly names
    # a configured provider, someone almost certainly meant the old flag —
    # note it on STDERR only (stdout/JSON purity and exit code untouched; the
    # value still runs as the prompt).
    if prompt_flag and prompt_flag in (config.get("providers") or {}):
        click.echo(
            "note: -p is now the prompt flag (Claude Code parity); "
            "for provider selection use --provider <name>", err=True)

    # CLI overrides
    if endpoint:
        config["model"]["endpoint"] = endpoint
    if model_name:
        config["model"]["name"] = model_name
    if yolo or trust:
        config["permissions"]["mode"] = "trust"
    elif auto_mode:
        config["permissions"]["mode"] = "auto"

    # Agentic mode flag
    config["_agentic"] = yolo

    # Resume / continue mode
    resume_session = ""
    if resume or continue_prompt:
        resume_session = _get_latest_session()
        if not resume_session:
            click.echo("No previous session to resume.")
            if not continue_prompt:
                # Fall through to interactive mode without resume
                pass

    # One-shot mode (skip if --resume or --continue) — this is the headless
    # contract (Phase 4 Task 1): non-interactive by construction, so exit
    # codes matter to callers (scripts/cron/JARVIS). Click's own main()
    # ignores this function's return value in standalone mode (it always
    # ctx.exit()s with 0), so the exit code must be forced via sys.exit here.
    if prompt and not resume and not continue_prompt:
        prompt_text = " ".join(prompt)
        exit_code = asyncio.run(_one_shot(
            config, prompt_text, output=output_format, max_rounds=max_rounds))
        sys.exit(exit_code)

    # Interactive mode
    asyncio.run(run_interactive(
        config,
        resume_session=resume_session,
        continue_prompt=continue_prompt,
    ))


async def _one_shot(config: dict, prompt: str, output: str = "text",
                     max_rounds: int | None = None) -> int:
    """Run a single prompt and exit — the headless contract (Phase 4 Task 1).

    This is the integration surface scripts/cron/JARVIS use to drive Spark as
    an engine. ``output="text"`` streams the same Rich rendering one-shot mode
    has always produced. ``output="json"`` prints exactly ONE JSON object to
    stdout (see spark_code.headless) and nothing else — the Console is built
    quiet so none of the agent's tool-call/streaming rendering reaches stdout.

    Non-interactive by construction (one-shot is never a TTY prompt session):
    the PermissionManager is ALWAYS built with ``interactive=False`` so a mode
    that would otherwise prompt (e.g. "ask", or "auto" on a write) fails safe
    by recording a denial instead of blocking on a prompt nothing can answer.
    ``--trust``/``--auto``/``--yolo`` (written into config by main()) still
    widen what's allowed without a prompt in the first place.

    Returns a process exit code: 0 = completed, 1 = error/exception (including
    a model/transport stream error), 2 = hit the tool-round cap without a
    final answer. main() forwards this to sys.exit().
    """
    json_mode = output == "json"
    # quiet=True (same mechanism dispatch.py already uses for worker
    # sub-agents) discards every Console.print call — Live display, tool-call
    # panels, permission-denial notices, everything — so stdout in JSON mode
    # carries ONLY the JSON line printed at the end of this function.
    console = Console(theme=get_theme(), quiet=json_mode)
    model = ModelClient(
        endpoint=get(config, "model", "endpoint"),
        model=get(config, "model", "name"),
        temperature=get(config, "model", "temperature", default=0.7),
        max_tokens=get(config, "model", "max_tokens", default=8192),
        api_key=get(config, "model", "api_key", default=""),
        provider=get(config, "model", "provider", default="ollama"),
        timeout=float(get(config, "model", "timeout", default=300)),
        supports_vision=bool(get(config, "model", "vision", default=False)),
    )
    # Honor --trust/--auto/--yolo (main() writes them into config) instead of
    # hardcoding "auto", so `spark --trust "do X"` doesn't still prompt.
    agentic = config.get("_agentic", False)
    # 2026-07-10: one-shot runs must carry the SAME provider/platform prompts
    # as interactive sessions — a bare Context here silently dropped
    # providers.<name>.system_prompt (found when the fine-tuned 30B narrated
    # instead of calling tools headlessly; the act-first policy never reached it).
    context = Context(
        system_prompt=AGENTIC_PROMPT if agentic else SYSTEM_PROMPT,
        # Carry the configured context window (same as the interactive build)
        # — without it Context falls back to its 32768 default, so small-window
        # configs overflow and large-window configs compact far too early.
        max_tokens=get(config, "model", "context_window", default=32768),
        platform_prompt=format_platform_prompt(os.getcwd()),
        provider_prompt=get(config, "model", "system_prompt", default=""),
    )
    permissions = PermissionManager(
        mode=get(config, "permissions", "mode", default="auto"),
        interactive=False)
    todo_list = TodoList()
    # Phase 5 Task 4: same custom-agent-def load as run_interactive — a
    # one-shot/headless run gets the same dispatch_agent custom types.
    agent_defs = load_agent_defs(os.getcwd())
    # Dual-model routing (Phase 4 Task 2) — built here (before build_tools,
    # same reordering rationale as run_interactive) so DispatchAgentTool can
    # route a model_hint=="utility" custom def to it.
    utility_model = get_utility_client(config)

    # Initialize MCP (browser/MCP bugfix): headless one-shot runs previously
    # never connected MCP servers at all — only run_interactive did — so e.g.
    # Playwright browser tools were invisible to `spark -p`/`--print`. Same
    # config key, same client class, same connect_all/disconnect_all contract
    # as run_interactive (~2693-2698 / ~4090). Everything from here through
    # the end of the function is wrapped in a try/finally so disconnect_all
    # runs on every exit path — including a failure in build_tools/Agent
    # construction below, not just inside agent.run().
    mcp_client = MCPClient()
    mcp_configs = get(config, "mcp_servers", default={})
    mcp_tools = []
    if mcp_configs:
        mcp_tools = await mcp_client.connect_all(mcp_configs)

    try:
        tools = build_tools(todo_list=todo_list, model=model, config=config,
                            permissions=permissions, agent_defs=agent_defs,
                            utility_model=resolve_utility_for(config, "dispatch", utility_model),
                            mcp_tools=mcp_tools, interactive=False)
        # Register MCP tools directly on the lead's own registry too (mirrors
        # run_interactive) so the top-level headless agent can call them itself,
        # not only a dispatched sub-agent — except agent-only servers (e.g. the
        # browser), which stay sub-agent-only to protect the local context.
        _register_mcp_tools(tools, mcp_tools, config)
        # Verification habit (Task 6) — see run_interactive for the full rationale.
        test_command = detect_test_command(os.getcwd(), load_instructions(os.getcwd()).text)

        stats = SessionStats()
        commands_run: list[str] = []

        def _on_tool_start(tool_name, args):
            # Bash tool invocations only — files_changed already comes from
            # agent._verify_written_paths (Phase 2), so this is the one piece
            # the agent doesn't already expose (per the task brief). Records
            # only commands that actually RAN: on_tool_start fires AFTER the
            # permission check in agent._execute_single_tool, so a denied bash
            # call never reaches this hook.
            if tool_name == "bash":
                cmd = args.get("command", "")
                if cmd:
                    commands_run.append(cmd)

        # Dual-model routing (Phase 4 Task 2) — same pattern as run_interactive:
        # built above (before build_tools), own lifecycle, use_for-gated for
        # compaction (the only OTHER routing consumer reachable from a headless
        # one-shot run — /review and /compact are interactive-only commands).
        agent = Agent(model, context, tools, permissions, console,
                      stats=stats, on_tool_start=_on_tool_start,
                      result_budgets=get(config, "tools", "result_budgets", default=None),
                      test_command=test_command, editor=_resolve_editor(config),
                      diff_in_editor=get(config, "ui", "diff_in_editor", default=False),
                      utility_model=resolve_utility_for(config, "compaction", utility_model),
                      agent_defs=agent_defs)
        if max_rounds is not None:
            # Instance attribute shadows the class constant — same pattern
            # dispatch.py already uses to give sub-agents their own round cap.
            agent.MAX_TOOL_ROUNDS = max_rounds

        result_text = ""
        error: str | None = None
        try:
            result_text = await agent.run(prompt)
            # A model/transport error ends the turn without raising (agent.py
            # renders it and breaks the loop) — surface it as a headless error
            # too, since it's user-actionable and otherwise invisible in JSON
            # mode (the render went to a quiet console).
            error = agent._last_stream_error
        except Exception as e:
            error = str(e)
        finally:
            await model.close()
            if utility_model is not None:
                try:
                    await utility_model.close()
                except Exception:
                    pass
            from .editor import cleanup_diff_temp_dirs
            cleanup_diff_temp_dirs()

        if error is not None:
            exit_code = 1
        elif agent._hit_max_rounds:
            exit_code = 2
        else:
            exit_code = 0

        # Session -> training corpus export (Phase 4 Task 6, opt-in, default
        # OFF). ``error`` is the same headless "error flag" that already drives
        # exit_code — export_session skips whenever it's set, so only clean
        # headless completions are ever exported.
        _export_corpus_session(config, context, agent, error=error)

        if json_mode:
            from .headless import HeadlessResult, build_headless_json
            headless_result = HeadlessResult(
                result=result_text,
                files_changed=list(agent._verify_written_paths),
                commands_run=commands_run,
                tool_calls=stats.total_tool_calls,
                rounds=agent._last_rounds,
                input_tokens=stats.input_tokens,
                output_tokens=stats.output_tokens,
                error=error,
            )
            click.echo(build_headless_json(headless_result))
        elif error is not None:
            # Text mode: still surface the error visibly (previously an uncaught
            # exception here crashed with a raw traceback) rather than exiting
            # silently with code 1.
            render_error(console, error)

        return exit_code
    finally:
        await mcp_client.disconnect_all()


if __name__ == "__main__":
    main()
