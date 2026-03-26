# Spark Code Polish — 12 Improvements Design Spec

**Date:** 2026-03-26
**Status:** Approved
**Scope:** 11 features + 1 refactor across agent, team, UI, and session systems

---

## Implementation Order

| Phase | Items | Risk | Description |
|-------|-------|------|-------------|
| 1 | #5, #1, #3 | Low | Platform prompt, provider system prompt, round warnings |
| 2 | #4, #8 | Low | Tokens/sec display, cost tracking |
| 3 | #11, #2, #6 | Medium | Worker progress, wait_for_workers, file notifications |
| 4 | #7, #10, #12 | Medium | Session resume, bash warnings, /clean |
| 5 | #9 | Higher | cli.py refactor (isolated, last) |

---

## Phase 1: Agent & Prompt Improvements

### #5 — Platform-Aware System Prompt

**File:** `spark_code/context.py`

Auto-detect platform info at startup and inject at the top of every system prompt:

```
Platform: macOS (Darwin), Shell: zsh, CWD: /Users/.../learning_python
Python: 3.14, Package manager: brew
```

Gathered once via `platform` module + quick shell checks (`which brew`, `python3 --version`). Cached at startup — no per-request overhead. No config needed.

**Why:** Models like qwen3-coder-next tried `/scratch`, `/home/user/`, and `timeout` (Linux-only) because they didn't know the environment.

### #1 — System Prompt Per Provider

**Files:** `spark_code/config.py`, `spark_code/context.py`

Add optional `system_prompt` field to provider config in `config.yaml`:

```yaml
providers:
  coder:
    model: qwen3-coder-next:q8_0
    system_prompt: |
      Always write files to the current working directory.
      Use the bash tool for shell commands. Do not use unknown tool names.
  ollama:
    model: qwen3.5:122b
    system_prompt: ""
```

In `context.py`, when building the system prompt:
1. Start with platform info (#5)
2. Append provider-specific system_prompt (if non-empty)
3. Append the default SYSTEM_PROMPT or AGENTIC_PROMPT

Empty string or missing field = skip, use defaults only.

**Why:** Different models have different tool-calling conventions and assumptions. The coder model needs explicit guidance that the MoE model doesn't.

### #3 — Graceful Tool Round Warning

**File:** `spark_code/agent.py`

Replace the hard cutoff surprise with a two-stage wind-down. In the agent loop, inject system messages at specific round thresholds:

- **Round MAX_TOOL_ROUNDS - 15 (60):** Inject system message:
  `"You have 15 tool rounds remaining. Begin wrapping up — summarize progress and finish current work."`
- **Round MAX_TOOL_ROUNDS - 5 (70):** Inject system message:
  `"5 tool rounds remaining. Finish immediately."`
- **Round MAX_TOOL_ROUNDS (75):** Hard stop (existing behavior).

System messages are appended to `context.messages` as role=system before the next model call.

**Why:** The model was mid-debugging when it hit the wall at 25 rounds. A warning lets it wrap up gracefully.

---

## Phase 2: Stats & Display

### #4 — Tokens/Sec Display

**Files:** `spark_code/model.py`, `spark_code/stats.py`, `spark_code/ui/input.py`

In `model.py` streaming response handler:
- Record `first_token_time = time.monotonic()` on first chunk
- Increment `token_count` per chunk (estimate: split on whitespace, ~1.3 tokens per word; or count chunk characters / 4)
- On stream end, calculate `tokens_per_sec = token_count / (end_time - first_token_time)`
- Store latest `tokens_per_sec` on the stats object

In `ui/input.py` bottom toolbar, append: `42 tok/s` after existing token count display.

**Why:** User wanted to compare model speeds. Currently no visibility into generation performance.

### #8 — Cost Tracking Per Provider

**Files:** `spark_code/config.py`, `spark_code/stats.py`, `spark_code/ui/input.py`

Add optional cost fields to provider config:

```yaml
providers:
  gemini:
    cost_per_million_input: 0.075
    cost_per_million_output: 0.30
  ollama:
    cost_per_million_input: 0
    cost_per_million_output: 0
```

In `stats.py`:
- Track input_tokens and output_tokens separately (already partially done)
- Calculate `session_cost = (input_tokens * cost_input + output_tokens * cost_output) / 1_000_000`
- Expose `session_cost` property

Display in session stats summary and bottom toolbar: `$0.003` (or `$0.00` for Ollama).

Default to $0 if cost fields are missing (local models are free).

**Why:** Useful for Gemini/OpenAI usage awareness. Ollama users see $0.00 confirming they're running free.

---

## Phase 3: Worker & Team Improvements

### #11 — Worker Progress in Team Bar

**Files:** `spark_code/team.py`, `spark_code/ui/hotkeys.py`

Add `current_tool: str = ""` field to the `Worker` dataclass.

In the worker's agent loop, before executing a tool:
```python
worker.current_tool = tool.name
```
After execution:
```python
worker.current_tool = ""
```

Update the team status bar renderer to show:
```
⟳ worker-calc [write_file]  ⟳ worker-test [read_file]  ✓ worker-cli
```

Format: `⟳ {name} [{current_tool}]` when tool is active, `⟳ {name}` when generating text.

**Why:** Currently the team bar just shows spinning icons. No visibility into what each worker is actually doing.

### #2 — Worker Completion Awareness (wait_for_workers)

**Files:** `spark_code/tools/wait_for_workers.py` (new), `spark_code/tools/base.py`, `spark_code/team.py`

New tool:

```python
class WaitForWorkersTool(Tool):
    name = "wait_for_workers"
    description = (
        "Wait for background workers to complete and return their results. "
        "Optionally specify worker names; defaults to waiting for all running workers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific worker names to wait for. Empty = all."
            },
            "timeout": {
                "type": "number",
                "description": "Max seconds to wait. Default: 300 (5 min)."
            }
        }
    }
```

Implementation:
- Poll `team.workers` every 2 seconds
- Return when all specified workers have status != "running"
- Return value: summary of each worker's result/status
- Timeout returns partial results with warning

Additionally: when any worker completes, auto-inject a system message into the lead's context:
```
[team] worker-calc completed: "Calculator module created and tested successfully."
```

This is done in `team.py` by appending to `lead_inbox` on worker completion.

**Why:** The lead agent checked for files before workers finished writing them. It needs a way to wait.

### #6 — Worker File Notifications

**Files:** `spark_code/team.py`

When a worker executes `write_file` or `edit_file`, broadcast a notification to all other running workers' inboxes:

```
[team] worker-calc wrote calc.py (131 lines)
```

Implementation:
- In `team.py`, add a `notify_file_written(worker_name, file_path, line_count)` method
- Hook into the worker's tool execution: after write_file/edit_file succeeds, call notify
- Notification goes into each other worker's `inbox` as a system message
- Workers receive it on their next agent loop iteration

No filesystem watcher needed — piggybacks on existing tool execution.

**Why:** worker-test kept polling for calc.py via glob. With notifications, it knows immediately when the file is available.

---

## Phase 4: Session & UX

### #7 — Session Resume with Checkpoint

**Files:** `spark_code/agent.py`, `spark_code/cli.py` (or `cli/commands.py` after refactor), `spark_code/context.py`

**Checkpoint save** — When agent hits MAX_TOOL_ROUNDS, auto-save:

```python
checkpoint = {
    "messages": context.messages,
    "cwd": os.getcwd(),
    "provider": config.active_provider,
    "model": config.model_name,
    "round_count": rounds,
    "timestamp": datetime.now().isoformat(),
    "stats": {
        "tokens_used": stats.total_tokens,
        "files_created": list(stats.files_created)
    }
}
# Save to ~/.spark/checkpoints/latest.json
```

**`/continue` command:**
1. Load `~/.spark/checkpoints/latest.json`
2. Restore context messages
3. Reset round counter to 0
4. Inject system message: `"Session resumed from checkpoint at round {N}/{MAX}. Continue where you left off."`
5. Resume the agent loop

**Cleanup:** Checkpoints older than 24 hours are deleted on startup.

**Edge cases:**
- No checkpoint exists: print "No checkpoint found"
- Provider changed since checkpoint: warn but allow (model can adapt)
- CWD changed: warn and show both paths

**Why:** Hitting the tool limit kills all context. Users had to type "continue" manually and hope. This preserves everything across restarts.

### #10 — Bash Side-Effect Warnings

**File:** `spark_code/tools/bash.py`

Before executing a bash command, check against known side-effect patterns:

```python
SIDE_EFFECT_PATTERNS = [
    (r'\bpip\s+install\b', "Installs packages into the active Python environment"),
    (r'\bnpm\s+install\b', "Modifies node_modules and package-lock.json"),
    (r'\brm\s', "Deletes files or directories"),
    (r'\bgit\s+(push|reset|checkout)\b', "Modifies git state"),
    (r'\bbrew\s+install\b', "Installs system-level packages"),
    (r'\bcurl\b.*\|\s*(bash|sh)\b', "Pipes remote script to shell"),
    (r'\bsudo\b', "Runs with elevated privileges"),
    (r'\bdocker\s+(rm|rmi|stop|kill)\b', "Modifies Docker containers/images"),
]
```

Behavior by permission mode:
- **trust mode:** No warning (autonomous by design)
- **auto mode:** Bash is auto-allowed for read-only commands. If a side-effect pattern matches, force a permission prompt even if bash is in the always_allow list. Show the warning in the permission prompt.
- **ask mode:** Show the warning alongside the normal permission prompt.

Pattern matching uses `re.search` on the full command string. Multiple matches show all warnings.

**Why:** The coder model ran `pip install` without asking. In auto mode, bash was auto-allowed so the user had no chance to review.

### #12 — `/clean` Command

**Files:** `spark_code/skills/base.py`, `spark_code/stats.py`

Track files created (not just edited) during the session in `stats.py`:
- `files_created: set[str]` — paths written by `write_file` that didn't exist before
- `files_edited: set[str]` — paths modified by `edit_file` that existed before

New `/clean` skill:
1. List all files in `stats.files_created`
2. Check which still exist (some may have been deleted already)
3. Show the list with line counts
4. Prompt for confirmation
5. Delete confirmed files
6. Remove empty parent directories left behind (only if they were also created this session)

```
> /clean
Files created this session:
  calc.py (131 lines)
  test_calc.py (289 lines)
  calc_cli.py (91 lines)

Delete all? [y/N/select]
```

`select` option lets user pick which to keep.

**Why:** Testing leaves files behind. User had to ask me to clean up manually.

---

## Phase 5: Refactor (Last)

### #9 — Split cli.py Into Submodules

**Current:** `spark_code/cli.py` — 114KB single file.

**Target structure:**
```
spark_code/cli/
  __init__.py      # Entry point: main(), imports from submodules
  session.py       # Session class: save, load, history, conversation state
  commands.py      # Slash command dispatcher: /help, /model, /team, /plan, etc.
  repl.py          # Main REPL loop: input → agent → display cycle
  startup.py       # Banner rendering, config loading, provider connection check
  image.py         # Image handling: /image command, drag-and-drop
```

**Approach:**
1. Extract one module at a time, starting with the most isolated (`startup.py`)
2. Run full test suite after each extraction
3. Keep `cli/__init__.py` as the public API — `pyproject.toml` entry point unchanged
4. No behavior changes — pure structural refactor
5. Order: startup → image → session → commands → repl (most dependent last)

**Shared state:** Create a `SessionState` dataclass that holds config, context, agent, team, console, stats. Pass it to each submodule instead of relying on module-level globals.

**Why:** 114KB is unmaintainable. Functions are hard to find, changes risk unrelated breakage, and it's difficult for both humans and AI to reason about.

---

## Files Changed Summary

| File | Change |
|------|--------|
| `spark_code/context.py` | Platform info injection, provider system prompt |
| `spark_code/config.py` | New fields: system_prompt, cost_per_million_input/output |
| `spark_code/agent.py` | Round warnings, checkpoint save |
| `spark_code/model.py` | Token speed tracking |
| `spark_code/stats.py` | tokens_per_sec, session_cost, files_created tracking |
| `spark_code/team.py` | Worker current_tool, file notifications, completion notifications |
| `spark_code/tools/wait_for_workers.py` | **New** — wait for workers tool |
| `spark_code/tools/bash.py` | Side-effect pattern warnings |
| `spark_code/tools/base.py` | Register new tool |
| `spark_code/ui/input.py` | Bottom toolbar: tok/s, cost display |
| `spark_code/ui/hotkeys.py` | Worker progress display |
| `spark_code/skills/base.py` | /clean skill, /continue command |
| `spark_code/cli.py` → `spark_code/cli/` | Phase 5 refactor |

## New Files

| File | Purpose |
|------|---------|
| `spark_code/tools/wait_for_workers.py` | Wait for workers tool |
| `~/.spark/checkpoints/latest.json` | Session checkpoint (auto-managed) |

## Config Changes (Non-Breaking)

All new config fields are optional with sensible defaults:
- `providers.*.system_prompt` — default: "" (empty, use default prompt)
- `providers.*.cost_per_million_input` — default: 0
- `providers.*.cost_per_million_output` — default: 0
