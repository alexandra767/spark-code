# Spark Code Polish V2 — 10 Improvements Design Spec

**Date:** 2026-03-26
**Status:** Approved
**Scope:** 10 features across agent, team, UI, and session systems

---

## Implementation Order

| Phase | Items | Risk | Description |
|-------|-------|------|-------------|
| 1 | #1, #4, #5, #6 | Low | /benchmark, compact notification, gen timer, /retry |
| 2 | #7, #8 | Medium | Model preload, edit confidence scoring |
| 3 | #2, #3 | Low | Worker timeout, /stats upgrade |
| 4 | #9, #10 | Medium | /branches list, multi-file diff summary |

---

## Phase 1: Quick Wins

### #1 — /benchmark

**Files:** `spark_code/skills/base.py`, `spark_code/cli.py`

New `/benchmark` slash command that measures model performance:

1. Send a standard prompt: "Write a Python function that checks if a number is prime"
2. Measure time-to-first-token (TTFT) and total generation speed
3. Display results:

```
> /benchmark
  Benchmarking qwen3.5:122b...
  Time to first token: 1.2s
  Generation speed: 9.4 tok/s
  Total tokens: 47 in 5.0s
```

Implementation: Send the prompt via `model.chat()` with streaming, track `first_token_time` and `token_count`, display results. Discard the response text — this is measurement only. Don't add the benchmark to conversation context.

### #4 — Auto-Compact Notification

**Files:** `spark_code/context.py`, `spark_code/cli.py`

When `context.compact()` runs, return a summary of what happened:

```python
def compact(self, keep_recent=6) -> str:
    """Returns summary string describing what was compacted."""
    before_count = len(self.messages)
    # ... existing compact logic ...
    after_count = len(self.messages)
    return f"Context compacted: {before_count} messages → {after_count} (freed ~{tokens_freed:,} tokens)"
```

In cli.py, wherever compact() is called, display the returned summary:

```
  ⚡ Context compacted: 45 messages → 7 (freed ~12,000 tokens)
```

### #5 — Generation Timer in Spinner

**Files:** `spark_code/ui/output.py` (StreamingRenderer)

Track elapsed time during generation. Update the spinner display to show seconds:

```
⠼ Generating... (4.2s)
```

In the StreamingRenderer, record `start_time = time.monotonic()` when `start()` is called. The spinner text updates each tick to show elapsed: `f"Generating... ({elapsed:.1f}s)"`.

Reset on each new generation cycle.

### #6 — /retry

**Files:** `spark_code/cli.py`

Store the last user message in a variable. `/retry` re-sends it.

```python
last_user_message = ""

# In the REPL loop, before sending to agent:
last_user_message = user_input

# /retry handler:
elif cmd == "/retry":
    if not last_user_message:
        console.print("  [#ebcb8b]No previous message to retry.[/#ebcb8b]")
        return None
    return f"__RETRY__{last_user_message}"
```

In the REPL loop, handle `__RETRY__` by calling `agent.run(message)` with the stored message.

---

## Phase 2: Model & Performance

### #7 — Model Preload

**Files:** `spark_code/cli.py`

After the banner renders and before the first prompt, fire a background warmup request:

```python
async def _warmup_model(model):
    """Send a tiny request to force model into VRAM."""
    try:
        async for _ in model.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[], stream=True
        ):
            break  # One chunk confirms model is loaded
    except Exception:
        pass  # Silent failure — don't block startup
```

Launch as `asyncio.create_task(_warmup_model(model))` right after the connection check succeeds. The task runs in the background while the user reads the banner and types their first message.

For Ollama: saves 30-60s on first message (model loads during warmup).
For Gemini/cloud: negligible (already fast), but warms the connection.

The warmup request is NOT added to conversation context.

### #8 — Edit Confidence Scoring

**Files:** `spark_code/tools/edit_file.py`

When `edit_file` fails with "old_string not found", enhance the error message with the closest matching block from the file:

1. After confirming old_string isn't found, use a sliding-window approach:
   - Split old_string into lines
   - Slide a window of the same line count across the file
   - Score each window using `difflib.SequenceMatcher.ratio()`
   - Keep the highest-scoring match

2. If best match is > 40% similar, include it in the error:

```
Error: old_string not found in calc.py.

Closest match (lines 47-63, 82% similar):
    def multiply(a, b):
        if not isinstance(a, (int, float)):
            raise TypeError(...)
        return a * b

Hint: Check whitespace, indentation, and exact string content.
```

3. If no match is > 40%, just return the standard error with a suggestion to read the file first.

Uses `difflib` from stdlib — no external deps. The model sees what's actually in the file and can self-correct on the next attempt.

---

## Phase 3: Worker & Stats

### #2 — Worker Timeout

**Files:** `spark_code/team.py`

Add `WORKER_TIMEOUT = 300` (5 minutes) constant. In `_run_worker()`, wrap the agent execution:

```python
async def _run_worker(self, worker, task_id):
    try:
        result = await asyncio.wait_for(
            worker.agent.run(worker.prompt),
            timeout=WORKER_TIMEOUT
        )
        worker.status = "completed"
        worker.result = result or "(completed with no text output)"
    except asyncio.TimeoutError:
        worker.status = "failed"
        worker.result = f"Timed out after {WORKER_TIMEOUT}s"
        self.lead_inbox.append(Message(
            from_name=worker.name,
            to_name="lead",
            content=f"[team] {worker.name} timed out after {WORKER_TIMEOUT}s"
        ))
    except asyncio.CancelledError:
        ...  # existing handling
```

Console output on timeout: `[worker-name] ✗ Timed out (5m)`

### #3 — /stats Upgrade

**Files:** `spark_code/cli.py` (the /stats handler)

Enhance the existing `/stats` display with new fields from the polish v1 work:

```
  Session Stats (3m 42s)
  ─────────────────────
  Generation: 9.1 tok/s avg | 1,247 tokens in / 892 tokens out
  Cost: $0.00 (ollama)
  Tools: 12 calls (bash: 5, write_file: 3, read_file: 2, glob: 2)
  Files: 3 created, 2 read, 1 edited
  Workers: 2 spawned, 2 completed
```

Data sources (all already available):
- `session_stats.last_tokens_per_sec` → Generation speed
- `session_stats.input_tokens` / `output_tokens` → Token counts
- `session_stats.session_cost` → Cost
- `session_stats.tool_calls` → Tool breakdown
- `session_stats.files_created` → Files created count
- `session_stats.files_read` / `files_written` / `files_edited` → File counts
- `team_manager.workers` → Worker count and status

Worker stats require passing `team_manager` reference into the stats handler.

---

## Phase 4: UX Enhancements

### #9 — /branches List

**Files:** `spark_code/cli.py`, `spark_code/branches.py`

The existing `/fork` command saves conversation branches. Add `/branches` to list them:

```
> /branches
  Conversation Branches:
    1. main (current) — 12 turns, 3m ago
    2. fork-auth-fix — 8 turns, 15m ago
    3. fork-refactor — 3 turns, 1h ago
```

Implementation:
- Read saved session files from `~/.spark/history/`
- Filter by current session prefix or branch metadata
- Display with turn count, time ago, and label
- Use existing `Context.read_metadata()` to get turn counts and timestamps

### #10 — Multi-File Diff Summary

**Files:** `spark_code/team.py`

After all workers complete, display a combined file change summary:

```
  Worker Summary:
    + fizzbuzz.py (29 lines, new)
    + test_fizzbuzz.py (46 lines, new)
    ~ calc.py (131 → 130 lines, modified)
```

Implementation:
- Track file operations per worker in team.py (already have `notify_file_written`)
- Maintain a `files_changed: list[dict]` on TeamManager: `{path, worker, action (created/modified), lines}`
- When `wait_for_workers` returns or all workers complete, format and display the summary
- Distinguish new files (write_file to non-existing path) from modifications (edit_file)

---

## Files Changed Summary

| File | Change |
|------|--------|
| `spark_code/cli.py` | /benchmark, /retry, /stats upgrade, /branches, model preload, compact display |
| `spark_code/context.py` | compact() returns summary string |
| `spark_code/tools/edit_file.py` | Closest-match error enhancement |
| `spark_code/team.py` | Worker timeout, file change tracking, diff summary |
| `spark_code/ui/output.py` | Generation timer in spinner |
| `spark_code/skills/base.py` | /benchmark and /retry skills |
| `spark_code/ui/input.py` | /benchmark, /retry, /branches autocomplete |
| `spark_code/branches.py` | List branches (may need new helper functions) |

## Config Changes

None — all features work with existing config.

## New Dependencies

None — uses `difflib` and `time` from stdlib.
