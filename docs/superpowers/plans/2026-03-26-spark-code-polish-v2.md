# Spark Code Polish V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 9 improvements to Spark Code: /benchmark, compact notification, generation timer, /retry, model preload, edit confidence scoring, worker timeout, /stats upgrade, and multi-file diff summary.

**Architecture:** Features are organized into 4 phases by dependency. Each phase is independent. Tests first, then implementation, frequent commits.

**Tech Stack:** Python 3.10+, asyncio, Rich, prompt_toolkit, difflib, pytest

**Note:** `/branches` was already implemented — removed from this plan.

---

## Phase 1: Quick Wins

### Task 1: Generation Timer in Spinner (#5)

**Files:**
- Modify: `spark_code/ui/output.py:392-410` (StreamingRenderer)
- Modify: `tests/test_polish_features.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_polish_features.py`:

```python
import time
from spark_code.ui.output import StreamingRenderer


class TestGenerationTimer:
    def test_renderer_tracks_start_time(self):
        import io
        from rich.console import Console
        console = Console(file=io.StringIO(), force_terminal=True)
        renderer = StreamingRenderer(console, live_mode=False)
        renderer.start()
        assert hasattr(renderer, '_start_time')
        assert renderer._start_time > 0

    def test_renderer_elapsed_increases(self):
        import io
        from rich.console import Console
        console = Console(file=io.StringIO(), force_terminal=True)
        renderer = StreamingRenderer(console, live_mode=False)
        renderer.start()
        time.sleep(0.1)
        assert renderer.elapsed > 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestGenerationTimer -v`
Expected: FAIL — AttributeError: '_start_time'

- [ ] **Step 3: Add timer to StreamingRenderer**

In `spark_code/ui/output.py`, modify `StreamingRenderer.__init__` (line 392) to add:

```python
        self._start_time: float = 0.0
```

Add property:

```python
    @property
    def elapsed(self) -> float:
        if self._start_time > 0:
            return time.monotonic() - self._start_time
        return 0.0
```

In `start()` method (line 400), add before the Live creation:

```python
        self._start_time = time.monotonic()
```

Update the Spinner text in `start()` to use a dynamic function. Replace the static Spinner with a timer-aware one. In `feed()`, update the spinner text when live is active:

In `_render()` method (line 424), before the `self._live.update()` call, if the buffer is empty (still generating, no text yet), update the spinner:

```python
    def _render(self):
        """Re-render the full buffer as markdown."""
        if not self._live:
            return
        full = "".join(self._buffer)
        if not full.strip():
            # Still waiting for text — show timer
            elapsed = self.elapsed
            self._live.update(
                Spinner("dots", text=Text(f" Generating... ({elapsed:.1f}s)", style=f"bold {_C_TOOL}"))
            )
            return
        try:
            self._live.update(Markdown(full, code_theme="nord-darker"))
        except Exception:
            self._live.update(Text(full))
```

Also add a periodic render call in `feed()` even when no content to update the timer. Actually, the timer only shows before first text arrives — the Spinner is replaced by markdown once text starts. So just update the spinner in `start()` and in `_render()` when buffer is empty.

- [ ] **Step 4: Run tests**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestGenerationTimer -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/ui/output.py tests/test_polish_features.py
git commit -m "feat: show elapsed time in generation spinner"
```

---

### Task 2: /retry Command (#6)

**Files:**
- Modify: `spark_code/cli.py` (slash command handler + REPL loop)
- Modify: `spark_code/ui/input.py` (autocomplete)
- Modify: `spark_code/skills/base.py` (skill entry)

- [ ] **Step 1: Add /retry to slash command handler**

In `spark_code/cli.py`, find the `handle_slash_command()` function. Add handler:

```python
    elif command == "/retry":
        return "__RETRY__"
```

- [ ] **Step 2: Store last user message and handle __RETRY__ in REPL loop**

In `spark_code/cli.py`, find where user_input is sent to the agent (the main REPL loop). Before the agent.run() call, store the message:

```python
                last_user_message = user_input
```

Declare `last_user_message = ""` before the REPL loop.

Add handler for `__RETRY__` in the result handling section:

```python
                elif result == "__RETRY__":
                    if not last_user_message:
                        console.print("  [#ebcb8b]No previous message to retry.[/#ebcb8b]")
                    else:
                        console.print(f"  [#88c0d0]Retrying: {last_user_message[:60]}...[/#88c0d0]")
                        result = await agent.run(last_user_message)
                    continue
```

- [ ] **Step 3: Add to autocomplete and help**

In `spark_code/ui/input.py`, add to `_BUILTIN_COMMANDS`:

```python
    "/retry": "Re-send the last message",
```

In `spark_code/cli.py`, add `/retry` to the help text.

- [ ] **Step 4: Commit**

```bash
cd ~/spark-code && git add spark_code/cli.py spark_code/ui/input.py
git commit -m "feat: add /retry command to re-send last message"
```

---

### Task 3: Auto-Compact Notification (#4)

**Files:**
- Modify: `spark_code/context.py:207-298` (compact method)
- Modify: `spark_code/cli.py` (display notification)
- Modify: `tests/test_polish_features.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_polish_features.py`:

```python
from spark_code.context import Context


class TestCompactNotification:
    def test_compact_returns_summary(self):
        ctx = Context()
        # Add enough messages to trigger compact
        for i in range(20):
            ctx.add_user(f"Message {i}")
            ctx.add_assistant(f"Response {i}")
        result = ctx.compact(keep_recent=6)
        assert result is not None
        assert isinstance(result, str)
        assert "compacted" in result.lower() or "messages" in result.lower()

    def test_compact_returns_none_when_nothing_to_compact(self):
        ctx = Context()
        ctx.add_user("hello")
        ctx.add_assistant("hi")
        result = ctx.compact(keep_recent=6)
        assert result is None or result == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestCompactNotification -v`
Expected: FAIL — compact() returns None (no return statement)

- [ ] **Step 3: Modify compact() to return summary**

In `spark_code/context.py`, modify `compact()` (line 207):

At the start of the method, capture the before count:

```python
    def compact(self, keep_recent: int = 6) -> str | None:
        """Compact conversation. Returns summary string or None if nothing to compact."""
        if len(self.messages) <= keep_recent:
            return None

        before_count = len(self.messages)
        before_tokens = self.estimate_tokens()
```

At the end of the method (after `self.messages = ...`), add:

```python
        after_count = len(self.messages)
        after_tokens = self.estimate_tokens()
        freed = before_tokens - after_tokens
        return f"Context compacted: {before_count} messages → {after_count} (freed ~{freed:,} tokens)"
```

- [ ] **Step 4: Display notification in cli.py**

Find where `context.compact()` is called in cli.py. Add display after the call:

```python
        compact_msg = context.compact()
        if compact_msg:
            console.print(f"  [#ebcb8b]⚡ {compact_msg}[/#ebcb8b]")
```

- [ ] **Step 5: Run tests**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestCompactNotification -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd ~/spark-code && git add spark_code/context.py spark_code/cli.py tests/test_polish_features.py
git commit -m "feat: show notification when context is auto-compacted"
```

---

### Task 4: /benchmark Command (#1)

**Files:**
- Modify: `spark_code/cli.py` (slash command handler + REPL handler)
- Modify: `spark_code/ui/input.py` (autocomplete)

- [ ] **Step 1: Add /benchmark handler**

In `spark_code/cli.py` `handle_slash_command()`, add:

```python
    elif command == "/benchmark":
        return "__BENCHMARK__"
```

- [ ] **Step 2: Implement benchmark in REPL loop**

In the REPL result handling, add:

```python
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
                        speed = token_count / (end - first_token_time) if first_token_time and (end - first_token_time) > 0 else 0
                        console.print(f"  [#a3be8c]Time to first token: {ttft:.1f}s[/#a3be8c]")
                        console.print(f"  [#a3be8c]Generation speed: {speed:.1f} tok/s[/#a3be8c]")
                        console.print(f"  [#a3be8c]Total: {token_count} tokens in {total_time:.1f}s[/#a3be8c]")
                    except Exception as e:
                        console.print(f"  [#bf616a]Benchmark failed: {e}[/#bf616a]")
                    continue
```

- [ ] **Step 3: Add to autocomplete and help**

In `spark_code/ui/input.py`, add to `_BUILTIN_COMMANDS`:

```python
    "/benchmark": "Measure model speed (time-to-first-token, tok/s)",
```

Add to help text in cli.py.

- [ ] **Step 4: Commit**

```bash
cd ~/spark-code && git add spark_code/cli.py spark_code/ui/input.py
git commit -m "feat: add /benchmark command to measure model speed"
```

---

## Phase 2: Model & Performance

### Task 5: Model Preload (#7)

**Files:**
- Modify: `spark_code/cli.py` (after banner, before REPL)

- [ ] **Step 1: Add warmup function**

In `spark_code/cli.py`, add a module-level async function (or near where the model is created):

```python
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
```

- [ ] **Step 2: Launch warmup after connection check**

Find where the connection check / ping happens (after banner). After a successful ping, add:

```python
        # Preload model into VRAM in background
        warmup_task = asyncio.create_task(_warmup_model(model))
```

Don't await it — let it run in the background while the user reads the banner.

- [ ] **Step 3: Commit**

```bash
cd ~/spark-code && git add spark_code/cli.py
git commit -m "feat: preload model into VRAM during startup"
```

---

### Task 6: Edit Confidence Scoring (#8)

**Files:**
- Modify: `spark_code/tools/edit_file.py:37-75`
- Modify: `tests/test_polish_features.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_polish_features.py`:

```python
import asyncio
from spark_code.tools.edit_file import EditFileTool


class TestEditConfidence:
    def test_not_found_shows_closest_match(self):
        import tempfile, os
        tool = EditFileTool()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
            f.write("def hello():\n    print('hello world')\n    return True\n")
            f.flush()
            path = f.name
        try:
            result = asyncio.run(tool.execute(
                file_path=path,
                old_string="def hello():\n    print('hello wrld')\n    return True\n",
                new_string="replaced",
            ))
            assert "closest match" in result.lower() or "similar" in result.lower()
            assert "hello" in result  # should show the actual content
        finally:
            os.unlink(path)

    def test_not_found_no_match_shows_hint(self):
        import tempfile, os
        tool = EditFileTool()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
            f.write("completely different content\n")
            f.flush()
            path = f.name
        try:
            result = asyncio.run(tool.execute(
                file_path=path,
                old_string="this string does not exist anywhere at all in the file whatsoever",
                new_string="replaced",
            ))
            assert "not found" in result.lower()
        finally:
            os.unlink(path)

    def test_exact_match_still_works(self):
        import tempfile, os
        tool = EditFileTool()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
            f.write("line one\nline two\nline three\n")
            f.flush()
            path = f.name
        try:
            result = asyncio.run(tool.execute(
                file_path=path,
                old_string="line two",
                new_string="line TWO",
            ))
            assert "successfully" in result.lower()
        finally:
            os.unlink(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestEditConfidence -v`
Expected: test_not_found_shows_closest_match FAILS (no "closest match" in error), others may pass

- [ ] **Step 3: Add closest-match logic to edit_file.py**

In `spark_code/tools/edit_file.py`, add import at top:

```python
import difflib
```

Replace the "not found" error section (line 53-54):

```python
        if old_string not in content:
            return f"Error: old_string not found in {path}. Make sure it matches exactly (including whitespace)."
```

With:

```python
        if old_string not in content:
            # Find closest matching block
            hint = self._find_closest_match(content, old_string)
            base_msg = f"Error: old_string not found in {path}."
            if hint:
                return f"{base_msg}\n\n{hint}\n\nHint: Check whitespace, indentation, and exact string content."
            return f"{base_msg} Make sure it matches exactly (including whitespace). Try reading the file first."
```

Add the helper method to the class:

```python
    @staticmethod
    def _find_closest_match(content: str, old_string: str) -> str:
        """Find the most similar block in the file to old_string."""
        old_lines = old_string.splitlines()
        file_lines = content.splitlines()
        window_size = len(old_lines)

        if window_size == 0 or len(file_lines) == 0:
            return ""

        best_ratio = 0.0
        best_start = 0

        for i in range(max(1, len(file_lines) - window_size + 1)):
            window = file_lines[i:i + window_size]
            window_text = "\n".join(window)
            ratio = difflib.SequenceMatcher(None, old_string, window_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio < 0.4:
            return ""

        best_window = file_lines[best_start:best_start + window_size]
        match_text = "\n".join(f"    {line}" for line in best_window)
        start_line = best_start + 1
        end_line = best_start + window_size
        return f"Closest match (lines {start_line}-{end_line}, {best_ratio:.0%} similar):\n{match_text}"
```

- [ ] **Step 4: Run tests**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestEditConfidence -v`
Expected: All 3 PASS

- [ ] **Step 5: Run full test suite**

Run: `cd ~/spark-code && python -m pytest tests/ -v --timeout=30`
Expected: No regressions

- [ ] **Step 6: Commit**

```bash
cd ~/spark-code && git add spark_code/tools/edit_file.py tests/test_polish_features.py
git commit -m "feat: show closest matching block when edit_file old_string not found"
```

---

## Phase 3: Worker & Stats

### Task 7: Worker Timeout (#2)

**Files:**
- Modify: `spark_code/team.py` (_run_worker method)
- Modify: `tests/test_polish_features.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_polish_features.py`:

```python
import asyncio
import io
from rich.console import Console
from spark_code.team import TeamManager, Worker, WORKER_TIMEOUT
from spark_code.tools.base import ToolRegistry
from spark_code.task_store import TaskStore


class TestWorkerTimeout:
    def test_worker_timeout_constant_exists(self):
        from spark_code.team import WORKER_TIMEOUT
        assert WORKER_TIMEOUT > 0
        assert WORKER_TIMEOUT == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestWorkerTimeout -v`
Expected: FAIL — ImportError: cannot import name 'WORKER_TIMEOUT'

- [ ] **Step 3: Add WORKER_TIMEOUT and wrap _run_worker with asyncio.wait_for**

In `spark_code/team.py`, add constant after MAX_WORKERS:

```python
WORKER_TIMEOUT = 300  # 5 minutes
```

In `_run_worker()` method, wrap the agent.run() call:

```python
    async def _run_worker(self, worker, task_id):
        try:
            result = await asyncio.wait_for(
                worker.agent.run(worker.prompt),
                timeout=WORKER_TIMEOUT
            )
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
                Text(f"  [{worker.name}] ✓ Completed", style=_C_GREEN))

        except asyncio.TimeoutError:
            worker.status = "failed"
            worker.result = f"Timed out after {WORKER_TIMEOUT}s"
            self.task_store.update(task_id, status="failed",
                                   result=worker.result)
            self.lead_inbox.append(Message(
                from_name=worker.name,
                to_name="lead",
                content=f"[team] {worker.name} timed out after {WORKER_TIMEOUT}s"
            ))
            self.console.print(
                Text(f"  [{worker.name}] ✗ Timed out ({WORKER_TIMEOUT}s)",
                     style=_C_RED))

        except asyncio.CancelledError:
            worker.status = "failed"
            worker.result = "Cancelled"
            self.task_store.update(task_id, status="failed", result="Cancelled")
            self.console.print(
                Text(f"  [{worker.name}] ✗ Cancelled", style=_C_YELLOW))

        except Exception as e:
            worker.status = "failed"
            worker.result = str(e)[:500]
            self.task_store.update(task_id, status="failed", result=str(e)[:200])
            self.console.print(
                Text(f"  [{worker.name}] ✗ Failed: {e}", style=_C_RED))
```

- [ ] **Step 4: Run tests**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestWorkerTimeout -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/team.py tests/test_polish_features.py
git commit -m "feat: add 5-minute timeout for background workers"
```

---

### Task 8: /stats Upgrade (#3)

**Files:**
- Modify: `spark_code/cli.py:606-634` (/stats handler)

- [ ] **Step 1: Upgrade the /stats display**

Replace the existing /stats handler (lines 606-634 in cli.py) with an enhanced version:

```python
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
        table.add_row("Tokens in / out",
                       f"{stats.input_tokens:,} / {stats.output_tokens:,}")

        # Cost
        cost_str = stats.format_cost()
        if cost_str:
            provider_name = get(config, "model", "provider", default="")
            table.add_row("Session cost", f"{cost_str} ({provider_name})")
        else:
            table.add_row("Session cost", "$0.00 (local)")

        # Tool breakdown
        table.add_row("Total tool calls", str(stats.total_tool_calls))
        if stats.tool_calls:
            for tool_name, count in sorted(stats.tool_calls.items(),
                                            key=lambda x: -x[1]):
                table.add_row(f"  {tool_name}", str(count))

        # Files
        table.add_row("Files",
                       f"{len(stats.files_created)} created, "
                       f"{len(stats.files_read)} read, "
                       f"{len(stats.files_edited)} edited")
        table.add_row("Commands run", str(stats.commands_run))

        # Workers (if team exists)
        if team_manager:
            workers = team_manager.workers
            total = len(workers)
            completed = sum(1 for w in workers.values() if w.status == "completed")
            failed = sum(1 for w in workers.values() if w.status == "failed")
            if total > 0:
                table.add_row("Workers",
                               f"{total} spawned, {completed} completed"
                               + (f", {failed} failed" if failed else ""))

        if pinned and pinned.count > 0:
            table.add_row("Pinned files", str(pinned.count))
        console.print(table)
        return None
```

Note: `team_manager` needs to be accessible in the handle_slash_command function scope. Check how it's passed — it may need to be added as a parameter or accessed via closure.

- [ ] **Step 2: Verify team_manager is accessible**

Check if `handle_slash_command` has access to `team_manager`. If not, add it as a parameter or as a nonlocal variable.

- [ ] **Step 3: Commit**

```bash
cd ~/spark-code && git add spark_code/cli.py
git commit -m "feat: upgrade /stats with speed, cost, tokens, workers, files created"
```

---

## Phase 4: Multi-File Diff Summary

### Task 9: Worker File Change Summary (#10)

**Files:**
- Modify: `spark_code/team.py` (track file changes, format summary)
- Modify: `tests/test_polish_features.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_polish_features.py`:

```python
class TestWorkerFileSummary:
    def test_team_tracks_file_changes(self):
        import io
        from rich.console import Console
        from spark_code.tools.base import ToolRegistry
        from spark_code.task_store import TaskStore
        from spark_code.team import TeamManager
        console = Console(file=io.StringIO(), force_terminal=True)
        team = TeamManager(model=None, tools=ToolRegistry(),
                          console=console, task_store=TaskStore())
        assert hasattr(team, 'files_changed')
        assert isinstance(team.files_changed, list)

    def test_notify_records_file_change(self):
        import io
        from rich.console import Console
        from spark_code.tools.base import ToolRegistry
        from spark_code.task_store import TaskStore
        from spark_code.team import TeamManager
        console = Console(file=io.StringIO(), force_terminal=True)
        team = TeamManager(model=None, tools=ToolRegistry(),
                          console=console, task_store=TaskStore())
        team.notify_file_written("worker-1", "/tmp/test.py", 50)
        assert len(team.files_changed) == 1
        assert team.files_changed[0]["path"] == "/tmp/test.py"

    def test_format_file_summary(self):
        import io
        from rich.console import Console
        from spark_code.tools.base import ToolRegistry
        from spark_code.task_store import TaskStore
        from spark_code.team import TeamManager
        console = Console(file=io.StringIO(), force_terminal=True)
        team = TeamManager(model=None, tools=ToolRegistry(),
                          console=console, task_store=TaskStore())
        team.notify_file_written("worker-1", "/tmp/foo.py", 29)
        team.notify_file_written("worker-2", "/tmp/bar.py", 46)
        summary = team.format_file_summary()
        assert "foo.py" in summary
        assert "bar.py" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestWorkerFileSummary -v`
Expected: FAIL — AttributeError 'files_changed'

- [ ] **Step 3: Add file tracking to TeamManager**

In `spark_code/team.py`, in `__init__`:

```python
        self.files_changed: list[dict] = []
```

In `notify_file_written()`, add tracking:

```python
        self.files_changed.append({
            "path": file_path,
            "worker": writer_name,
            "lines": line_count,
        })
```

Add format method:

```python
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
```

- [ ] **Step 4: Display summary when wait_for_workers returns**

In `spark_code/tools/wait_for_workers.py`, at the end of `execute()`, before returning, add the file summary if the team has it:

```python
        # Append file change summary
        if self._team and self._team.files_changed:
            result_lines = lines  # already built above
            result_lines.append("")
            result_lines.append(self._team.format_file_summary())
```

- [ ] **Step 5: Run tests**

Run: `cd ~/spark-code && python -m pytest tests/test_polish_features.py::TestWorkerFileSummary -v`
Expected: All 3 PASS

- [ ] **Step 6: Commit**

```bash
cd ~/spark-code && git add spark_code/team.py spark_code/tools/wait_for_workers.py tests/test_polish_features.py
git commit -m "feat: track and display worker file change summary"
```

---

## Summary

| Task | Feature | Files | Tests |
|------|---------|-------|-------|
| 1 | Generation timer | ui/output.py | 2 |
| 2 | /retry | cli.py, ui/input.py | manual |
| 3 | Compact notification | context.py, cli.py | 2 |
| 4 | /benchmark | cli.py, ui/input.py | manual |
| 5 | Model preload | cli.py | manual |
| 6 | Edit confidence | tools/edit_file.py | 3 |
| 7 | Worker timeout | team.py | 1 |
| 8 | /stats upgrade | cli.py | manual |
| 9 | File change summary | team.py, wait_for_workers.py | 3 |

**Total: 9 tasks, ~11 new tests, 4 phases**
