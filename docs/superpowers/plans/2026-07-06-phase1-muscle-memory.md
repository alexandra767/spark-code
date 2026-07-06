# Phase 1: Drop-in Muscle Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Claude Code user's hands work unchanged in Spark: CLAUDE.md respected, `@`-file mentions, Esc interrupts generation, `/resume` picker, `/rewind`, Claude Code permission-mode names and Shift+Tab order.

**Architecture:** New `spark_code/instructions.py` unifies project-instruction loading (SPARK.md + CLAUDE.md hierarchy); everything else extends existing machinery at known anchors — `_detect_file_mentions` (cli.py:378) for `@`-mentions, `_run_with_notify` (cli.py:2324) for the Esc watcher, `/history` machinery (cli.py:1426) for `/resume`, the BackTab binding (ui/input.py:251) for cycle order. One hygiene task pays down Phase 0 deferred items that this phase's tasks touch.

**Tech Stack:** Python 3.11, click, prompt_toolkit, Rich, pytest (+pytest-asyncio, auto mode), ruff, pexpect (already in the venv; used for interactive verification).

## Global Constraints

- All edits in `~/spark-code` (the editable install). `~/CodingProjects/spark-code` is a symlink.
- vLLM served-model alias `qwen3.5:122b` on `http://spark-4a54.local:30000` is a hard contract; `coder` = Ollama `qwen3-coder:30b` on `:11434`. Never key model behavior on model names.
- Gates before every commit: `.venv/bin/python -m pytest -q` (610 green at start) and `.venv/bin/ruff check .` (0 errors at start).
- Existing Spark command names/flags keep working — every parity name is an ADDITION (alias), never a rename-and-break.
- Live acceptance: `scripts/smoke_live.py --provider llm` (and `coder` when loadable) 7/7 at phase end.
- Work on a new branch `phase1/muscle-memory` off current `main` (22c5dd3).
- Commits: small, per-feature, `feat:`/`fix:`/`style:` style.

---

### Task 1: Unified project instructions (CLAUDE.md + SPARK.md) + banner

**Files:**
- Create: `spark_code/instructions.py`
- Create: `tests/test_instructions.py`
- Modify: `spark_code/cli.py` — replace direct `load_spark_md()` usages (definition at cli.py:87; call sites: `grep -n "load_spark_md" spark_code/cli.py`) with the new loader; banner currently prints "SPARK.md loaded" at cli.py:495-496 — extend to name each loaded source.

**Interfaces:**
- Produces: `load_instructions(cwd: str) -> Instructions` where `Instructions` is a dataclass with `text: str` (merged, precedence-ordered, budget-truncated) and `sources: list[str]` (display names in load order, e.g. `["SPARK.md", "CLAUDE.md", "~/.claude/CLAUDE.md"]`). Task 7 (acceptance) and the banner consume `sources`; the system prompt consumes `text`.
- Consumes: nothing from other tasks.

**Semantics (from the spec):**
- Project CLAUDE.md: walk UP from cwd to filesystem root (like Claude Code), first hit wins. SPARK.md: current behavior (cwd `SPARK.md` or `.spark/SPARK.md` — no walk-up, unchanged).
- Global: `~/.claude/CLAUDE.md` if it exists.
- Precedence on merge: SPARK.md first (highest), then project CLAUDE.md, then global CLAUDE.md — each under a header line `## Instructions from <source>`.
- Per-file budget: 8000 chars; truncate at the budget with a trailing `\n[... truncated]` marker. Files read with `errors="replace"`; any OSError → skip that source silently.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_instructions.py
import os

from spark_code.instructions import Instructions, load_instructions


def _write(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_loads_project_claude_md(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "proj/CLAUDE.md", "use tabs")
    got = load_instructions(str(tmp_path / "proj"))
    assert "use tabs" in got.text
    assert "CLAUDE.md" in got.sources


def test_claude_md_walks_up_first_hit_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "root/CLAUDE.md", "root rules")
    _write(tmp_path, "root/pkg/CLAUDE.md", "pkg rules")
    (tmp_path / "root" / "pkg" / "deep").mkdir(parents=True, exist_ok=True)
    got = load_instructions(str(tmp_path / "root" / "pkg" / "deep"))
    assert "pkg rules" in got.text
    assert "root rules" not in got.text


def test_precedence_spark_md_first(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "proj/SPARK.md", "spark says")
    _write(tmp_path, "proj/CLAUDE.md", "claude says")
    got = load_instructions(str(tmp_path / "proj"))
    assert got.text.index("spark says") < got.text.index("claude says")
    assert got.sources[0] == "SPARK.md"


def test_global_claude_md_loaded_last(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "home/.claude/CLAUDE.md", "global rules")
    _write(tmp_path, "proj/CLAUDE.md", "project rules")
    got = load_instructions(str(tmp_path / "proj"))
    assert got.text.index("project rules") < got.text.index("global rules")
    assert got.sources[-1] == "~/.claude/CLAUDE.md"


def test_per_file_budget_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    _write(tmp_path, "proj/CLAUDE.md", "x" * 20000)
    got = load_instructions(str(tmp_path / "proj"))
    assert len(got.text) < 10000
    assert "[... truncated]" in got.text


def test_no_files_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path / "home")))
    (tmp_path / "empty").mkdir()
    got = load_instructions(str(tmp_path / "empty"))
    assert got.text == ""
    assert got.sources == []
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_instructions.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'spark_code.instructions'`

- [ ] **Step 3: Implement `spark_code/instructions.py`**

```python
"""Unified project-instruction loading: SPARK.md + CLAUDE.md hierarchy.

Precedence (first = highest): SPARK.md, project CLAUDE.md (nearest ancestor),
~/.claude/CLAUDE.md. Mirrors Claude Code's CLAUDE.md discovery so the same
repos work in both tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PER_FILE_BUDGET = 8000


@dataclass
class Instructions:
    text: str = ""
    sources: list[str] = field(default_factory=list)


def _read_budgeted(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read(PER_FILE_BUDGET + 1)
    except OSError:
        return None
    if not content.strip():
        return None
    if len(content) > PER_FILE_BUDGET:
        content = content[:PER_FILE_BUDGET] + "\n[... truncated]"
    return content


def _find_project_claude_md(cwd: str) -> str | None:
    current = os.path.realpath(cwd)
    while True:
        candidate = os.path.join(current, "CLAUDE.md")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def load_instructions(cwd: str) -> Instructions:
    """Load and merge all project-instruction files for ``cwd``."""
    candidates: list[tuple[str, str | None]] = []

    for rel in ("SPARK.md", os.path.join(".spark", "SPARK.md")):
        path = os.path.join(cwd, rel)
        if os.path.isfile(path):
            candidates.append(("SPARK.md", path))
            break
    else:
        candidates.append(("SPARK.md", None))

    candidates.append(("CLAUDE.md", _find_project_claude_md(cwd)))
    global_path = os.path.expanduser(os.path.join("~", ".claude", "CLAUDE.md"))
    candidates.append(("~/.claude/CLAUDE.md",
                       global_path if os.path.isfile(global_path) else None))

    parts: list[str] = []
    sources: list[str] = []
    for name, path in candidates:
        if path is None:
            continue
        content = _read_budgeted(path)
        if content is None:
            continue
        parts.append(f"## Instructions from {name}\n\n{content}")
        sources.append(name)
    return Instructions(text="\n\n".join(parts), sources=sources)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_instructions.py -q`
Expected: 6 passed

- [ ] **Step 5: Wire into cli.py**

Find every `load_spark_md()` call site (`grep -n "load_spark_md" spark_code/cli.py`) and replace the system-prompt injection with `load_instructions(os.getcwd()).text`, keeping `load_spark_md` itself (other code may import it; deprecate in place with a comment pointing at instructions.py). Banner: where cli.py:495-496 appends `"SPARK.md loaded\n"`, iterate `instructions.sources` and append one `f"{name} loaded\n"` line per source, same style. `/clear` re-injection (the post-clear banner path — `grep -n "spark_md" spark_code/cli.py` finds it) uses the same loader.

- [ ] **Step 6: Full gates + commit**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check .`
```bash
git add spark_code/instructions.py tests/test_instructions.py spark_code/cli.py
git commit -m "feat: load CLAUDE.md hierarchy alongside SPARK.md (Claude Code parity)"
```

- [ ] **Step 7: `/init` command**

Add slash command `/init` in cli.py's command dispatch (pattern: any `elif command == "/..."` block) + autocomplete entry in `spark_code/ui/input.py` (the `SLASH_COMMANDS` dict at input.py:~50). Behavior: if `CLAUDE.md` exists in cwd, print a note and stop. Otherwise send this one-shot prompt through the EXISTING agent object (same path a normal user message takes), so the model writes the file with its own tools:

```
Analyze this project and create a CLAUDE.md file in the project root.
Include: what the project is, build/test/run commands you can verify from
project files, code style conventions you can observe, and any directory
layout worth knowing. Keep it under 60 lines. Write the file with write_file.
```

Test (in `tests/test_cli_helpers.py`, following its existing patterns): `/init` with an existing CLAUDE.md prints the note and does NOT invoke the agent (patch the agent run; assert not called).

- [ ] **Step 8: Full gates + commit**

```bash
git add spark_code/cli.py spark_code/ui/input.py tests/test_cli_helpers.py
git commit -m "feat: /init generates CLAUDE.md via the agent"
```

---

### Task 2: `@`-file mentions

**Files:**
- Modify: `spark_code/cli.py:378` (`_detect_file_mentions`) and its call site at cli.py:2967-2968
- Modify: `spark_code/ui/input.py:134` (`FilePathCompleter`) — trigger on `@` mid-line
- Test: `tests/test_cli_helpers.py` (mention parsing), `tests/test_input_completer.py` (completer)

**Interfaces:**
- Consumes: existing `_detect_file_mentions(text: str) -> list[str]` and the auto-read wiring that feeds its results into context.
- Produces: `@path` tokens are stripped of `@` and returned by `_detect_file_mentions` with highest priority; `@dir/` mentions of a directory return the directory path (the existing auto-read path renders directories as listings — verify with `grep -n "mentioned" spark_code/cli.py` how results are consumed; if directories are currently skipped, route them through the `list_dir` tool's function).

- [ ] **Step 1: Write failing tests for mention parsing**

```python
# append to tests/test_cli_helpers.py
from spark_code.cli import _detect_file_mentions


def test_at_mention_extracted(tmp_path, monkeypatch):
    (tmp_path / "auth.py").write_text("x = 1")
    monkeypatch.chdir(tmp_path)
    got = _detect_file_mentions("fix the bug in @auth.py please")
    assert "auth.py" in got


def test_at_mention_with_path(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    monkeypatch.chdir(tmp_path)
    got = _detect_file_mentions("look at @src/app.py")
    assert "src/app.py" in got


def test_at_mention_directory(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    monkeypatch.chdir(tmp_path)
    got = _detect_file_mentions("what is in @src")
    assert "src" in got


def test_email_not_treated_as_mention(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = _detect_file_mentions("email me at alexandra@gmail.com")
    assert "gmail.com" not in got
```

- [ ] **Step 2: Run, verify the new tests fail** (`.venv/bin/python -m pytest tests/test_cli_helpers.py -q`; existing behavior likely ignores `@`).

- [ ] **Step 3: Implement**

In `_detect_file_mentions`: before the existing extraction, find `@`-prefixed tokens with `re.findall(r"(?:^|\s)@([\w./-]+)", text)`; the `(?:^|\s)` guard excludes emails (`user@host` has no leading whitespace before `@`... it does after "at " — note `alexandra@gmail.com` the `@` is mid-token so the guard rejects it). For each token: include it if `os.path.exists(token)` (file OR directory — existing code may filter to files; loosen for these tokens only). Keep all existing detection behavior unchanged; `@` results go first in the returned list.

Completer: in `FilePathCompleter.get_completions` (input.py:142), also activate when the word before the cursor starts with `@` — complete the path portion after `@` and re-prepend `@` in the completion text. Add a test to `tests/test_input_completer.py` following its existing document-fixture pattern: text `"fix @sr"` with a `src/` dir in cwd offers `@src/`.

- [ ] **Step 4: Verify directory mentions render.** Check how cli.py:2967's consumer turns paths into context (read the ~10 lines after it). If it calls read_file only, add a directory branch that injects the `list_dir` result instead. Cover with a quick test if you had to change the consumer.

- [ ] **Step 5: Full gates + commit**

```bash
git add spark_code/cli.py spark_code/ui/input.py tests/test_cli_helpers.py tests/test_input_completer.py
git commit -m "feat: @-file mentions with autocomplete (Claude Code parity)"
```

---

### Task 3: Esc to interrupt generation

**Files:**
- Create: `spark_code/ui/esc_watcher.py`
- Create: `tests/test_esc_watcher.py`
- Modify: `spark_code/cli.py:2324-2363` (`_run_with_notify` — the existing SIGINT + ISIG machinery)

**Interfaces:**
- Consumes: the cancel routine used by `_cancel_on_sigint` (cli.py:2339 region) — reuse the SAME routine so Esc and Ctrl+C are one code path; `Agent.cancel()` (agent.py:180).
- Produces: `EscWatcher(on_escape: Callable[[], None])` context manager: `with EscWatcher(cancel_fn): await coro`. Also `classify_escape(buf: bytes) -> bool` (pure): True only for a LONE 0x1b, False when 0x1b is followed by more bytes within the grace window (arrow keys/function keys send `\x1b[...`).

**Design (read before coding):** during generation the terminal is NOT inside prompt_toolkit — a raw reader is safe. The watcher thread puts the tty in cbreak (`tty.setcbreak`, saving/restoring termios attrs), `select()`s stdin with a 50ms timeout in a loop, and on reading 0x1b waits one extra 50ms tick to see whether continuation bytes arrive (escape sequences); a lone Esc fires `on_escape` once. The thread exits when the context manager exits (an Event). CRITICAL interplay: `_run_with_notify` already mutates termios to force ISIG (cli.py:2341-2349) — the watcher must start AFTER that mutation and restore attrs BEFORE `_run_with_notify`'s own cleanup, i.e. innermost scope. cbreak keeps ISIG on, so Ctrl+C still delivers SIGINT while the watcher runs. If stdin is not a tty (`os.isatty(0)` false — one-shot/piped), EscWatcher is a no-op.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_esc_watcher.py
import os
import threading
import time

from spark_code.ui.esc_watcher import EscWatcher, classify_escape


def test_lone_escape_is_interrupt():
    assert classify_escape(b"\x1b") is True


def test_escape_sequence_is_not_interrupt():
    assert classify_escape(b"\x1b[A") is False   # up arrow
    assert classify_escape(b"\x1bOP") is False   # F1


def test_watcher_fires_on_lone_escape_from_pipe():
    r, w = os.pipe()
    fired = threading.Event()
    watcher = EscWatcher(fired.set, fd=r, raw_mode=False)
    with watcher:
        os.write(w, b"\x1b")
        assert fired.wait(timeout=2.0)
    os.close(r); os.close(w)


def test_watcher_ignores_arrow_key():
    r, w = os.pipe()
    fired = threading.Event()
    with EscWatcher(fired.set, fd=r, raw_mode=False):
        os.write(w, b"\x1b[A")
        time.sleep(0.3)
        assert not fired.is_set()
    os.close(r); os.close(w)


def test_noop_on_non_tty():
    # default fd=0 under pytest is not a tty; enter/exit must not raise
    with EscWatcher(lambda: None):
        pass
```

- [ ] **Step 2: Run, verify failure** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `spark_code/ui/esc_watcher.py`**

```python
"""Esc-to-interrupt: raw stdin watcher active only while the model generates."""

from __future__ import annotations

import os
import select
import threading

_TICK = 0.05  # seconds; also the grace window for escape-sequence continuation


def classify_escape(buf: bytes) -> bool:
    """True iff ``buf`` is a lone Esc keypress (not an escape sequence)."""
    return buf == b"\x1b"


class EscWatcher:
    """Context manager: fires ``on_escape`` once if the user presses Esc.

    ``raw_mode=False`` (tests) skips termios manipulation and tty checks,
    reading the given fd directly. In real use (fd=0, raw_mode=True) the
    watcher no-ops unless stdin is a tty, puts it in cbreak for the duration
    (ISIG stays on, so Ctrl+C keeps working), and restores attrs on exit.
    """

    def __init__(self, on_escape, fd: int = 0, raw_mode: bool = True):
        self._on_escape = on_escape
        self._fd = fd
        self._raw_mode = raw_mode
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved_attrs = None

    def __enter__(self):
        if self._raw_mode:
            if not os.isatty(self._fd):
                return self
            import termios
            import tty
            self._saved_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._saved_attrs is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
        return False

    def _watch(self):
        fired = False
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], _TICK)
            if not ready:
                continue
            buf = os.read(self._fd, 32)
            if not buf.startswith(b"\x1b"):
                continue  # swallow stray typing during generation
            if buf == b"\x1b":
                ready, _, _ = select.select([self._fd], [], [], _TICK)
                if ready:
                    buf += os.read(self._fd, 32)
            if classify_escape(buf) and not fired:
                fired = True
                self._on_escape()
```

- [ ] **Step 4: Run tests, verify pass.** Note the pipe tests exercise the real thread loop (raw_mode=False path).

- [ ] **Step 5: Wire into `_run_with_notify`**

Inside `_run_with_notify` (cli.py:2324), after the ISIG block (cli.py:2341-2349), wrap the awaited coroutine: build `cancel = <the same routine _cancel_on_sigint invokes>` as a zero-arg closure and use `with EscWatcher(lambda: loop.call_soon_threadsafe(cancel)):` around the `await`. IMPORTANT: the watcher thread must marshal onto the event loop — never call agent/task methods directly from the thread. Read `_cancel_on_sigint`'s body first and extract its logic into a shared closure both handlers call. Update the generation-time status hint (the spinner/toolbar text — `grep -n "Generating" spark_code/cli.py spark_code/ui/*.py`) to read `esc to interrupt` like Claude Code.

- [ ] **Step 6: Live interactive verification (pexpect)**

From a temp dir, drive real spark against the live engine (pattern proven in Phase 0 Task 1): spawn `spark -p llm --trust`, wait for the prompt, send a long-generation request ("count from 1 to 500 slowly, one number per line"), wait ~2s, send `\x1b`, assert the process returns to the prompt within ~5s and a subsequent short message still gets a reply (session not wedged; GPU freed). Save the script as your own scratch (not committed) but paste its transcript into the report.

- [ ] **Step 7: Full gates + commit**

```bash
git add spark_code/ui/esc_watcher.py tests/test_esc_watcher.py spark_code/cli.py
git commit -m "feat: Esc interrupts generation (Claude Code parity)"
```

---

### Task 4: `/resume` picker + `/rewind` unification

**Files:**
- Modify: `spark_code/cli.py` — new `/resume` + `/rewind` commands next to `/history` (cli.py:1426), `/undo` (cli.py:1501), `/checkpoint` (cli.py:1729), `/rollback` (cli.py:1757)
- Modify: `spark_code/ui/input.py` — autocomplete entries (SLASH_COMMANDS dict ~line 50)
- Test: `tests/test_cli_helpers.py`

**Interfaces:**
- Consumes: `/history`'s session-listing machinery (read cli.py:1426's block first — it already builds the sorted session list with labels/ages from `_sessions_dir()`), `Context.load(path)`.
- Produces: `/resume` (no args) = numbered session list (reuse /history's rendering) + a Rich `Prompt.ask` for the number, then loads that session into the live context and reprints the banner; `/resume <n>` skips the prompt. `/rewind` = a 3-option menu: `1 conversation` (delegates to the /undo block's logic), `2 files` (delegates to /checkpoint-apply logic), `3 both`. `/undo`, `/checkpoint`, `/rollback` keep working untouched.

- [ ] **Step 1: Extract the reusable pieces.** Read the `/history`, `/undo`, `/checkpoint`, `/rollback` blocks. Extract `_list_sessions() -> list[dict]` (path, label, age, turns) from /history's body and `_resume_session_into(context, path) -> bool` if not already factored (grep `resume_session` — startup `--resume` already loads a session; reuse that function, do not duplicate). Refactor /history to consume `_list_sessions()` — behavior unchanged.

- [ ] **Step 2: Failing tests**

```python
# append to tests/test_cli_helpers.py
from spark_code.cli import _list_sessions


def test_list_sessions_sorted_newest_first(tmp_path, monkeypatch):
    import spark_code.cli as cli
    monkeypatch.setattr(cli, "_sessions_dir", lambda create=True: str(tmp_path))
    (tmp_path / "20260101_000000_old-task.json").write_text('{"messages": []}')
    (tmp_path / "20260706_120000_new-task.json").write_text('{"messages": []}')
    sessions = _list_sessions()
    assert len(sessions) == 2
    assert "new-task" in sessions[0]["path"]
```

Adjust the session-file JSON shape to whatever `Context.save` actually writes (inspect one file in `~/.spark/history/` or `Context.save` at context.py:370) so the test creates valid fixtures.

- [ ] **Step 3: Implement `/resume` and `/rewind`** per the Produces block. Menus use the codebase's existing Rich prompt style (`grep -n "Prompt.ask" spark_code/cli.py` for the pattern, including the Ctrl+C-safe wrapper if one exists — /clean had an uncaught-Ctrl+C bug historically, don't repeat it: wrap the prompt in try/except KeyboardInterrupt → return). Autocomplete entries: `"/resume": "Pick a past session to resume"`, `"/rewind": "Undo conversation and/or file changes"`.

- [ ] **Step 4: Gates + commit**

```bash
git add spark_code/cli.py spark_code/ui/input.py tests/test_cli_helpers.py
git commit -m "feat: /resume session picker and /rewind unified restore"
```

---

### Task 5: Permission-mode parity + `allows_without_prompt()` extraction

**Files:**
- Modify: `spark_code/permissions.py` (mode aliases + extraction), `spark_code/agent.py:356-361` (parallel-path predicate → use extraction), `spark_code/ui/input.py:89` (cycle order) and the BackTab handler at input.py:251-253, `spark_code/cli.py:972-974` (/mode help text)
- Test: `tests/test_permissions.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: Phase 0's fail-safe non-interactive `check()` semantics (do NOT change them).
- Produces: `PermissionManager.allows_without_prompt(tool_name: str) -> bool` — single source of truth for "no prompt needed", used by BOTH `check()`'s pre-prompt short-circuits and agent.py's parallel-path `authorized` predicate. `MODE_ALIASES = {"default": "ask", "acceptedits": "auto", "bypasspermissions": "trust", "plan": "plan"}` accepted (case-insensitive) everywhere a mode name is parsed (`/mode`, `/config set permissions.mode`, config file load). Shift+Tab cycle becomes `ask → auto → plan → ask` (trust leaves the cycle; reachable via `/trust`, `--trust`, `/mode trust` — matches Claude Code where bypassPermissions is never in the cycle).

**Ordering rule (from docs/superpowers/plans/phase0-deferred.md):** the extraction MUST land before the cycle/alias changes — it is the guard against the two policy copies drifting. Do Step 1-3 (extraction) as its own commit, then aliases/cycle as a second commit.

- [ ] **Step 1: Failing tests for extraction equivalence**

```python
# append to tests/test_permissions.py
import pytest

from spark_code.permissions import PermissionManager


@pytest.mark.parametrize("mode", ["ask", "auto", "trust"])
@pytest.mark.parametrize("tool", ["read_file", "glob", "write_file", "bash"])
def test_allows_without_prompt_matches_noninteractive_check(mode, tool):
    pm = PermissionManager(mode=mode, interactive=False)
    pm2 = PermissionManager(mode=mode, interactive=False)
    assert pm.allows_without_prompt(tool) == pm2.check(tool, {})
```

Adjust the `check()` call signature to the real one (`grep -n "def check" spark_code/permissions.py`) and match how `interactive=False` is actually spelled (Phase 0 added it — verify).

- [ ] **Step 2: Implement extraction.** Move the shared allow-logic (trust / always_allow / session-allow / auto+read-only) into `allows_without_prompt()`; `check()` calls it first (True → allow; False → prompt if interactive, deny if not); replace agent.py:356-361's inline predicate with `permissions.allows_without_prompt(name)`. Existing tests are the safety net — the full suite must stay green with NO test edits in this step (behavior identical by construction).

- [ ] **Step 3: Gates + commit** — `git commit -m "refactor: single allows_without_prompt() policy for both tool paths"`

- [ ] **Step 4: Failing tests for aliases + cycle**

```python
# append to tests/test_permissions.py
from spark_code.permissions import resolve_mode_name


def test_claude_code_mode_aliases():
    assert resolve_mode_name("default") == "ask"
    assert resolve_mode_name("acceptEdits") == "auto"
    assert resolve_mode_name("bypassPermissions") == "trust"
    assert resolve_mode_name("ask") == "ask"          # native names pass through
    assert resolve_mode_name("nonsense") is None
```

And in the input-side test file that covers the cycle (grep `MODE` in tests/; else add to tests/test_input_completer.py): assert the cycle constant equals `["ask", "auto", "plan"]`.

- [ ] **Step 5: Implement.** `resolve_mode_name(name: str) -> str | None` in permissions.py; apply it at every mode-parsing site (`grep -rn '"ask"\|mode <' spark_code/cli.py | grep -i mode` to find them; minimally: `/mode` handler, `--mode`/flag handling if any, config load in config.py). Change the cycle list at input.py:89 to `["ask", "auto", "plan"]` and update the docstring at input.py:253 and the /mode usage string at cli.py:972 to mention both name sets. IMPORTANT: cycling into/out of plan mode must keep whatever enter/exit-plan bookkeeping the current cycle does — read the BackTab handler fully first; Phase 2 owns the plan-mode rebuild, do not touch plan semantics here.

- [ ] **Step 6: Gates + commit** — `git commit -m "feat: Claude Code permission-mode names + Shift+Tab cycle order"`

---

### Task 6: Phase 1 hygiene from phase0-deferred.md

**Files:**
- Modify: `spark_code/cli.py` (startup), `spark_code/model.py:253-254` (stale docstring)
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `fetch_server_max_context` (preflight.py), `resolve_real_model_name` (cli.py — find with grep), startup ping.
- Produces: startup performs ONE `/v1/models` fetch whose parsed payload feeds both the real-model-name banner display and the preflight context-window check (ping stays as-is — it is the health probe). New signature: `fetch_server_models(endpoint, api_key, timeout, transport) -> list[dict] | None` in preflight.py; `fetch_server_max_context` becomes a thin filter over it (public behavior/signature unchanged — existing tests must pass untouched).

- [ ] **Step 1: Failing test**

```python
# append to tests/test_preflight.py
@pytest.mark.asyncio
async def test_fetch_server_models_returns_data_list():
    payload = {"data": [{"id": "m1"}, {"id": "m2", "max_model_len": 32768}]}
    got = await fetch_server_models("http://x:30000", transport=_transport(payload))
    assert [m["id"] for m in got] == ["m1", "m2"]
```

(plus import update at top of the file)

- [ ] **Step 2: Implement**: extract the fetch body into `fetch_server_models`; `fetch_server_max_context` filters it. In cli.py startup, call `fetch_server_models` once, derive both the display name (replacing `resolve_real_model_name`'s own HTTP call — read that function first; keep its name-selection logic, feed it the fetched list) and the preflight warning. Fix the model.py:253-254 docstring to say `real_model_name` is display-only.

- [ ] **Step 3: Gates + commit** — `git commit -m "refactor: single /v1/models fetch feeds banner + preflight; fix stale docstring"`

Note: the `get_messages()` mutating-getter rename stays deferred (load-bearing, tested, cosmetic) — do NOT do it in this phase.

---

### Task 7: Acceptance + merge + push

**Files:** none new.

- [ ] **Step 1:** `.venv/bin/python -m pytest -q` (expect ≥640) and `.venv/bin/ruff check .` — green, tree clean.
- [ ] **Step 2:** `.venv/bin/python scripts/smoke_live.py --provider llm` — 7/7. Then `--provider coder` if `curl -s http://spark-4a54.local:11434/v1/models` lists `qwen3-coder:30b`, else documented SKIP.
- [ ] **Step 3:** Muscle-memory live checklist via pexpect against the live engine, from a temp project containing a `CLAUDE.md` with a distinctive rule (e.g. "Always address the user as Captain"): banner shows `CLAUDE.md loaded`; a reply obeys the rule (proves injection); `@`-mention of a real file gets used without the model reading it via tool; Esc interrupts mid-generation and the session continues; `/resume` lists sessions; Shift+Tab cycles ask→auto→plan in the toolbar. Paste the transcript into the final report.
- [ ] **Step 4:** Whole-branch review gate (controller runs it per subagent-driven-development), then merge to main + `git push origin main`.

---

## Amendment (2026-07-06, post-Task-4 review)

1. The Task 4 `/rewind` menu labels were a plan defect: `/undo` restores FILE
   snapshots (no conversation rewind exists in the codebase). Amended contract:
   menu options are labeled by what they do — `1 last file edits (undo stack)`,
   `2 checkpoint (git stash)`, `3 both`; autocomplete text "Restore files from
   undo stack or checkpoint". True conversation rewind is recorded as a Phase 2
   candidate (context checkpointing), not faked here.
2. `_list_sessions()` must not parse full session JSON for every file in
   ~/.spark/history — metadata reads are capped to the entries actually
   displayed/matched (top-10 slice or the name-matched entry).
