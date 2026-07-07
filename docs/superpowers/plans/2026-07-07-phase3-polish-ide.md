# Phase 3: Feel & Polish + IDE Interoperability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the feel gap with Claude Code and make Spark first-class inside Cursor/VS Code terminals and next to Xcode: safe background printing, argument-aware autocomplete, embedded-terminal correctness, clickable/openable file references, editor diffs, and a Swift-aware verification loop.

**Architecture:** Small additive units on existing seams: `patch_stdout` around the prompt session; dynamic value-providers plugged into the existing `SlashCommandCompleter` (ui/input.py:114); a new `spark_code/editor.py` for host-editor detection + handoff consumed by output rendering and a new `/open` command; targeted fixes where Phase 2 acceptance found real terminal gaps (toolbar invisible in non-CPR ptys). The StreamingMarkdown Live renderer already exists (ui/output.py:445) — it gets gap tests, not a rewrite.

**Tech Stack:** Python 3.11, prompt_toolkit (patch_stdout, completers), Rich (OSC 8 links via `Text.stylize`/`style="link ..."`), pytest, pexpect.

## Global Constraints

- All edits in `~/spark-code`. vLLM alias `qwen3.5:122b` (:30000) hard contract; `coder` = Ollama `qwen3-coder:30b` (:11434).
- Gates before every commit: `.venv/bin/python -m pytest -q` (856 green at start) and `.venv/bin/ruff check .` (0 at start).
- Additive-only: existing commands/flags/rendering keep working; every new behavior has a config off-switch or graceful fallback.
- No new third-party dependencies.
- Editor handoff must NEVER block or crash the CLI: editor launches are fire-and-forget subprocesses with failure swallowed to a dim console note.
- Branch: `phase3/polish-ide` off main (6d2f437). Phase-end acceptance: suite + ruff + smoke 7/7 both engines + focused live checks + merge + push.
- Conscious downscope vs the spec: the arrow-key session picker is NOT built — Phase 1's numbered `/resume` picker plus Task 1's `/resume <tab>` completion covers the workflow; revisit only on real friction.

---

### Task 1: Argument-aware autocomplete

**Files:**
- Modify: `spark_code/ui/input.py` (SlashCommandCompleter at :114)
- Modify: `spark_code/cli.py` (wire value-providers where the completer is constructed — find with `grep -n "SlashCommandCompleter(" spark_code/`)
- Test: `tests/test_input_completer.py`

**Interfaces:**
- Produces: `SlashCommandCompleter(commands=..., value_providers=...)` where `value_providers: dict[str, Callable[[], list[str]]]` maps a command name (e.g. `"/model"`) to a zero-arg callable returning candidate argument strings. When the typed text is `<known-command> <partial-arg>`, completions come from the provider filtered by prefix; providers are called lazily per keystroke and exceptions yield no completions (never crash the completer).
- Wired providers: `/model` + `/providers`-style switching → provider names from config (`config["providers"].keys()`); `/mode` → `["ask","auto","plan","trust","default","acceptEdits","bypassPermissions"]`; `/resume` → session labels from `_list_sessions()` (top 10, name portion only); `/rewind` → `["undo","checkpoint","both"]`.

- [ ] **Step 1: Failing tests**

```python
# append to tests/test_input_completer.py
from prompt_toolkit.document import Document

from spark_code.ui.input import SlashCommandCompleter


def _completions(completer, text):
    doc = Document(text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, None)]


def test_argument_completion_from_provider():
    c = SlashCommandCompleter(value_providers={"/model": lambda: ["llm", "coder", "gemini"]})
    got = _completions(c, "/model co")
    assert any(x.endswith("coder") for x in got)
    assert not any(x.endswith("gemini") for x in got)


def test_argument_completion_lists_all_on_bare_space():
    c = SlashCommandCompleter(value_providers={"/model": lambda: ["llm", "coder"]})
    got = _completions(c, "/model ")
    assert len(got) == 2


def test_provider_exception_yields_no_completions():
    def boom():
        raise RuntimeError("x")
    c = SlashCommandCompleter(value_providers={"/model": boom})
    assert _completions(c, "/model l") == []


def test_command_completion_unchanged_without_arg():
    c = SlashCommandCompleter(value_providers={"/model": lambda: ["llm"]})
    got = _completions(c, "/mod")
    assert any("/model" in x for x in got)
```

- [ ] **Step 2:** Run, verify fail (constructor rejects `value_providers`). **Step 3:** Implement — in `get_completions`, when text splits into a known command + trailing space/partial, consult the provider; completion replaces only the argument portion (compute `start_position` from the partial arg length — the file's own comment at :128-132 documents the past off-by-replacement bug; don't reintroduce it). **Step 4:** Pass. **Step 5:** Wire real providers in cli.py (sessions provider imports `_list_sessions` lazily inside the lambda to avoid import cycles). Full gates. Commit `feat: argument-aware slash autocomplete`.

---

### Task 2: Background-print safety + streaming-markdown gap tests

**Files:**
- Modify: `spark_code/cli.py` (the prompt call inside the REPL — find `session.prompt` / the input loop; wrap with `patch_stdout`)
- Test: `tests/test_polish_features.py` (streaming gaps), manual/pexpect evidence for patch_stdout

**Interfaces:**
- Produces: the interactive input prompt runs under `prompt_toolkit.patch_stdout.patch_stdout(raw=True)` so anything printed by background tasks (worker status timers, notification sounds, future auto-review output) appears ABOVE the input line instead of mangling it. Scope: only around the blocking prompt call, not around generation (Rich Live owns the screen there — patch_stdout during Live causes double-buffering; keep them disjoint).

- [ ] **Step 1: Read first.** The input loop (how `session.prompt` is invoked — likely via `run_in_executor`; patch_stdout must wrap in the SAME thread that calls prompt), and any existing background printers (`grep -n "call_soon_threadsafe\|create_task" spark_code/cli.py | head`). If prompt runs in an executor thread, apply patch_stdout inside the executor callable.
- [ ] **Step 2: StreamingMarkdown gap tests** (unit, no engine): feed the class (ui/output.py:445) chunk sequences that historically break naive renderers — a fenced code block split mid-fence marker (```py chunk-boundary), a table split mid-row, a heading arriving char-by-char — assert no exception and final rendered text contains the full content (use a recording Console). If a case fails, fix minimally in the renderer.
- [ ] **Step 3: patch_stdout wiring** + a pexpect check: start spark, trigger a background print (simplest: a `threading.Timer(2, print, args=["BG-MARKER"])` injected via a temporary debug hook — or use an existing background printer if one fires naturally) while the prompt is idle, assert the marker appears and the prompt line survives (send a command afterward and it executes cleanly). If injecting a hook is too invasive, document the manual verification with a transcript instead.
- [ ] **Step 4:** Full gates. Commit `feat: patch_stdout input safety + streaming renderer gap tests`.

---

### Task 3: Embedded-terminal correctness (the non-CPR gap)

**Files:**
- Modify: `spark_code/ui/input.py` (toolbar/session construction), `spark_code/cli.py` (status fallback)
- Test: `tests/test_input_completer.py` or new `tests/test_terminal_compat.py`

**Background (from Phase 2 acceptance, task-9-checklist.md):** in ptys that don't answer CPR (cursor-position requests), prompt_toolkit printed "WARNING: your terminal doesn't support cursor position requests (CPR)" and the bottom toolbar NEVER rendered — mode/ctx% invisible. Cursor/VS Code integrated terminals DO support CPR, but CI ptys, some SSH setups, and pexpect drivers don't; Spark must degrade gracefully, not silently lose its status surface.

**Interfaces:**
- Produces: `terminal_supports_cpr() -> bool` heuristic in ui/input.py (prompt_toolkit exposes the app/output's `responds_to_cpr` — investigate `prompt_toolkit.output`; if unavailable pre-prompt, detect the condition via env: `TERM=dumb`, missing tty, or a `SPARK_NO_CPR=1` escape hatch). When CPR is unsupported: (a) suppress the noisy warning if prompt_toolkit allows (check `PromptSession` kwargs / `Output` construction), (b) fall back to printing a one-line status (mode + ctx%) ABOVE the prompt after each turn (reuse the toolbar segment builder — extract it so toolbar and fallback share one source), (c) everything else (Esc, Shift+Tab, completion) keeps working — they don't need CPR.
- Config: `ui.statusline: "auto"|"toolbar"|"inline"|"off"` (default auto = toolbar when CPR, inline when not).

- [ ] **Step 1: Investigate** prompt_toolkit's CPR surface (`python -c "import prompt_toolkit; ..."` spikes are fine) and where the warning comes from. Write findings in the report — this task's design depends on what the library exposes; adjust mechanism (not behavior) accordingly.
- [ ] **Step 2: Failing tests** — extract `build_status_segments(...) -> list[tuple[str,str]]` from the toolbar closure (pure function over mode/ctx/provider inputs, testable); test the inline fallback formats the same content as plain text; test `ui.statusline` config routing (auto/inline/off).
- [ ] **Step 3:** Implement + pexpect verification: drive spark in a pexpect pty (which lacks CPR — exactly the acceptance condition), assert the status line appears inline (was: never visible) and no CPR warning spam. **Step 4:** Full gates. Commit `feat: graceful status fallback for non-CPR terminals`.

---

### Task 4: Editor detection + file handoff (`/open`, OSC 8 links)

**Files:**
- Create: `spark_code/editor.py`, `tests/test_editor.py`
- Modify: `spark_code/ui/output.py` (linkify rendered file paths), `spark_code/cli.py` (/open command + autocomplete entry)

**Interfaces:**
- Produces (spark_code/editor.py):
  - `detect_editor(env: dict | None = None, path_checker=shutil.which) -> str | None` — returns one of `"cursor"|"code"|"xed"|None`. Logic: config override wins (caller passes it); else `TERM_PROGRAM == "vscode"` → `"cursor"` if `path_checker("cursor")` else `"code"`; else on macOS with `path_checker("xed")` → `"xed"`; else None. Pure + injectable for tests.
  - `open_in_editor(editor: str, path: str, line: int | None = None) -> bool` — builds the right argv (`cursor|code --goto path:line`, `xed --line <line> <path>`), launches via `subprocess.Popen` detached, returns False (with no exception) on any failure.
  - `file_link(path: str) -> str` — `file://` URL for OSC 8 (absolute, quoted).
- Config: `ui.editor: "auto"|"cursor"|"code"|"xed"|"none"` (default auto).
- `/open <file[:line]>` command: resolves the editor (config+detection), calls open_in_editor, dim note on failure ("No editor detected — set ui.editor"). Autocomplete entry + file-path provider (reuse FilePathCompleter logic via a value-provider from Task 1 if trivial, else plain).
- OSC 8 links: in ui/output.py where tool calls/results render file paths (`_format_tool_args` :104, `_abbreviate_path` :193 — read them), style path text with Rich's `style=f"link {file_link(abs_path)}"` when stdout is a tty AND `ui.editor != "none"`. Cursor/VS Code/iTerm render these clickable; dumb terminals ignore them harmlessly.

- [ ] **Step 1: Failing tests** — detect_editor matrix (vscode+cursor-on-path → cursor; vscode only code → code; no TERM_PROGRAM + xed → xed; nothing → None; env injection, no real which calls); open_in_editor argv shapes for all three (patch Popen, assert argv; failure → False no raise); file_link absolute+encoded space.
- [ ] **Step 2-3:** Implement, pass, wire /open + linkify. Linkify must not break existing rendering tests (recording Console output text unchanged — links are style-layer).
- [ ] **Step 4:** Full gates. Commit `feat: editor detection, /open handoff, OSC 8 file links`.

---

### Task 5: Diff-in-editor option

**Files:**
- Modify: `spark_code/ui/diff.py` (render_diff at :10 / render_inline_diff at :44 — read both first), `spark_code/cli.py` or wherever the edit-permission preview calls them (grep `render_inline_diff` callers)
- Test: `tests/test_editor.py` (extend)

**Interfaces:**
- Consumes: `detect_editor`/`open_in_editor` foundations from Task 4 (import from spark_code.editor).
- Produces: config `ui.diff_in_editor: false` (default). When true AND editor is cursor/code: before the edit permission prompt, write `old` and `new` contents to two temp files (named `<basename>.before`/`<basename>.after` under a mkdtemp) and launch `cursor|code --diff before after` fire-and-forget; the inline terminal diff STILL renders (editor view is additive, not a replacement); temp files cleaned up at session exit (track in a module list, atexit or session teardown). xed has no --diff: option is a no-op with a one-time dim note for xed users.

- [ ] **Step 1: Failing tests** — enabled+code → Popen called with `--diff` + two paths whose contents match old/new; disabled → no Popen; xed → no Popen + note flag set; temp files registered for cleanup.
- [ ] **Step 2-3:** Implement, wire at the preview call site (config read once per session). **Step 4:** Full gates. Commit `feat: optional side-by-side diff in Cursor/VS Code`.

---

### Task 6: Swift/Xcode verification polish

**Files:**
- Modify: `spark_code/project_detect.py` (detect_test_command — Phase 2 added it; read it fully)
- Test: `tests/test_project_detect.py`

**Background:** `detect_test_command` currently returns bare `xcodebuild` for Swift/Xcode projects — not actually runnable without scheme/destination args, so the verification nudge suggests a broken command. The user's iOS projects follow a Makefile convention (`make test` wraps xcodebuild with correct args).

**Interfaces:**
- Produces, in priority order for ALL project types: (1) instructions-text patterns (existing, unchanged); (2) **NEW: a `Makefile` in cwd containing a `test:` target → `make test`**; (3) existing type-specific detection — EXCEPT bare Xcode projects (`.xcodeproj`/`.xcworkspace` without Makefile or Package.swift): return None (nudging an unrunnable command is worse than no nudge). `Package.swift` (SwiftPM) → `swift test` (runnable as-is, better than xcodebuild).

- [ ] **Step 1: Failing tests** — Makefile with test target beats pyproject (`make test`); Makefile without test target ignored (falls through to pytest); bare .xcodeproj → None; Package.swift → `swift test`; instructions override still wins over Makefile.
- [ ] **Step 2-3:** Implement (Makefile scan = regex `^test:` multiline on the file text, read errors→skip), pass. **Step 4:** Full gates. Commit `fix: runnable test commands for Make/SwiftPM/Xcode projects`.

---

### Task 7: Acceptance + merge + push

- [ ] **Step 1:** Full gates: pytest (expect ≥880), ruff 0, tree clean.
- [ ] **Step 2:** `scripts/smoke_live.py --provider llm` 7/7; `--provider coder` 7/7 or documented SKIP.
- [ ] **Step 3: Focused live checks** (pexpect, real engine): (a) non-CPR pty shows the inline status fallback with mode+ctx% and no CPR warning spam; (b) `/model <tab>`-equivalent — completer unit evidence is acceptable if tab-completion can't be driven through pexpect reliably; (c) `/open` on a real file with SPARK-forced `ui.editor=none` → graceful note (don't actually launch her editor); (d) a normal 2-round task still renders streaming markdown + todo panel cleanly (no regression from patch_stdout).
- [ ] **Step 4:** Whole-branch final review (controller dispatches), fix round if needed, merge to main, `git push origin main`.
