# Phase 2: Agent Brains at 32K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Spark believable on real multi-step work with a local 80B at 32K: a context engine that never lies or wedges, a live todo checklist, subagents as context isolation, true plan mode with an approval gate, a verification habit, a find-the-bugs review swarm, and skills that actually fire.

**Architecture:** Build on what exists: model.py already requests/parses `stream_options.include_usage` (model.py:322, 444-447) — T1 surfaces it; `Context.compact()` and the head+tail `_truncate_result` (agent.py:45) exist — T2 upgrades trigger + summary quality; the dormant global-file `TodoTool` (tools/todo.py) is replaced by a session-scoped statused list with live rendering; `dispatch_agent` composes existing pieces (Agent + Context + build_tools + the Phase-0 non-interactive PermissionManager); plan mode graduates from the `config["_plan_mode"]` flag (cli.py:2508-2526) to registry-level enforcement + an `exit_plan_mode` tool.

**Tech Stack:** Python 3.11, Rich (Panel/Live), prompt_toolkit toolbar hook (`status_callback`, ui/input.py:341-353), pytest (+asyncio auto), ruff, pexpect for live verification.

## Global Constraints

- All edits in `~/spark-code`. vLLM alias `qwen3.5:122b` on `http://spark-4a54.local:30000` is a hard contract; `coder` = Ollama `qwen3-coder:30b` on `:11434`. Never key model behavior on model names.
- Gates before every commit: `.venv/bin/python -m pytest -q` (697 green at start) and `.venv/bin/ruff check .` (0 at start).
- Existing commands/flags keep working; parity features are additions.
- **Fixed prompt overhead budget:** system prompt + tool schemas + instructions + todo state must stay under ~6,000 tokens (spec risk item). Any task that grows the system prompt measures before/after with `Context.estimate_tokens()` and reports the delta.
- **Prefix-cache rule (spec):** system-prompt content must be ordered static-first (identical text across turns first; volatile content — cwd, git status, todo state — last). Any task touching system-prompt assembly preserves/improves this ordering and says so in its report.
- Subagent concurrency: `agents.max_concurrent` config, default 3.
- Branch: `phase2/agent-brains` off main (b932fcd). Live acceptance at phase end: smoke 7/7 both engines + the P2 live checklist + the DOGFOOD test (spec's final gate).
- Commits: small, per-feature, `feat:`/`fix:`/`refactor:` style.

---

### Task 1: Real context meter (server-reported usage → toolbar)

**Files:**
- Modify: `spark_code/model.py` (usage already parsed at :444-451 — verify variable flow), `spark_code/agent.py` (surface usage into Context after each request), `spark_code/context.py`, `spark_code/cli.py` (toolbar status_callback), `spark_code/ui/input.py` only if the toolbar hook needs a new field
- Test: `tests/test_context.py`, `tests/test_agent.py`

**Interfaces:**
- Produces: `Context.record_usage(prompt_tokens: int)` (stores `last_prompt_tokens: int | None`); `Context.context_left(window: int) -> float` returning 1.0..0.0 — uses `last_prompt_tokens` when known, else `estimate_tokens()`; toolbar renders `ctx {int(pct*100)}%` colored green >40%, yellow 15-40%, red <15%. T2 consumes `context_left` for its trigger.

- [ ] **Step 1: Failing tests** — `record_usage` stores; `context_left(32768)` uses recorded value (`record_usage(16384)` → 0.5); falls back to estimate when never recorded; recorded value resets to None after `compact()` and `clear()` (stale usage must not survive a context change).

```python
# append to tests/test_context.py
def test_context_left_uses_server_usage():
    ctx = Context()
    ctx.record_usage(16384)
    assert abs(ctx.context_left(32768) - 0.5) < 0.01

def test_context_left_falls_back_to_estimate():
    ctx = Context()
    ctx.add_user("hi")
    assert ctx.context_left(32768) > 0.9

def test_recorded_usage_cleared_on_compact_and_clear():
    ctx = Context()
    ctx.record_usage(30000)
    ctx.clear()
    assert ctx.last_prompt_tokens is None
```

- [ ] **Step 2:** Run, verify fail. **Step 3:** Implement in context.py. **Step 4:** Wire agent.py: where the streaming loop receives the `done` chunk's usage (trace from model.py:444's `_req_input` — find where it reaches agent.py; the July-2 fix round wired usage into stats — reuse that path), call `self.context.record_usage(prompt_tokens)`. **Step 5:** Toolbar: in cli.py's `status_callback` (find: `grep -n "status_callback" spark_code/cli.py`), append the `ctx N%` segment using the active provider's `context_window`. **Step 6:** Full gates + commit `feat: real context-left meter from server-reported usage`.

---

### Task 2: Auto-compaction upgrade (model summary, real trigger, /compact instructions)

**Files:**
- Modify: `spark_code/context.py` (`compact()` at :219+), `spark_code/agent.py` (trigger at :223), `spark_code/cli.py` (`/compact` handler at :877)
- Test: `tests/test_context.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `Context.context_left(window)` from T1.
- Produces: `Context.compact(keep_recent: int = 6, summary: str | None = None) -> str | None` — when `summary` is provided it replaces the internally-generated digest (mechanical fallback stays for offline/test use); `Agent._maybe_compact()` — called at round boundaries, triggers when `context_left(window) < 0.25`, generates the summary BY ASKING THE MODEL (one non-streamed request: "Summarize this conversation so far for your own future reference. Preserve: task goal, key decisions, file paths touched, unresolved items. Under 300 words.") then calls `compact(summary=...)`; `/compact <instructions>` passes user instructions into the summary request. Boundary safety (`_safe_split_index`) and pinned-file survival are EXISTING behavior — regression tests must prove they still hold with model summaries.

- [ ] **Step 1: Read first.** `Context.compact()` + `_safe_split_index` (context.py:219-340), the trigger site (agent.py:223 — what threshold does it use today?), pinned re-injection (cli.py — `apply_to_prompt`), and how `model._blocking_request` works (model.py:585+) for the summary call.
- [ ] **Step 2: Failing tests** — `compact(summary="CUSTOM")` uses the provided text verbatim in the summary message; trigger math: fake context with `record_usage(int(32768*0.8))` → `_maybe_compact` fires; `record_usage(int(32768*0.5))` → doesn't; summary-request failure (model raises) → falls back to mechanical digest, session continues (never wedge); tool-boundary safety still holds (compact right after add_assistant_tool_calls + add_tool_result leaves a valid sequence — extend the existing Phase-0 regression tests).
- [ ] **Step 3-4:** Implement + pass. The model-summary call goes through the agent's existing model client (non-streamed); it must NOT recurse (`_maybe_compact` sets a guard flag while running).
- [ ] **Step 5:** `/compact focus on the database work` → instructions appended to the summary request. Notification line matches existing style (`⚡ Context compacted…`).
- [ ] **Step 6:** Per-tool result budgets: `_truncate_result` (agent.py:45) gains an optional per-tool limit map — `{"read_file": 10000, "bash": 15000, "grep": 8000, "dispatch_agent": 8000}` (module const, config-overridable via `tools.result_budgets`), default stays 15000. Head+tail semantics unchanged. Test: a 30K grep result truncates to ~8K keeping head+tail.
- [ ] **Step 7:** Prefix-cache pass: read the system-prompt assembly (context.py SYSTEM_PROMPT + cli.py injections: instructions, skills list, platform info, pinned files, memory) and reorder so static content precedes volatile; report the before/after order + estimate_tokens delta. **Step 8:** Full gates + commit `feat: model-generated auto-compaction with real-usage trigger`.

---

### Task 3: Live session todo list

**Files:**
- Create: `spark_code/todos.py`, `tests/test_todos.py`
- Modify: `spark_code/tools/__init__.py` or wherever `build_tools()` lives (cli.py:388) to register the tool; `spark_code/ui/output.py` (renderer); `spark_code/context.py` (system-prompt nudge text); DELETE `spark_code/tools/todo.py` (dormant global-file tool, never registered — remove, don't port).
- Test: `tests/test_todos.py`

**Interfaces:**
- Produces: `TodoList` (session-scoped, in-memory): `set_items(items: list[dict]) -> str` where each item is `{"content": str, "status": "pending"|"in_progress"|"completed"}` (full-list replace semantics, like Claude Code's TodoWrite); `items` property; validation: max 20 items, at most one in_progress, unknown status → error string not exception. `TodoWriteTool(todo_list)` tool named `todo_write`, `is_read_only = True` (it never touches disk), schema mirroring set_items. `render_todos(console, items)` in ui/output.py: checklist panel — `✓` completed (dim), `▸` in_progress (bold), `○` pending. Agent renders the panel after each successful todo_write call (wire in the tool-result display path, output.py:220's render_tool_result — special-case todo_write).
- System-prompt nudge (append to the STATIC section, count the tokens): "For any task with 3 or more steps, maintain a todo list with the todo_write tool: set statuses as you start and finish each step. Keep exactly one item in_progress."

- [ ] **Step 1: Failing tests** — set_items replaces wholesale; one-in_progress rule enforced; max 20; render_todos output contains the three markers; tool registered in build_tools and schema valid; todo state is NOT persisted (fresh session → empty).
- [ ] **Step 2-4:** Implement, pass. The TodoList instance lives per-session (created in run_interactive alongside Context; one-shot gets one too).
- [ ] **Step 5:** Delete tools/todo.py + its imports; full gates; commit `feat: live session todo checklist (todo_write)`. Report the system-prompt token delta.

---

### Task 4: dispatch_agent — subagents as context isolation

**Files:**
- Create: `spark_code/dispatch.py`, `tests/test_dispatch.py`
- Modify: `build_tools()` registration; `spark_code/agent.py` only if the tool needs loop access (prefer not: the tool owns its sub-agent)
- Test: `tests/test_dispatch.py`

**Interfaces:**
- Produces: `DispatchAgentTool` named `dispatch_agent`, schema: `{prompt: str, agent_type: "explore"|"reviewer"|"implementer"}`. Behavior:
  - Builds a FRESH `Context` (own system prompt: a compact sub-agent prompt stating "your final message is returned to the caller as data — report findings, not conversation"), fresh tool registry filtered by type: explore/reviewer = read-only tools only (`tool.is_read_only`); implementer = full registry MINUS dispatch_agent (no nesting — depth guard).
  - Shares the lead's `ModelClient` (same engine; concurrent requests are fine) but a NEW `PermissionManager(mode=<lead's mode>, interactive=False)` — the Phase-0 fail-safe path.
  - Runs `await sub_agent.run(prompt)` with a round cap of 15 (config `agents.max_rounds`); returns the sub-agent's final text truncated to 8,000 chars head+tail via the existing `_truncate_result`.
  - `is_read_only = True` for explore/reviewer dispatches (parallel-path eligible), `False` for implementer.
  - Concurrency: a module-level `asyncio.Semaphore(agents.max_concurrent)` (config, default 3) wraps the sub-agent run.
  - Errors inside the sub-agent → returned as `"[dispatch_agent error] ..."` string, never an exception into the lead's loop.
- Config: `agents: {max_concurrent: 3, max_rounds: 15}` respected from `~/.spark/config.yaml` with those defaults.

- [ ] **Step 1: Failing tests** (mock ModelClient with a scripted stream, pattern exists in tests/test_agent.py): explore dispatch gets ONLY read-only tools (assert write_file absent from sub-registry); implementer registry lacks dispatch_agent; sub-agent final text returned; >8000-char result truncated with head+tail; sub-agent exception → error string; semaphore limits concurrent runs (launch 5 with a slow mock, assert ≤3 in flight via a counter); lead's context untouched by the sub-run (message count unchanged).
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Register; system prompt STATIC section gains one line: "Use dispatch_agent for open-ended codebase searches, reviews, or research — it runs in a fresh context and returns a summary, keeping your own context small." **Step 6:** Live verification (pexpect, real engine, temp project with a few source files): "Use dispatch_agent to find which file defines GREETING and report the value" → correct answer, lead context stays small (check /tokens or /stats output before/after). **Step 7:** Gates + commit `feat: dispatch_agent subagents with fresh-context isolation`.

---

### Task 5: True plan mode

**Files:**
- Create: `tests/test_plan_mode.py`
- Modify: `spark_code/permissions.py` OR `spark_code/agent.py` (enforcement — see design), `spark_code/cli.py` (plan flag sites :917, :1226-1244, :2508-2526; approval flow), `build_tools()` (exit_plan_mode tool), `spark_code/context.py` (plan-mode system nudge)

**Design (read before coding):** enforcement lives at tool-execution time, not registry construction (registry is built once per session): in `Agent`, before executing any tool, if plan mode is active and `not tool.is_read_only` and tool name != "exit_plan_mode" → return denial result "Plan mode: write actions are blocked. Present your plan with exit_plan_mode." The `plan_mode` state moves from the loose `config["_plan_mode"]` dict-flag into a small shared object the CLI and Agent both hold (`PlanState` with `.active: bool`) — keep the existing `_set_plan`/toolbar displays working. Remove the old prompt-wrapping (find it: `grep -n "plan" spark_code/cli.py | grep -i wrap` and the ≤3-word skip logic if still present).

**Interfaces:**
- Produces: `ExitPlanModeTool` named `exit_plan_mode`, schema `{plan: str}`, `is_read_only = True`. When called in plan mode: renders the plan as a Rich Panel ("Ready to code?"), then a Ctrl+C-safe `Prompt.ask` — approve → `plan_state.active = False`, `permissions.mode = "auto"`, tool returns "Plan approved. Proceed."; reject (with optional feedback) → stays in plan mode, returns "Plan rejected: <feedback>. Revise and present again." Called OUTSIDE plan mode → returns "Not in plan mode."
- Plan-mode system nudge (injected as a system message on ENTERING plan mode, removed pattern: like round-warnings): "You are in plan mode: read-only research, then present a plan via exit_plan_mode. Do not attempt writes."
- Shift+Tab and `/plan` keep working through `PlanState`.

- [ ] **Step 1: Failing tests** — write_file blocked in plan mode with the denial text; read_file allowed; exit_plan_mode approval flips state + mode to auto (mock Prompt.ask); rejection keeps plan mode + returns feedback; exit_plan_mode outside plan mode is a no-op message; Shift+Tab into plan then approval leaves permissions.mode == "auto".
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Live pexpect: enter plan mode, ask for a small change, model researches (read tools succeed), attempts or presents plan → approve → file gets written in the same turn or next. Report transcript. **Step 6:** Gates + commit `feat: true plan mode — enforced read-only + exit_plan_mode approval gate`.

---

### Task 6: Verification habit

**Files:**
- Modify: `spark_code/context.py` (STATIC system-prompt rule), `spark_code/agent.py` (round-boundary nudge), `spark_code/project_detect.py` (test-command detection if absent — check first: `grep -n "test" spark_code/project_detect.py`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Produces: `detect_test_command(cwd: str, instructions_text: str) -> str | None` — from project type (pytest/npm test/xcodebuild/cargo test/go test) with a CLAUDE.md/SPARK.md override (regex a line like "test: <cmd>" or a ```command in a Testing section — keep it simple: look for the literal patterns `pytest`, `npm test`, etc. in the instructions text first). Agent tracks per-turn: did any write/edit tool succeed AND has no bash command containing the test command run since? If so, when the model produces a final answer (no tool calls), inject ONE system nudge — "You changed files but have not run the project's tests (`<cmd>`). Run them before concluding, or state explicitly why not." — and continue the loop ONCE (a `_verify_nudged` flag prevents repeats; the flag resets per user turn).
- STATIC rule text: "Before declaring code work done, run the project's tests/build and report the result."

- [ ] **Step 1: Failing tests** — detect_test_command for a pyproject project → contains "pytest"; nudge fires exactly once (scripted mock stream: write_file round → final answer → assert nudge message appended and loop continued → second final answer ends turn); no nudge when tests were run (bash tool call containing "pytest" in the turn); no nudge on read-only turns.
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Gates + commit `feat: verification nudge — run tests before claiming done`. Report system-prompt token delta.

---

### Task 7: /review agent swarm

**Files:**
- Create: `spark_code/review.py`, `tests/test_review.py`
- Modify: `spark_code/cli.py` (/review command — NOTE a builtin `review` skill exists at skills/base.py:257; the command supersedes it: keep the skill but /review routes to the swarm), autocomplete entry

**Interfaces:**
- Consumes: `DispatchAgentTool`'s machinery from T4 — refactor its core into `run_subagent(model, prompt, agent_type, config) -> str` in dispatch.py so review.py calls it directly (the tool becomes a thin wrapper). Update T4's tests accordingly if this task lands after (it will).
- Produces: `/review [target]` — target: default = `git diff HEAD` (working tree changes; if clean, last commit `git diff HEAD~1..HEAD`), or an explicit path. Flow: (1) collect diff text (cap 20,000 chars head+tail); (2) run 3 reviewer subagents CONCURRENTLY (asyncio.gather under the T4 semaphore) with distinct lens prompts — correctness/bugs, edge cases + error handling, project conventions (conventions lens gets the instructions text) — each returns findings as lines `SEVERITY|file:line|description`; (3) parse findings (tolerant parser; unparseable lines kept as raw notes); (4) skeptic pass: each finding goes to one reviewer subagent with "Try to refute this finding against the code" → verdict CONFIRMED/REFUTED; (5) render: confirmed findings grouped by severity in a panel, refuted count shown as one dim line. Empty diff → friendly message, no dispatches.
- Config: `review: {auto: false}` — when true, after any agent turn where a write/edit tool succeeded, run the swarm on the turn's changed files automatically (reuse the same entry point; keep OFF by default).

- [ ] **Step 1: Failing tests** (mock run_subagent): three lenses each called once with the diff; findings parsed; skeptic refutes → excluded from output; empty diff short-circuits; auto-review fires only when config true + files changed.
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Live: make a deliberate off-by-one in a temp repo file, `/review` → at least one confirmed correctness finding names the file. Report transcript + wall time. **Step 6:** Gates + commit `feat: /review multi-agent swarm with skeptic verification`.

---

### Task 8: Skills that fire on an 80B

**Files:**
- Modify: `spark_code/skills/base.py` (registry — verify how skills inject today: `grep -rn "skill" spark_code/context.py spark_code/cli.py | head`), `spark_code/cli.py` (slash fallback), `spark_code/context.py` (trigger-hint line)
- Test: `tests/test_skills_claude_bridge.py` (extend)

**Interfaces:**
- Produces: (1) **Slash invocation**: unknown `/name [args]` in handle_slash_command falls through to `skills.get(name)` — if found, the skill's `get_prompt(args)` becomes the user message for this turn (return it the way `/init` returns its prompt). Autocomplete: skill names appear in slash completion (source from registry, capped at reasonable count). (2) **Budget-aware injection**: PROVE (test) that the system prompt carries only a compact skill INDEX (name + one-line description each, capped total 1,500 chars — truncate the list beyond, prefer builtins first), never full skill bodies; full text enters context only on invocation. If today's code preloads bodies, fix it. (3) **Trigger hint** (STATIC): "Available skills are listed above. When a user request matches a skill's description, invoke it by replying with exactly `/<skill-name> <args>` on its own line — it will be expanded for you." — and handle that reply shape: if a model final answer consists solely of a slash-skill line, expand and continue the turn (one hop max, guard flag).
- [ ] **Step 1: Failing tests** — `/commit` (builtin skill) expands to its prompt; unknown `/nope` still errors; system prompt contains skill names but NOT bodies (pick a bridged skill with a long body; assert a distinctive body phrase absent + name present); index cap enforced; model-reply `/codesearch foo` auto-expands once (scripted stream), guard prevents loops.
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Gates + commit `feat: 80B-friendly skills — slash invocation, lean index, trigger hints`. Report system-prompt token delta (this is the task most likely to blow the 6K budget — measure!).

---

### Task 9: Acceptance — P2 live checklist + DOGFOOD + merge + push

**Files:** none new (scripts/scratch only).

- [ ] **Step 1:** Full gates: pytest (expect ≥760), ruff 0, tree clean.
- [ ] **Step 2:** `scripts/smoke_live.py --provider llm` 7/7; `--provider coder` 7/7 or documented SKIP.
- [ ] **Step 3: P2 live checklist** (pexpect vs real engine, fresh temp project): ctx% appears in toolbar (or /tokens shows server-derived numbers); a 3-step task renders a todo checklist that updates; `dispatch_agent` search returns a correct answer; plan mode blocks a write + approval gate flow works end-to-end; `/review` on a seeded bug yields a confirmed finding; `/commit` skill invocation works. Each item PASS/FAIL + evidence.
- [ ] **Step 4: THE DOGFOOD GATE (spec's final acceptance):** in a REAL copy of a real Python project (fresh git clone into temp of any small repo with tests — or generate a realistic ~5-file project with a test suite), run interactive `spark -p llm` (default ask mode is fine to exercise prompts, --auto acceptable) and give it ONE realistic feature request ("add a --json flag to the CLI that prints results as JSON, with a test"). Success = feature implemented + tests written + tests pass (verified by the harness re-running pytest itself) without manual code edits. Two attempts allowed (record both). This is the spec's definition of done for the daily-driver bar.
- [ ] **Step 5:** Whole-branch final review (controller dispatches per subagent-driven-development), fix round if needed, then merge to main + `git push origin main`.
