# Spark Code = Claude Code at 32K — Design

**Date:** 2026-07-06
**Status:** Approved by Alexandra (2026-07-06)
**Goal owner:** Alexandra

## Purpose

Spark Code is the fallback daily driver for months when the Claude API/subscription
budget is unavailable. The bar: drop into `spark` in the same repo and have hands,
habits, and workflow *just work* — Claude Code's experience, running on the local
vLLM 80B (Qwen3-Coder-Next int4+ngram, served as `qwen3.5:122b` on
`spark-4a54.local:30000`, 32K context) with Ollama 30B as the secondary profile.

"100% like Claude Code" means **parity of experience re-engineered for a 32K local
budget**, not a copy of internals designed for 200K context. Every parity feature
gets a "local budget" interpretation, stated explicitly in this spec.

## Current state (verified 2026-07-06)

- Repo: `~/spark-code` (the editable-install dir spark runs from;
  `~/CodingProjects/spark-code` is a symlink to it). `main` at `1f61f34`, clean tree.
- `1f61f34` = July 2 audit round: ~55 bugs fixed, 572 tests green, ruff clean.
  **Never smoke-tested against the live engine. Never pushed to GitHub.**
- 54 slash commands exist, including `/compact`, `/clear`, `/model`, `/memory`,
  `/plan`, `/continue`, `/history`, `/undo`, `/checkpoint`, `/rollback`.
- Claude Code skills + plugin skills + memory bridges already load from `~/.claude/`.
- Confirmed gaps (grep-verified): no CLAUDE.md support anywhere, no `@`-file
  mentions, no Esc-to-interrupt (Ctrl+C only), `TodoTool` exists at
  `spark_code/tools/todo.py` but is never registered, plan mode is prompt-wrapping
  (not read-only enforcement) with no approval gate.
- Config drift: `~/.spark/config.yaml` `llm` provider has `context_window: 16384`
  but the server runs `--max-model-len 32768`. `worker_model: qwen3.5:122b` +
  `max_workers: 1` re-enabled despite known single-GPU worker flakiness.

## Constraints (design center, not footnotes)

1. **32K context total.** System prompt + tools + history + files + generation all
   fit in 32,768 tokens. Budget math appears in every Phase 2 decision.
2. **One shared engine.** JARVIS, Claude UI, and Spark Code all hit the same vLLM
   instance concurrently. Spark must not assume exclusive throughput, and the
   `qwen3.5:122b` served-name alias is a hard contract (never "fix" it).
3. **~70–80 tok/s.** Long generations are expensive; features should minimize
   wasted regeneration (interrupt must actually stop GPU work; retries must not
   duplicate streamed text).
4. **The model is an 80B, not Opus.** It wanders off-task, over-explores, and
   claims success without verifying. Features that impose structure (todo list,
   verification nudges, scoped subagents) are quality levers, not cosmetics.

## Phases

### Phase 0 — Trust the foundation

Deliverable: current feature set proven working against the real engine, pushed.

- Fresh audit pass over current `main` (the July 2 fixes have never run live;
  review with fresh eyes rather than assuming the fix-round landed correctly).
- **Live smoke suite** — a scripted, repeatable end-to-end run against vLLM
  `:30000`: connect/ping → read file → edit file (diff preview) → bash with
  streaming → a 3+ tool-round task → `/compact` mid-task → session save →
  `--resume`. Also run the suite against the `coder` (Ollama 30B :11434) profile.
  Lives in `scripts/smoke_live.py` (or similar); runnable any time the engine is up.
- Config repair: `context_window: 32768` for `llm`; review `max_tokens`;
  re-evaluate `worker_model`/`max_workers` against current single-GPU reality.
- Fix everything found; each fix gets a regression test.
- `git push` main to `alexandra767/spark-code` once green.

### Phase 1 — Drop-in muscle memory

Deliverable: hands don't know the difference for the core loop.

- **CLAUDE.md loading.** At startup, merge into the system prompt: project
  `CLAUDE.md` (walk up from cwd like Claude Code does), `~/.claude/CLAUDE.md`
  (user-global), and existing `SPARK.md`. Precedence on conflict: SPARK.md >
  project CLAUDE.md > global CLAUDE.md. Shown in the startup banner like the
  current SPARK.md indicator. `/init` generates a starter CLAUDE.md (or appends
  Spark section to an existing one) by analyzing the project.
- **@-file mentions.** `@` in the input triggers the existing FilePathCompleter;
  on submit, `@path` tokens are resolved and routed through the existing
  auto-read machinery (mentioned file content enters context, subject to the
  same truncation budgets). Works for directories (listing) like Claude Code.
- **Esc to interrupt.** Esc during generation = interrupt (primary), keeping
  Ctrl+C as fallback. Must reuse the existing three-layer cancel path AND close
  the response stream so vLLM stops generating (GPU-stop is part of the
  acceptance test). Toolbar shows "esc to interrupt" during generation.
- **Command parity.** `/resume` opens a session picker (wraps existing history
  machinery). `/rewind` becomes the single checkpoint/restore entry (subsumes
  `/undo`, `/checkpoint`, `/rollback`, which remain as aliases). Permission-mode
  names and Shift+Tab cycle order match Claude Code: default → acceptEdits →
  plan → (bypassPermissions only via flag). Existing Spark names stay as aliases.

### Phase 2 — Agent brains at 32K

Deliverable: believable quality on real multi-step tasks.

- **Context engine.**
  - Request `stream_options.include_usage`; track real server-reported tokens
    (fix the known `choices: []` chunk pitfall when enabling it).
  - Auto-compaction triggers on a % threshold of the *real* window; compaction
    summarizes with the same model, and the cut point always lands on a
    user/assistant-text boundary — never inside a tool call/result exchange
    (the current blind-slice wedges sessions with 400s).
  - Pinned files re-inject after compaction. `/compact <instructions>` supported.
  - Toolbar shows remaining-context %, like Claude Code's statusline.
  - Tool-result hygiene: head+tail truncation that always preserves the tail
    (exit codes, final errors), per-tool output budgets.
- **Live todo list.** Register `TodoTool`; render as a live checklist panel
  (in-progress spinner, completed checkmarks) like Claude Code's; system prompt
  instructs the model to maintain it for any 3+ step task. On-task discipline is
  the point; the UI is the enforcement mechanism.
- **Subagents as context isolation.** A `dispatch_agent` (Task-equivalent) tool:
  spawns a scoped sub-session with a *fresh* 32K window, restricted toolset
  (read-only for explore/search/review agent types; write-capable for
  implementation only when the lead grants it), returns a bounded summary
  (≤ ~2K tokens) to the lead. Sequential by default (shared engine); the
  existing team/worker system remains for explicit `/team` use but the agentic
  path prefers dispatch. This is the 32K reinterpretation of Claude Code's
  subagents: context isolation first, parallelism second.
- **True plan mode.** In plan mode the tool registry itself is filtered to
  read-only (enforced at execution, not by prompt). Model signals plan
  completion via an `exit_plan_mode` tool; the plan renders for approval;
  approve → mode drops to acceptEdits and execution starts; reject → stays in
  plan mode with feedback. Shift+Tab still cycles.
- **Verification habit.** System prompt + round-boundary nudges: before claiming
  done, run the project's test/build command (detected from project type or
  CLAUDE.md). A failed verification re-enters the loop instead of ending the turn.

### Phase 3 — Feel & polish

Deliverable: the remaining texture of Claude Code.

- Streaming markdown rendering quality (code blocks, tables) during generation.
- `patch_stdout` for background printers (worker status, notifications) so the
  input line never mangles.
- Argument-aware autocomplete (`/model <tab>` lists providers, `/resume <tab>`
  lists sessions).
- Session picker UX for `--resume`/`/resume` (arrow-key list with age/label,
  like `claude -r`).
- Banner/toolbar refinements: context %, model real-name unmasking (verify the
  existing uncommitted-then-fixed resolve path live), git branch, mode indicator.

## Non-goals (explicit)

- No cloud/account features (Artifacts, web, teleport, GitHub app).
- No plugin marketplace — the existing `~/.claude` skills bridge already loads
  superpowers/code-review/frontend-design skills.
- No parallel-worker swarm expansion; dispatch-style subagents replace it.
- No MCP transport rewrite beyond verifying the July 2 fixes against a real
  server.
- No new providers; no changes to the vLLM/Ollama server topology (that's
  `~/spark-vllm-docker` territory, separate project).

## Error handling principles

- Non-200 / mid-stream failures raise typed errors; retry only before first
  streamed token; FallbackChain (already wired opt-in) is the failover path —
  verify live rather than re-designing.
- Interrupt (Esc/Ctrl+C) closes the HTTP stream (GPU stops), preserves partial
  response in history, never leaves orphaned tool messages.
- Compaction and truncation must never produce an OpenAI-invalid message
  sequence (tool result without its call, etc.) — this class of bug wedges
  sessions permanently and is treated as P0 wherever found.

## Testing

- Unit/regression: extend the existing pytest suite (572 tests) per feature;
  every audit fix lands with a regression test.
- **Live smoke suite** (Phase 0 deliverable) re-run at the end of every phase,
  against both `llm` (vLLM 80B) and `coder` (Ollama 30B) profiles.
- **Dogfood gate (final acceptance):** one real task on a real repo — implement
  a small feature + tests in a Python project, tests verified passing — completed
  in Spark end-to-end without babysitting. Plus the muscle-memory checklist:
  CLAUDE.md respected, @-mentions work, Esc interrupts (and stops GPU), Shift+Tab
  plan mode with approval gate, mid-task /compact without wedging, /resume works.

## Risks

- **Context budget arithmetic** — system prompt + tool schemas + CLAUDE.md +
  todo + pinned files could crowd a 32K window before work starts. Mitigation:
  measure the fixed overhead in Phase 2 and keep it under ~6K tokens; CLAUDE.md
  files are truncated at a budget like Claude Code does.
- **Qwen tool-calling quirks** — the qwen3_coder parser on vLLM has its own
  failure modes; the smoke suite (not unit tests) is the guard.
- **Shared-engine interference** — smoke runs while JARVIS/Claude UI are live
  are the realistic test condition, not a quiet engine.
- **Two-directory gotcha** — all edits happen in `~/spark-code` (the editable
  install). `~/CodingProjects/spark-code` is a symlink; verify before bulk ops.
