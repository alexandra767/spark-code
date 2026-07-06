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
- **Agent review pass (find-the-bugs swarm).** After substantive code changes
  (and on demand via `/review`), reviewer subagents sweep the diff — each with
  a distinct lens (correctness/bugs, edge cases + error handling, project
  conventions), each in its own fresh context window. Findings pass a
  verification step (a skeptic agent tries to refute each finding) before
  they're reported or fixed, so the 80B's false positives die quietly.
  Concurrency: `agents.max_concurrent` config, default 3 — measured live
  2026-07-06: the vLLM engine's KV cache holds 108,256 tokens = 11.88
  concurrent full-32K requests, so capacity is not the constraint; shared
  throughput (~70-80 tok/s across JARVIS + Claude UI + Spark) is. Extra agents
  queue rather than fail; reviewers route to the utility 30B when dual-model
  routing (Phase 4) is enabled. Auto-review
  after task completion is opt-in config (`review.auto: true`), on-demand
  `/review` always available.
- **True plan mode.** In plan mode the tool registry itself is filtered to
  read-only (enforced at execution, not by prompt). Model signals plan
  completion via an `exit_plan_mode` tool; the plan renders for approval;
  approve → mode drops to acceptEdits and execution starts; reject → stays in
  plan mode with feedback. Shift+Tab still cycles.
- **Verification habit.** System prompt + round-boundary nudges: before claiming
  done, run the project's test/build command (detected from project type or
  CLAUDE.md). A failed verification re-enters the loop instead of ending the turn.
- **Skills that fire on an 80B.** The Claude-skills bridge (built-ins +
  `~/.claude/skills` + plugin skills) already loads; make it *work* with the
  local model: user-invocable skills as slash commands (`/<skill-name>` like
  Claude Code), explicit trigger hints in the system prompt (an 80B won't
  infer skill relevance from descriptions the way Claude does), and
  budget-aware injection (a skill's full text only enters context when
  invoked, never preloaded — 32K is too small to carry idle skills).
  Verified live as part of the smoke suite.

### Phase 3 — Feel & polish + IDE interoperability

Deliverable: the remaining texture of Claude Code, usable alongside Xcode,
Cursor, and VS Code the way Claude Code is.

- Streaming markdown rendering quality (code blocks, tables) during generation.
- `patch_stdout` for background printers (worker status, notifications) so the
  input line never mangles.
- Argument-aware autocomplete (`/model <tab>` lists providers, `/resume <tab>`
  lists sessions).
- Session picker UX for `--resume`/`/resume` (arrow-key list with age/label,
  like `claude -r`).
- Banner/toolbar refinements: context %, model real-name unmasking (verify the
  existing uncommitted-then-fixed resolve path live), git branch, mode indicator.

**IDE interoperability** (how Claude Code actually "works with" IDEs — a CLI
that behaves perfectly inside and alongside editors, not an IDE plugin):

- **Embedded-terminal correctness.** Spark runs clean inside Cursor/VS Code
  integrated terminals (prompt_toolkit rendering, keybindings, Shift+Tab, Esc)
  — verified explicitly, since that's where Claude Code lives for most users.
- **Editor handoff.** Detect the host editor (`$TERM_PROGRAM`, running apps):
  file references in output become openable — `cursor --goto file:line` /
  `code --goto` for Cursor/VS Code, `xed --line` for Xcode. Configurable
  `ui.editor` override.
- **Diff-in-editor option.** When inside Cursor/VS Code, offer `code --diff`
  view of proposed edits as an alternative to the inline terminal diff.
- **Disk-first edits (already true, keep it that way).** All edits are plain
  file writes, so Xcode/Cursor auto-reload changes instantly — same interop
  model as Claude Code. Nothing may buffer edits in memory.
- **Xcode build loop.** The Phase 2 verification habit uses `xcodebuild`
  build/test (and simulator via `simctl` where relevant) for Swift projects —
  Spark's existing project detection already identifies them. Working next to
  Xcode on GigLedger/Boonpoint-class projects is the acceptance scenario.

### Phase 4 — Beyond parity (local-stack superpowers)

Deliverable: capabilities Claude Code doesn't have, enabled by owning the stack.
All four approved 2026-07-06. Independent of each other; ship in this order
(cheapest-first, each immediately useful):

- **Headless mode.** `spark -p "<prompt>" [--output json|text] [--max-rounds N]`
  — non-interactive single run: executes the agent loop with the configured
  permission mode (default: auto, never prompts; refused actions are reported,
  not asked). JSON output contract: result text, files changed, commands run,
  token/time stats, exit code semantics (0 done / nonzero failed). This is the
  integration surface for JARVIS, cron, and revdash-style automation.
- **Dual-model routing.** A `utility_model` config (default: `qwen3-coder:30b`
  via Ollama :11434) receives janitorial work: compaction summaries, session
  labels, commit-message drafting, and read-only subagent dispatches. The 80B
  keeps the main loop. Falls back to the primary model silently if the utility
  endpoint is down. Mirrors Claude Code's Haiku offload pattern.
- **RAG codebase brain.** On startup (and on demand via `/index`), index the
  current repo into the existing RAG service (:8010) as a per-project
  collection; expose a `code_search` tool for semantic queries alongside grep.
  Reuse the `/projectplan` query plumbing. Apple HIG/SwiftUI doc queries stay
  available for Swift projects. Degrades gracefully (tool absent) when :8010
  is unreachable — Spark must work away from home.
- **iOS vision loop.** For Swift projects: a `simulator_screenshot` tool
  (`xcrun simctl io booted screenshot`) whose image output feeds the model's
  vision path (image support already exists for drag-and-drop). Combined with
  the Phase 2 verification habit, enables build → screenshot → critique →
  iterate on SwiftUI work. Requires the vision-capable served model; feature
  no-ops with a clear message when the active model lacks vision.

- **Browser automation via MCP (added 2026-07-06).** Claude-in-Chrome-style
  capability: wire an existing browser MCP server (Playwright MCP or Chrome
  DevTools MCP, stdio via npx) into Spark's MCP config so the agent can drive
  real Chrome — navigate, click, type, read pages, screenshot. Text-first:
  accessibility-tree snapshots work with the text-only 80B; tight per-tool
  truncation budgets are mandatory at 32K. Screenshots route to a
  vision-capable model (Ollama 122B or Gemini) via dual-model routing.
  Gate: Spark's MCP stdio transport verified live against a real server
  first (the July 2 MCP fixes have only ever run against unit-test fakes).

Also folded in (small, no approval ceremony needed):
- **Session → training corpus export.** Opt-in config: completed Spark sessions
  export to `~/training-corpus` in the same shape as the existing Claude
  UI/JARVIS corpus pipeline, feeding future LoRA training.
- **Prefix-cache-aware prompt layout** (Phase 2 engineering rule): stable
  system-prompt ordering (static text first, volatile context last) to maximize
  vLLM prefix-cache hits → faster time-to-first-token on every turn.

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
