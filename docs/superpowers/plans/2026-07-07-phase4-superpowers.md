# Phase 4: Beyond-Parity Local-Stack Superpowers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Spark capabilities Claude Code doesn't have, enabled by owning the DGX stack: scriptable headless runs, a 30B utility model doing the grunt work, semantic code search over the RAG service, an iOS build→screenshot→critique loop, browser automation via MCP, and session export into the training corpus.

**Architecture:** Six mostly-independent, cheapest-first units. Headless (Task 1) hardens the existing `_one_shot` path (cli.py:3942) into a machine contract. Dual-model routing (Task 2) adds a `utility_model` client used by compaction/labels/reviews. Code search (Task 3) reuses projectplan.py's async httpx RAG client (projectplan.py:167+) as a per-project index + `code_search` tool. Vision (Task 4) adds a `simulator_screenshot` tool feeding the existing image path (context.py:207 `add_user_with_image`). Browser (Task 5) is config + a live MCP verification, not new transport code. Corpus export (Task 6) is an opt-in session hook.

**Tech Stack:** Python 3.11, click, httpx (async, already used), Rich, pytest, pexpect; `xcrun simctl` (Task 4, macOS); an npx-launched browser MCP server (Task 5).

## Global Constraints

- All edits in `~/spark-code`. vLLM alias `qwen3.5:122b` (:30000) hard contract; utility model default = Ollama `qwen3-coder:30b` (:11434). Never key model behavior on model names.
- Gates before every commit: `.venv/bin/python -m pytest -q` (1003 green at start) and `.venv/bin/ruff check .` (0 at start).
- Additive-only; every new capability degrades gracefully when its backend is absent (RAG :8010 down, no simulator, MCP server missing) — Spark must still work away from the DGX.
- No new hard dependencies; browser MCP is launched via npx at runtime (documented, not vendored).
- Branch: `phase4/superpowers` off main (1044a52). Phase-end: suite + ruff + smoke 7/7 both engines + focused live checks + merge + push.
- Sub-agent/utility model calls must respect the Phase-0 fail-safe permission semantics and the Phase-2 concurrency semaphore.

---

### Task 1: Headless mode

**Files:**
- Modify: `spark_code/cli.py` (add `--output`, `--max-rounds` options near :3880; rework `_one_shot` at :3942)
- Create: `spark_code/headless.py` (JSON result assembly), `tests/test_headless.py`

**Interfaces:**
- Produces: `spark -p "<prompt>" [--output text|json] [--max-rounds N]` — non-interactive. `--output json` (default text) prints ONE JSON object to stdout at completion: `{"result": str, "files_changed": [str], "commands_run": [str], "tool_calls": int, "rounds": int, "input_tokens": int, "output_tokens": int, "error": str|null}`. Exit code: 0 = completed, 1 = error/exception, 2 = hit max-rounds without final answer. Permission mode in headless defaults to `auto` and NEVER prompts (non-interactive fail-safe: denied writes are recorded, not asked); `--trust` widens. `HeadlessResult` dataclass + `build_headless_json(result) -> str` in headless.py.
- The agent already tracks written paths (Phase 2 `_verify_written_paths`) and usage — reuse those; add a lightweight command-run tracker (bash tool invocations) if not already exposed.

- [ ] **Step 1: Failing tests**

```python
# tests/test_headless.py
import json
from spark_code.headless import HeadlessResult, build_headless_json


def test_json_shape_complete():
    r = HeadlessResult(result="done", files_changed=["a.py"], commands_run=["pytest"],
                       tool_calls=3, rounds=2, input_tokens=100, output_tokens=50, error=None)
    obj = json.loads(build_headless_json(r))
    assert obj["result"] == "done"
    assert obj["files_changed"] == ["a.py"]
    assert obj["error"] is None
    assert obj["tool_calls"] == 3


def test_json_shape_error():
    r = HeadlessResult(result="", files_changed=[], commands_run=[], tool_calls=0,
                       rounds=0, input_tokens=0, output_tokens=0, error="boom")
    obj = json.loads(build_headless_json(r))
    assert obj["error"] == "boom"
    assert obj["result"] == ""


def test_json_is_single_line_parseable():
    r = HeadlessResult(result="multi\nline\noutput", files_changed=[], commands_run=[],
                       tool_calls=0, rounds=1, input_tokens=0, output_tokens=0, error=None)
    out = build_headless_json(r)
    assert "\n" not in out.rstrip("\n")   # single JSON line for easy piping
    assert json.loads(out)["result"] == "multi\nline\noutput"
```

- [ ] **Step 2:** Run, verify fail. **Step 3:** Implement headless.py (dataclass + `json.dumps(asdict(r))`, ensure the result string's newlines are JSON-escaped not literal). **Step 4:** Pass.
- [ ] **Step 5:** Rework `_one_shot`: thread an accumulator through the agent run (files/commands/tool count/rounds/tokens — read how Agent exposes these post-run), catch exceptions into `error`, map completion state to exit code, emit text (current behavior) or JSON per `--output`. `--max-rounds` overrides the agent's cap for this run. In JSON mode, suppress Rich's normal streaming render (quiet Console) so stdout is ONLY the JSON object.
- [ ] **Step 6:** Live check (real engine): `spark -p "create hello.txt with the text HI and report done" --output json --trust` in a temp dir → parse stdout as JSON, assert `files_changed` contains hello.txt and file exists; `echo $?` == 0. `spark -p "..." ` (text mode) unchanged. Paste in report.
- [ ] **Step 7:** Full gates. Commit `feat: headless mode with JSON output contract`.

---

### Task 2: Dual-model routing

**Files:**
- Create: `spark_code/routing.py`, `tests/test_routing.py`
- Modify: `spark_code/config.py` (utility_model config), `spark_code/context.py` (compaction summary call), `spark_code/review.py` (reviewer/skeptic dispatch), `spark_code/cli.py` (session-label + commit-message generation if model-driven — grep)

**Interfaces:**
- Produces: `get_utility_client(config) -> ModelClient | None` in routing.py — builds a ModelClient for `config["utility_model"]` (a provider-shaped dict: endpoint/model/etc., default `{endpoint: http://spark-4a54.local:11434, model: qwen3-coder:30b, provider: ollama}`); returns None if unconfigured or explicitly disabled (`utility_model: null`). `route_or_fallback(utility, primary)` returns the utility client when non-None else primary — the ONE place callers decide. Callers: compaction summary (Task 2 wires `generate_compaction_summary` to prefer utility), review lenses+skeptics (run_subagent gains an optional `model` param already? verify — if not, thread one), session labeling.
- Fallback: if a utility call raises/times out, silently fall back to the primary model for that call (never fail the operation). Utility client is built once per session (lazy) and reused.
- Config: `utility_model` block; `routing.use_for: [compaction, review, labels]` (list, default all three) so the user can scope it.

- [ ] **Step 1: Failing tests** — `get_utility_client` returns a client for a valid block; None when `utility_model: null`; `route_or_fallback(None, primary)` → primary; `route_or_fallback(utility, primary)` → utility; `routing.use_for` gating (compaction not in list → compaction uses primary). Use a fake ModelClient factory injected for testing (don't hit network).
- [ ] **Step 2-4:** Implement, pass. The compaction/review call sites choose their model via `route_or_fallback` gated by `use_for`. IMPORTANT: the utility client must NOT be closed when the primary `/model` switch closes clients (own lifecycle) — verify teardown.
- [ ] **Step 5:** Live check: enable utility_model, run a task that triggers compaction (or force `/compact`), confirm via logs/stats that the summary request went to :11434 not :30000 (temporary stderr note or check both servers' access; if not observable cleanly, assert the routing decision in an integration test with two fake clients and document). Full gates. Commit `feat: dual-model routing — 30B utility model for compaction/review/labels`.

---

### Task 3: RAG code search

**Files:**
- Create: `spark_code/codesearch.py`, `tests/test_codesearch.py`
- Modify: `build_tools()` registration; `spark_code/cli.py` (/index command + autocomplete)

**Interfaces:**
- Consumes: projectplan.py's async RAG client pattern (read `_run_query` at projectplan.py:167+ — reuse the httpx.AsyncClient + timeout + graceful-empty-on-error shape; DEFAULT_RAG_SERVICE_URL at :100, config-overridable via `rag.service_url`).
- Produces:
  - `async index_project(cwd, service_url, collection) -> dict` — walks the repo (respect .gitignore; cap file size ~100KB, text files only, skip binary/venv/node_modules/.git), posts chunks to the RAG service's index endpoint as a per-project collection named `spark_code_<sha1(realpath(cwd))[:12]>`. Returns `{indexed: int, skipped: int, collection: str}`. If the service is unreachable → `{error: ...}` (no exception).
  - `CodeSearchTool` named `code_search`, `is_read_only=True`, schema `{query: str, k?: int}` — queries the project collection, returns formatted `path:line — snippet` results (capped, truncated via the Task-2 per-tool budget for a new `code_search` entry ~6000). Empty/unreachable → a clear "code search unavailable (index this project with /index; RAG service at <url>)" string, never an error.
  - `/index` command: runs `index_project` for cwd, shows progress + result; autocomplete entry.
- Config: `rag.service_url` (default DEFAULT_RAG_SERVICE_URL), `rag.code_search_enabled` (default true — but the TOOL only appears/works when a collection exists or the service is up; when the service is unreachable at startup the tool still registers but returns the unavailable message, so away-from-home sessions don't error).
- STATIC system-prompt line (budget-checked, report delta): "For semantic 'where/how is X handled' questions, use code_search (falls back to grep if unavailable)."

- [ ] **Step 1: Investigate** the RAG service's actual index + query API (read projectplan.py fully for the query shape; the index endpoint may need discovery — if the service only supports query, document that indexing must reuse whatever ingestion the service exposes, and scope Task 3 to query-against-existing-collections + a documented index path; adjust deliverable to reality, report it). This task's index half depends on what :8010 accepts.
- [ ] **Step 2: Failing tests** (mock the httpx client via MockTransport like tests/test_preflight.py): code_search formats results; unreachable service → unavailable message (not exception); collection name is deterministic from cwd; index_project respects size cap + skip patterns (use a temp tree, mock the POST).
- [ ] **Step 3-4:** Implement, pass. **Step 5:** Live check IF the RAG service is up (`curl -s http://spark-4a54.local:8010/health` or equivalent): `/index` a temp project, then `code_search` for a known symbol → correct file. If service down, document SKIP + the graceful-unavailable path proven by unit test. Full gates. Commit `feat: RAG code search (/index + code_search tool)`.

---

### Task 4: iOS vision loop

**Files:**
- Create: `spark_code/tools/simulator.py`, `tests/test_simulator.py`
- Modify: `build_tools()` registration (conditional — macOS only)

**Interfaces:**
- Produces: `SimulatorScreenshotTool` named `simulator_screenshot`, `is_read_only=True`, schema `{}` (no args — captures the booted simulator). Behavior: runs `xcrun simctl io booted screenshot <tmpfile.png>` (subprocess, timeout), reads the PNG, returns it as the tool result IN THE IMAGE FORMAT the context expects — i.e. the tool result must route into `add_user_with_image`-style multimodal content (context.py:207-217 shows the data-URL shape). Since tool results are normally strings, this needs a small mechanism: the tool returns a sentinel + the agent's tool-result handler detects `simulator_screenshot` and injects an image content block instead of text (mirror how drag-and-drop images enter context — grep `add_user_with_image` callers). Requires the active model to be vision-capable: `model.supports_vision` (config flag `providers.<p>.vision: true` — qwen3.5:122b has vision per the user's setup) — if not vision-capable, the tool returns "current model has no vision; screenshot saved to <path>" and still saves the file.
- Guarded registration: only register on Darwin AND when `xcrun` exists (shutil.which); otherwise absent (no error on Linux/DGX).
- Config: `providers.<name>.vision: bool`.

- [ ] **Step 1: Failing tests** (patch subprocess + the booted-check): tool builds the right simctl argv; no booted simulator (`simctl` returns error) → clear message not exception; non-vision model → saves file + text message; PNG readback → image content block shape (assert the returned structure carries base64 + mime, matching context.py:207's expectations). Registration guard: absent on non-Darwin (monkeypatch platform).
- [ ] **Step 2-4:** Implement, pass. The image-injection mechanism is the subtle part — keep it minimal and mirror the existing drag-and-drop path exactly.
- [ ] **Step 5:** Live check IF a simulator is bootable (this is macOS; `xcrun simctl list devices booted` — if none booted, boot one or document SKIP). With a booted sim + vision model: ask Spark "screenshot the simulator and describe what's on screen" → it calls the tool and describes real UI. Full gates. Commit `feat: simulator_screenshot vision tool for iOS work`.

---

### Task 5: Browser automation via MCP

**Files:**
- Modify: `docs/` (a browser-mcp setup note), `~/.spark/config.yaml` (user config — document the mcp_servers block, don't hardcode secrets)
- Test: live MCP verification (the transport code exists; this task VERIFIES it against a real server for the first time)

**Interfaces:**
- Consumes: the existing MCP stdio transport (the July-2 audit rewrote it but it was only ever tested against `tests/_mcp_fake_server.py` — this task is its first real-server trial). Config shape `mcp_servers` already supported (config.py:53).
- Produces: a documented, working `mcp_servers` entry launching a browser MCP server via npx (Playwright MCP: `npx -y @playwright/mcp@latest` OR Chrome DevTools MCP — pick whichever the investigation finds most reliable stdio-wise), exposing navigate/click/read/screenshot tools into Spark. Text-first: prefer accessibility-tree snapshots (work with the text 80B); screenshots route through Task 4's vision path / the utility-or-vision model. Per-tool truncation budget entries for the browser tools (they return large DOM/a11y dumps — cap ~8000).
- Setup doc: `docs/browser-mcp.md` — the exact config block, prerequisites (Node/npx), and a smoke procedure.

- [ ] **Step 1: Investigate + choose** the MCP server (spike: can the existing transport spawn `npx -y @playwright/mcp@latest` and complete the initialize handshake? Watch for the July-2 audit's known-fixed issues: stderr drain, id→response matching, `notifications/initialized`. Report what actually happens — this is a real-transport shakeout, findings drive the rest).
- [ ] **Step 2:** If the handshake works: add per-tool budgets for the browser tools; write docs/browser-mcp.md with the working config; a live check driving one real flow (navigate to a page, read its a11y snapshot, assert content). If the handshake reveals transport bugs, FIX them (with a regression test using the fake server reproducing the bug class) — this is the task's real value.
- [ ] **Step 3:** Full gates (unit). Commit `feat: browser automation via MCP (verified transport + setup docs)` OR `fix: MCP transport bugs found in first real-server trial` if fixes dominate.

---

### Task 6: Session → training corpus export

**Files:**
- Create: `spark_code/corpus.py`, `tests/test_corpus.py`
- Modify: `spark_code/cli.py` (session-end hook)

**Interfaces:**
- Produces: `export_session(context, path_dir, meta) -> str | None` in corpus.py — writes the completed session as a JSONL record (one object: system prompt hash, message list, model, timestamp-from-meta, files-changed) into `<path_dir>` shaped like the existing Claude UI/JARVIS corpus pipeline (the user's `~/training-corpus` — investigate one existing corpus file's shape first if reachable, else use a clean documented schema). Returns the written path or None if disabled. Opt-in config `corpus.export_enabled` (default false), `corpus.dir` (default `~/training-corpus/spark-code`).
- Guardrails: never export sessions that errored out mid-way (only clean completions); scrub obvious secrets (reuse any existing scrub util, else skip API-key-shaped strings); no network — local file write only. Timestamp comes from `meta` (Date.now unavailable in some contexts — pass it in from the CLI where `time` is available).
- Hook: at interactive session teardown (the run_interactive finally) and after a headless run, if enabled and the session completed cleanly, call export_session.

- [ ] **Step 1: Failing tests** — export writes a parseable JSONL line with the expected keys; disabled → returns None writes nothing; errored session (error flag set) → skipped; secret-shaped string scrubbed; dir created if absent.
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Full gates. Commit `feat: opt-in session export to training corpus`.

---

### Task 7: Acceptance + merge + push

- [ ] **Step 1:** Full gates: pytest (expect ≥1040), ruff 0, tree clean.
- [ ] **Step 2:** `scripts/smoke_live.py --provider llm` 7/7; `--provider coder` 7/7 or documented SKIP.
- [ ] **Step 3: Focused live checks** (real engine): (a) headless JSON run round-trips (parse stdout, exit code, file created); (b) dual-model routing decision proven (integration or live); (c) code_search graceful-unavailable OR live hit if :8010 up; (d) simulator tool live if a sim is bootable, else the guarded-absent/non-vision path; (e) at least ONE of browser-MCP or its documented SKIP with the transport-shakeout findings recorded; (f) corpus export writes a clean record when enabled. Each PASS/FAIL/SKIP + evidence.
- [ ] **Step 4:** Whole-branch final review (controller dispatches), fix round if needed, merge to main, `git push origin main`. Update memory + the deferred doc; this completes the 5-phase parity+superpowers program's Phase 4 (Phase 5 remains as the next spec-approved round).
