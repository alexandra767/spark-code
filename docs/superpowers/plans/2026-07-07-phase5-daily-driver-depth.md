# Phase 5: Daily-Driver Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the real daily-driver friction points and make Spark usable off the DGX: cross-repo writes, cloud-model keys, a health check, custom agents, MCP images, formalized hooks, conversation rewind, and isolated implementer subagents.

**Architecture:** Mostly small, high-leverage units on existing seams. Write-roots widens the existing `_validate_path` allowlist (tools/base.py:11). Cloud keys layer a guided `/setkey` + preset table + gitignored keys store over the existing `${ENV}` interpolation (config.py:141) and FallbackChain (fallback.py). `/doctor` is a read-only diagnostic. Custom agents extend dispatch.py's agent_type registry. MCP images reuse Task-4's `add_user_with_image` injection. Hooks formalize the existing HookManager (hooks.py). Rewind adds a per-turn context snapshot ring. Worktree dispatch wraps `run_subagent` in a git worktree.

**Tech Stack:** Python 3.11, click, httpx, Rich, pytest, pexpect; git worktree (Task 8).

## Global Constraints

- All edits in `~/spark-code`. vLLM alias `qwen3.5:122b` (:30000) hard contract; utility model = Ollama `qwen3-coder:30b` (:11434). Never key model behavior on model names.
- Gates before every commit: `.venv/bin/python -m pytest -q` (1121 green at start) and `.venv/bin/ruff check .` (0 at start).
- Additive-only; every new capability degrades gracefully / is off by default where it widens trust or reaches the network.
- No new hard dependencies.
- **SECURITY (cloud keys):** a real API key must NEVER be committed, printed in full, or written to a repo-tracked file. Masked entry, gitignored keys file (chmod 600) or env only. The Phase-4 corpus scrub must catch cloud-key shapes.
- Branch: `phase5/daily-driver-depth` off main (ea18db3). Phase-end: suite + ruff + smoke 7/7 both engines + focused live checks + merge + push.

---

### Task 1: Path-scoped write permissions

**Files:**
- Modify: `spark_code/tools/base.py:11-58` (`_validate_path`), `spark_code/config.py` (default), `spark_code/tools/write_file.py` + `edit_file.py` (pass configured roots — check how they call `_validate_path` first)
- Test: `tests/test_tools/` (write-path tests)

**Interfaces:**
- Produces: `_validate_path(file_path, cwd=None, for_write=False, extra_roots=None)` — `extra_roots: list[str] | None` are additional absolute realpath'd roots allowed for writes (in addition to cwd + temp). write_file/edit_file read `permissions.write_roots` from config and pass expanded/realpath'd roots as `extra_roots`. Config: `permissions.write_roots: []` (default empty = current behavior exactly). Paths in write_roots are `~`-expanded and realpath'd; a non-existent root is skipped (not an error).

- [ ] **Step 1: Failing tests**

```python
# append to tests/test_tools/test_write_file.py (match its existing style/imports)
import os
from spark_code.tools.base import _validate_path


def test_write_root_allows_sibling_dir(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    sibling = tmp_path / "other"; sibling.mkdir()
    target = str(sibling / "f.txt")
    # without extra_roots → refused
    try:
        _validate_path(target, cwd=str(proj), for_write=True)
        assert False, "should have refused"
    except ValueError:
        pass
    # with the sibling as a write_root → allowed
    resolved = _validate_path(target, cwd=str(proj), for_write=True,
                              extra_roots=[str(sibling)])
    assert resolved.endswith("f.txt")


def test_write_root_still_refuses_outside_all_roots(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    allowed = tmp_path / "allowed"; allowed.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    try:
        _validate_path(str(outside / "x"), cwd=str(proj), for_write=True,
                       extra_roots=[str(allowed)])
        assert False
    except ValueError:
        pass


def test_write_root_nonexistent_skipped(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    # a non-existent root must not crash; write to cwd still works
    resolved = _validate_path(str(proj / "a.txt"), cwd=str(proj), for_write=True,
                              extra_roots=["/does/not/exist/anywhere"])
    assert resolved.endswith("a.txt")
```

- [ ] **Step 2:** Run, verify fail. **Step 3:** Add `extra_roots` param; realpath each existing root into `allowed_roots`. **Step 4:** Pass. **Step 5:** Wire write_file/edit_file to read `get(config, "permissions", "write_roots", default=[])`, expand/realpath, pass as extra_roots (check how they access config — they may need it threaded through the tool; if the tool doesn't currently see config, add a class attr set at build_tools time, mirroring how `editor`/`diff_in_editor` were threaded in Phase 3). **Step 6:** Full gates + commit `feat: permissions.write_roots for cross-repo writes`.

---

### Task 2: Cloud API keys — guided setup + presets + keys store

**Files:**
- Create: `spark_code/keys.py`, `tests/test_keys.py`
- Modify: `spark_code/config.py` (keys-file load + merge into provider api_keys), `spark_code/cli.py` (`/setkey` command + autocomplete), `spark_code/corpus.py` (verify scrub covers cloud keys)

**Interfaces:**
- Produces (spark_code/keys.py):
  - `PROVIDER_PRESETS: dict[str, dict]` — `anthropic` (endpoint `https://api.anthropic.com/v1/`, model `claude-sonnet-4-5`, key env `ANTHROPIC_API_KEY`), `openai` (`https://api.openai.com/v1`, `gpt-4o`, `OPENAI_API_KEY`), `gemini` (existing `generativelanguage.googleapis.com/v1beta/openai`, `gemini-2.0-flash`, `GEMINI_API_KEY`), `openrouter` (`https://openrouter.ai/api/v1`, `openrouter/auto`, `OPENROUTER_API_KEY`). Each: `{endpoint, model, key_env, doc_url}`.
  - `keys_path() -> str` = `~/.spark/keys` (expanduser).
  - `load_keys() -> dict[str, str]` — reads the gitignored keys file (simple `PROVIDER=value` lines or JSON — pick JSON for robustness), returns `{}` if absent/unreadable. Never raises.
  - `save_key(provider: str, value: str) -> str` — writes/updates the key in the keys file, `os.chmod(path, 0o600)`, returns the path. Creates `~/.spark/` if absent.
  - `mask(value: str) -> str` — `"sk-…" ` first 3 + last 4 with `…` middle for display; never full value.
  - `resolve_provider_key(provider_name: str, provider_conf: dict, keys: dict) -> str` — precedence: explicit `${ENV}`-resolved api_key in provider_conf (existing behavior) > keys-file entry > "".
- Config integration: `resolve_provider` (config.py) also consults `load_keys()` so a provider whose config has no api_key but whose key is in the keys file gets it. `/setkey <provider>` (no args → list presets): prompts (masked, EscWatcher-paused Prompt.ask like Phase 2/5 interactive prompts), saves via save_key, and if the provider isn't in config yet, adds a provider block from the preset. Prints masked confirmation + "use it: `/model <provider>`".
- SECURITY: `~/.spark/keys` added to a `.gitignore` if one exists in `~/.spark` (or documented as never-in-repo since ~/.spark isn't a repo); corpus scrub must catch `sk-ant-...`, `sk-...`, OpenRouter `sk-or-...` (extend Phase-4 `_SECRET_PATTERNS`, add a test).

- [ ] **Step 1: Failing tests** — PROVIDER_PRESETS has the 4 providers with required keys; save_key writes + chmod 600 + load_keys round-trips; mask never returns the full value and shows first3/last4; resolve_provider_key precedence (env-in-conf wins over keys-file wins over ""); corpus scrub catches `sk-ant-api03-...`. Use tmp keys_path via monkeypatch (never touch the real ~/.spark/keys).
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Wire `/setkey` + resolve_provider consulting keys + scrub patterns. Live check: `/setkey openrouter` with a DUMMY value (not a real key) → keys file written, chmod 600, masked confirmation, `/model openrouter` selects it (will fail to actually generate with a dummy key — that's expected; assert the provider is selectable + the request is ATTEMPTED against the right endpoint, don't need a successful completion). Document. **Step 6:** Full gates + commit `feat: cloud API key setup (/setkey) with provider presets + gitignored keys store`.

---

### Task 3: `/doctor` health command

**Files:**
- Create: `spark_code/doctor.py`, `tests/test_doctor.py`
- Modify: `spark_code/cli.py` (`/doctor` command + autocomplete)

**Interfaces:**
- Produces: `async run_doctor(config) -> list[DoctorCheck]` where `DoctorCheck` = dataclass `{name: str, status: "ok"|"warn"|"fail", detail: str}`. Checks (each isolated, never raises — a check that errors becomes a `fail` row): (1) primary engine reachable (`fetch_server_models` from preflight) + context-window drift; (2) served-model alias present in /v1/models; (3) utility model reachable (if configured); (4) RAG service :8010 reachable (rag.service_url; brief GET, short timeout); (5) MCP servers configured → count + whether each launches (best-effort, skip if none); (6) editor detection (detect_editor from Phase 3); (7) install version + git branch/dirty (subprocess `git`); (8) cloud keys present (names only, masked — never values). `render_doctor(console, checks)` → Rich table green/yellow/red.
- `/doctor` command runs it, renders the table. Read-only; safe to run anytime.

- [ ] **Step 1: Failing tests** (mock the network probes via injected async callables / MockTransport): run_doctor returns a check per subsystem; a probe raising → that row is `fail` not an exception; render_doctor produces a table containing each check name; masked-keys row shows names not values.
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Live: `/doctor` against the real stack → table with engine ok, alias ok, RAG status, editor detected, version/branch. Paste. **Step 6:** Full gates + commit `feat: /doctor health check command`.

---

### Task 4: Custom subagent definitions

**Files:**
- Modify: `spark_code/dispatch.py` (agent_type resolution), `spark_code/cli.py` (load definitions at startup), maybe `spark_code/agents_registry.py` (new, if cleaner)
- Create: `tests/test_custom_agents.py`

**Interfaces:**
- Produces: `load_agent_defs(cwd) -> dict[str, AgentDef]` — reads `.spark/agents/*.md` (project) and `~/.spark/agents/*.md` (global; project wins on name conflict). `AgentDef` = `{name, description, tools: list[str]|None (allowlist; None = type default), model_hint: "primary"|"utility"|None, system_prompt: str (the md body)}`. YAML frontmatter parse (reuse the skills loader's frontmatter parser — grep how skills/base.py parses frontmatter). `dispatch_agent`'s schema `agent_type` enum GAINS the custom names; when a custom type is dispatched, run_subagent uses that def's system_prompt + tool allowlist (intersect with the type's read-only/write rules — a custom agent can't exceed what its base permission allows; default read-only unless the def opts into writes AND the lead grants). Malformed def → skipped with a logged warning, never crashes startup.
- STATIC/registration: custom agent names appear in a lean index line like skills; the plan-mode dispatch special-case (Phase 2) treats custom read-only agents like explore/reviewer.

- [ ] **Step 1: Failing tests** — load_agent_defs reads a temp `.spark/agents/reviewer.md` with frontmatter; project overrides global on name clash; malformed frontmatter → skipped not raised; a custom def's tools allowlist filters the sub-registry; dispatch of the custom type uses its system_prompt.
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Live: create `.spark/agents/finder.md` (a read-only "find where X is defined" agent), dispatch it → correct answer using only read tools. **Step 6:** Full gates + commit `feat: custom subagent definitions (.spark/agents/*.md)`.

---

### Task 5: MCP image-content routing (Phase 4 carry-forward)

**Files:**
- Modify: `spark_code/mcp/client.py:40-47` (MCPTool result assembly), `spark_code/agent.py` (the Task-4 image-injection path)
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: Task-4's `_pending_image_injections` buffer + `_flush_pending_images` (agent.py) and `add_user_with_image` (context.py).
- Produces: when an MCP tool result contains an `image` content item (base64 + mimeType), instead of dropping it to `[Image: <mime>]` text, route it through the SAME buffered image-injection mechanism simulator_screenshot uses (so browser_take_screenshot's image reaches a vision model). Gate on `model.supports_vision` exactly like Task 4 — non-vision model keeps the text placeholder (but note the image was available). The MCPTool must signal the pending image up to the agent's `_store_tool_result` (mirror the simulator sentinel pattern, or have MCPTool return a structured result the handler detects — pick the approach that matches how Task 4 wired it; read agent.py:920/937 region first).

- [ ] **Step 1: Failing tests** — an MCP result with an image item + vision model → image buffered + injected after the tool-result run (valid sequence, mirror Task-4's mixed-round assertion); non-vision model → text placeholder retained, no image block; multiple images in one MCP result all land. Use the fake MCP server / a fake result dict.
- [ ] **Step 2-4:** Implement, pass (reuse Task-4 buffering — do NOT reintroduce the mid-round corruption; the flush-after-round invariant must hold for MCP images too). **Step 5:** Full gates + commit `feat: route MCP image content through the vision path`.

---

### Task 6: Hooks formalization + recipes

**Files:**
- Modify: `spark_code/hooks.py` (harden), `spark_code/cli.py` (`/hooks` command listing configured hooks)
- Create: `docs/hooks.md`, `tests/test_hooks.py` (extend if exists)

**Interfaces:**
- Consumes: the existing HookManager (hooks.py — Hook, HookManager, run_hooks, has_hooks, get_events).
- Produces: (1) hardening — a hook that fails/times out NEVER blocks or crashes the agent loop (verify + test the failure path; the agent.py:1044 `before_<tool>` call site must swallow hook errors to a dim note); {path}/{command} substitution must be shell-safe (no injection via a crafted filename — verify how the hook command is built and quote/escape or use argv). (2) `/hooks` command — lists configured hooks (event, pattern, command) in a table; read-only. (3) docs/hooks.md with two working recipe configs: `after_write_file` → `ruff check --fix {path}` (Python files), `before_git_commit`/a commit hook → `pytest -q` (documented as a gate). (4) config `hooks:` shape documented.
- SECURITY: filename-based injection is the key risk (a file named `; rm -rf ~` used in {path}) — the substitution MUST NOT expand into a shell string unescaped. Test with a hostile filename.

- [ ] **Step 1: Failing tests** — a failing hook (nonzero exit) doesn't raise / doesn't stop the tool; a hook timeout is caught; a hostile `{path}` (`; touch /tmp/pwned`) does NOT execute the injected command (assert the sentinel file is not created); `/hooks` lists configured hooks.
- [ ] **Step 2-4:** Implement/harden, pass. **Step 5:** docs/hooks.md written; full gates + commit `feat: harden hooks + /hooks command + recipe docs`.

---

### Task 7: True conversation rewind

**Files:**
- Modify: `spark_code/context.py` (per-turn snapshot ring), `spark_code/cli.py` (`/rewind` conversation option — the Phase-1 `_rewind_files` menu at cli.py:881)
- Test: `tests/test_context.py`, `tests/test_cli_helpers.py`

**Interfaces:**
- Produces: `Context.snapshot() -> None` (pushes a deep copy of `messages` + `turn_count` + `last_prompt_tokens` onto a bounded ring, default maxlen 10) called at the START of each user turn (before add_user — find the turn boundary in the loop/cli); `Context.rewind_conversation(steps: int = 1) -> bool` (pops `steps` snapshots, restores the messages/turn_count/last_prompt_tokens, returns False if nothing to rewind). `/rewind` menu (Phase-1 honest labels) GAINS option `conversation` → `rewind_conversation(1)`; the existing `last file edits` / `checkpoint` / `both` options stay. `both` now also offers conversation. Snapshot ring is session-only (not persisted).
- Interaction: rewinding conversation must leave a VALID message sequence (snapshots are taken at turn boundaries where the sequence is already valid, so this is inherent — assert it). Todo state: out of scope unless trivial (note in report).

- [ ] **Step 1: Failing tests** — snapshot then add_user+add_assistant then rewind_conversation(1) restores the pre-turn messages; ring bounded at maxlen (11 snapshots → oldest dropped); rewind with empty ring returns False; restored sequence is valid (no orphaned tool_calls).
- [ ] **Step 2-4:** Implement, pass. Wire snapshot() at the turn boundary + the /rewind menu option. **Step 5:** Live: multi-turn session, `/rewind` → conversation → prior turn restored, next message continues from there. **Step 6:** Full gates + commit `feat: true conversation rewind via per-turn snapshots`.

---

### Task 8: Worktree-isolated implementer dispatches

**Files:**
- Modify: `spark_code/dispatch.py` (worktree option in run_subagent / DispatchAgentTool)
- Test: `tests/test_dispatch.py`

**Interfaces:**
- Consumes: run_subagent (Phase 2/4).
- Produces: `dispatch_agent` schema gains `isolated?: bool` (default false); when `agent_type=="implementer" and isolated`, run_subagent creates a git worktree under `.spark/worktrees/<short-id>/` (from the repo HEAD), runs the sub-agent with that worktree as cwd (write_roots/validation scoped there), and the returned summary includes the worktree path + `git diff --stat` of its changes for the lead to review/merge. Auto-remove the worktree if unchanged (git diff empty); keep it if changed (lead decides). Not a git repo → falls back to non-isolated with a note. Concurrency: worktree creation is the EXPENSIVE part — keep it opt-in, gate under the existing semaphore.
- Guardrails: worktree paths strictly under `.spark/worktrees/`; cleanup on unchanged; never touch the user's working tree.

- [ ] **Step 1: Failing tests** (mock git subprocess + run_subagent): isolated implementer dispatch calls `git worktree add` under `.spark/worktrees/`; summary includes the diffstat; unchanged worktree → `git worktree remove` called; non-git-repo → falls back non-isolated with a note; explore/reviewer types ignore `isolated` (read-only, no worktree).
- [ ] **Step 2-4:** Implement, pass. **Step 5:** Live IF a temp git repo: isolated implementer dispatch writes a file in the worktree, summary reports the diffstat, user's main tree untouched. **Step 6:** Full gates + commit `feat: worktree-isolated implementer dispatches`.

---

### Task 9: Acceptance + merge + push

- [ ] **Step 1:** Full gates: pytest (expect ≥1180), ruff 0, tree clean.
- [ ] **Step 2:** `scripts/smoke_live.py --provider llm` 7/7; `--provider coder` 7/7 or documented SKIP.
- [ ] **Step 3: Focused live checks** (real engine): (a) write_roots lets a write land in a configured sibling dir + still refuses outside all roots; (b) `/setkey` with a DUMMY key writes a chmod-600 keys file + masks + provider selectable (no real completion needed); (c) `/doctor` table against the real stack; (d) a custom `.spark/agents/*.md` agent dispatches correctly; (e) `/rewind` conversation restores a prior turn; (f) an isolated implementer dispatch in a temp git repo (or documented SKIP); (g) MCP image routing via the fixed unit test (browser screenshot needs a vision model — unit-proof acceptable). Each PASS/FAIL/SKIP + evidence.
- [ ] **Step 4:** Whole-branch final review (controller dispatches), fix round if needed, merge to main, `git push origin main`. Update memory + deferred doc. This completes the spec-approved Phase 5; note any remaining spec items as the next round.
