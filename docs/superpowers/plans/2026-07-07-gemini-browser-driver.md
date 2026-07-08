# Gemini Browser-Driver Implementation Plan (Layer 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Spark Code sub-agent run on Gemini 2.5 Pro so it can see and drive a visible browser (navigate + fill forms) while the local 80B stays the primary that narrates progress.

**Architecture:** Extend agent definitions to name any configured provider; `dispatch.py` builds that sub-agent's `ModelClient` from the named provider block (reusing cli.py's `_model_factory` pattern) and closes it after the run. A `.spark/agents/web-driver.md` def pins `gemini-pro`, carries the `browser_*` tools, and drives Playwright MCP against visible Mac Chrome.

**Tech Stack:** Python 3.12, pytest, httpx (existing `ModelClient`), Playwright MCP (`@playwright/mcp` via npx), Gemini 2.5 Pro (OpenAI-compatible endpoint, already configured as provider `gemini-pro`).

## Global Constraints

- Changes land in the **live** repo `~/spark-code` (editable install), NOT `~/CodingProjects/spark-code`.
- Config file: `~/.spark/config.yaml` (Mac).
- Provider `gemini-pro` already exists in config (`model: gemini-2.5-pro`). Add `vision: true` to it.
- A per-provider sub-agent `ModelClient` is built fresh for the dispatch and MUST be closed after — the lead's `model` and `utility_model` are caller-managed and must never be closed here.
- Unknown/misconfigured provider name → fall back to the lead's primary `model` + a logged warning; NEVER crash dispatch.
- Follow the existing `Phase N Task M` comment style already in these files.

---

### Task 1: Add `provider` field to `AgentDef`

**Files:**
- Modify: `spark_code/agents_registry.py` (dataclass `AgentDef` ~66-84; parser `_parse_agent_def` ~97-160)
- Test: `tests/test_agents_registry.py`

**Interfaces:**
- Produces: `AgentDef.provider: str | None` — a configured provider name (e.g. `"gemini-pro"`) or `None`. Consumed by Task 2's `_resolve_subagent_model`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agents_registry.py
from pathlib import Path
from spark_code.agents_registry import _parse_agent_def

def test_provider_field_parsed(tmp_path):
    p = tmp_path / "web-driver.md"
    p.write_text(
        "---\nname: web-driver\ndescription: drives the browser\n"
        "base_type: implementer\nprovider: gemini-pro\n---\nBody.\n")
    d = _parse_agent_def(p, source="project")
    assert d is not None
    assert d.provider == "gemini-pro"

def test_provider_defaults_none(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("---\nname: plain\ndescription: x\n---\nBody.\n")
    d = _parse_agent_def(p, source="project")
    assert d is not None and d.provider is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_agents_registry.py -k provider -v`
Expected: FAIL — `AgentDef.__init__() got an unexpected keyword argument 'provider'` (or `AttributeError: provider`).

- [ ] **Step 3: Add the field + parse it**

In the `AgentDef` dataclass, add after `source`:
```python
    source: str = "project"
    # Optional configured provider name to run this sub-agent on (e.g.
    # "gemini-pro"). None => use the lead's model (or utility per model_hint).
    # Validated at dispatch time against config["providers"], not here.
    provider: str | None = None
```
In `_parse_agent_def`, where the `AgentDef(...)` is constructed, add:
```python
    raw_provider = meta.get("provider")
    provider = raw_provider.strip() if isinstance(raw_provider, str) and raw_provider.strip() else None
```
and pass `provider=provider` into the `AgentDef(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_agents_registry.py -k provider -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/agents_registry.py tests/test_agents_registry.py
git commit -m "feat: AgentDef.provider — name a configured provider for a sub-agent"
```

---

### Task 2: Resolve a sub-agent's model from its `provider` in dispatch

**Files:**
- Modify: `spark_code/dispatch.py` (imports; `run_subagent` model-choice at ~439-440; close in `finally`)
- Test: `tests/test_dispatch_provider.py`

**Interfaces:**
- Consumes: `AgentDef.provider` (Task 1).
- Produces: `_resolve_subagent_model(custom, model, utility_model, config) -> tuple[object, bool]` — returns `(chosen_model, owns_client)`. `owns_client=True` means the caller must `await chosen_model.close()` after the run. Used inside `run_subagent`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_provider.py
from spark_code.agents_registry import AgentDef
from spark_code.dispatch import _resolve_subagent_model

CFG = {"providers": {"gemini-pro": {
    "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
    "model": "gemini-2.5-pro", "vision": True, "api_key": "k"}}}

def test_named_provider_builds_owned_client():
    d = AgentDef(name="web-driver", description="x", system_prompt="",
                 base_type="implementer", provider="gemini-pro")
    chosen, owns = _resolve_subagent_model(d, model="LEAD", utility_model="UTIL", config=CFG)
    assert owns is True
    assert chosen.provider == "gemini-pro"
    assert chosen.supports_vision is True

def test_unknown_provider_falls_back_to_primary():
    d = AgentDef(name="x", description="x", system_prompt="", provider="nope")
    chosen, owns = _resolve_subagent_model(d, model="LEAD", utility_model="UTIL", config=CFG)
    assert chosen == "LEAD" and owns is False

def test_no_provider_preserves_utility_hint():
    d = AgentDef(name="x", description="x", system_prompt="", model_hint="utility")
    chosen, owns = _resolve_subagent_model(d, model="LEAD", utility_model="UTIL", config=CFG)
    assert chosen == "UTIL" and owns is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_dispatch_provider.py -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_subagent_model'`.

- [ ] **Step 3: Implement the resolver**

Add imports near the top of `dispatch.py`:
```python
from .config import load_keys, resolve_provider_key
from .model import ModelClient
```
Add the helper (mirrors cli.py `_model_factory`, lines ~2789-2798):
```python
def _resolve_subagent_model(custom, model, utility_model, config):
    """Pick the ModelClient a custom sub-agent runs on.

    Returns (chosen_model, owns_client). owns_client=True means the CALLER
    must close chosen_model after the run — it was built here, unlike the
    caller-managed lead model / utility_model.
    """
    if custom is not None and getattr(custom, "provider", None):
        name = custom.provider
        pconf = (config.get("providers", {}) or {}).get(name)
        if not pconf:
            logger.warning("agent '%s': unknown provider '%s' — using primary model",
                           custom.name, name)
            return model, False
        client = ModelClient(
            endpoint=pconf.get("endpoint", "http://localhost:11434"),
            model=pconf.get("model", "unknown"),
            temperature=pconf.get("temperature", 0.7),
            max_tokens=pconf.get("max_tokens", 8192),
            api_key=resolve_provider_key(name, pconf, load_keys()),
            provider=name,
            timeout=float(pconf.get("timeout", 300)),
            supports_vision=bool(pconf.get("vision", False)),
        )
        return client, True
    if custom is not None and custom.model_hint == "utility" and utility_model is not None:
        return utility_model, False
    return model, False
```
Replace the `chosen_model = (...)` block at ~439-440 with:
```python
            chosen_model, _owns_subagent_model = _resolve_subagent_model(
                custom, model, utility_model, config)
```
For the non-custom branch (~444) leave `chosen_model = model` but also set `_owns_subagent_model = False` so the variable always exists.
In the `finally`/cleanup that already force-removes the worktree, add (after the run completes, guarded so a fallback client is never closed):
```python
            if _owns_subagent_model:
                try:
                    await chosen_model.close()
                except Exception:
                    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_dispatch_provider.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the dispatch regression tests**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/ -k "dispatch or subagent" -v`
Expected: PASS — existing dispatch/subagent behavior unchanged (primary/utility paths untouched).

- [ ] **Step 6: Commit**

```bash
cd ~/spark-code && git add spark_code/dispatch.py tests/test_dispatch_provider.py
git commit -m "feat: run a custom sub-agent on a named provider (owned client, safe fallback)"
```

---

### Task 3: Wire the browser + `web-driver` agent, and prove it end-to-end

**Files:**
- Modify: `~/.spark/config.yaml` (add `mcp_servers.playwright`; add `vision: true` to `gemini-pro`)
- Create: `~/spark-code/.spark/agents/web-driver.md`

**Interfaces:**
- Consumes: Task 1 (`provider` field) + Task 2 (provider-based model resolution) + Playwright MCP tools (`playwright__browser_*`).

- [ ] **Step 1: Pre-warm Playwright MCP (one-time)**

Run: `npx -y @playwright/mcp@latest --help | head -5`
Expected: help text prints (package downloaded/cached).

- [ ] **Step 2: Enable visible browser + Gemini vision in config**

Add to `~/.spark/config.yaml` under `mcp_servers:` (alongside `nano_banana`):
```yaml
  playwright:
    command: npx
    args: ["-y", "@playwright/mcp@latest", "--browser", "chrome",
           "--isolated", "--output-dir", "/tmp/spark-playwright-mcp"]
```
(Note: NO `--headless` — the window must be visible.) And add `vision: true` to the existing `providers.gemini-pro` block.

- [ ] **Step 3: Create the web-driver agent definition**

```markdown
---
name: web-driver
description: Drives a real web browser to read pages and fill out forms. Use for any task that needs navigating, clicking, typing, or submitting web forms.
base_type: implementer
provider: gemini-pro
tools: playwright__browser_navigate, playwright__browser_snapshot, playwright__browser_click, playwright__browser_type, playwright__browser_fill_form, playwright__browser_take_screenshot, playwright__browser_press_key, playwright__browser_wait_for, playwright__browser_navigate_back
---
You drive a real web browser to accomplish the task you are given. You can SEE the page via screenshots — take one whenever you are unsure what is on screen. Work in small, verified steps: snapshot or screenshot, decide the single next action, do it, then confirm the result before the next action. After each action, state in one short line what you just did and what you see now, so the person watching understands your progress. Never submit, purchase, or send anything irreversible unless the task explicitly says to. When the task is done (or blocked), stop and report exactly what you did, step by step, and the final state of the page.
```

- [ ] **Step 4: Smoke test — observe Gemini drive a visible browser**

Run (interactive): `cd ~/spark-code && .venv/bin/spark`
Then prompt: `Use the web-driver agent to open https://www.selenium.dev/selenium/web/web-form.html and fill the text field with "hello" and select an option in the dropdown, then tell me what you did.`
Expected, all observed:
1. The lead (local 80B) dispatches `web-driver`.
2. A **visible Chrome window** opens and the field is filled / dropdown changed.
3. The lead narrates the web-driver's step summaries back in the session.
Confirm the driver ran on Gemini: `grep -c gemini ~/.spark/logs/*.log 2>/dev/null` or watch the token/provider indicator. If the browser never opens, check `Warning: MCP server 'playwright' failed` on startup and re-run Step 1.

- [ ] **Step 5: Commit the agent definition**

```bash
cd ~/spark-code && git add .spark/agents/web-driver.md
git commit -m "feat: web-driver agent — Gemini-pro drives a visible browser to fill forms"
```
(The user's `~/.spark/config.yaml` is outside the repo; note its edits in the commit body but don't add it.)

---

## Layer 2 (separate plan)

App submission via ASC/Play APIs with a confirm-before-submit gate is **not** in this plan — it gets its own spec + plan once Layer 1 is proven. The `web-driver` from this plan becomes its helper for web-only console steps.

## Self-Review

- **Spec coverage:** browser-on-visible (Task 3), per-sub-agent provider (Tasks 1–2), web-driver def + narration (Task 3 def prompt), smoke proof (Task 3 Step 4), Gemini vision flag (Task 3 Step 2). Layer 2 explicitly deferred. ✓
- **Placeholders:** none — every step has real code/commands. ✓
- **Type consistency:** `_resolve_subagent_model` signature + `(chosen_model, owns_client)` tuple used identically in Task 2 test and dispatch edit; `AgentDef.provider` defined in Task 1, consumed in Task 2. ✓
