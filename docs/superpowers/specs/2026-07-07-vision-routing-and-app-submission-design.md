# Spark Code — Gemini-driven browser automation, and app-submission

- **Date:** 2026-07-07
- **Status:** Design approved (verbal); spec awaiting user review before writing-plans
- **Supersedes:** the earlier "automatic per-image vision routing (local model drives, Gemini
  only sees)" approach in this file's first revision — rejected because a quantized local
  coder model driving long, high-stakes web flows off a *text description* isn't reliable
  enough. Chosen instead: **Gemini drives; the local model orchestrates and narrates.**

## Problem / Goal

Spark Code's daily-driver model is **text-only** — the active `llm` provider points at DGX
Spark `:30000`, which serves `Intel/Qwen3-Coder-Next-int4-AutoRound` (an ~80B int4 coder
model at 32K ctx) under the `qwen3.5:122b` label. A direct image probe returns
`400 … is not a multimodal model`.

We want Spark Code to:

- **Layer 1 (this spec):** drive a web browser accurately — navigate, read, **fill out
  forms** — using **Gemini 2.5 Pro** as the driver (it sees the pixels), while the local
  80B stays the primary the user talks to and *narrates what's happening* so she can watch.
- **Layer 2 (future, its own spec):** **submit Alexandra's apps** to App Store Connect /
  Google Play via the platform **APIs**, using the Layer 1 browser only for web-only steps.

### Runtime topology (confirmed)

- `spark` CLI runs on the **Mac** (editable install from `~/spark-code`, HEAD `47dd1d5`).
  Code + config land here.
- Config: `~/.spark/config.yaml` (Mac).
- Models run on the **DGX Spark** (`:30000` SGLang, `:11434` Ollama). Gemini is a cloud
  provider already in config (`gemini-pro` = `gemini-2.5-pro`).
- Playwright MCP browser runs **on the Mac against visible Mac Chrome** — the user watches
  it work.
- Non-live duplicate clone at `~/CodingProjects/spark-code` (flagged for later cleanup).

## Layer 1 — Gemini-driven browser sub-agent (implement now)

### Approach

The local 80B is the **primary orchestrator** the user interacts with. When a task needs
the web, it **dispatches a sub-agent pinned to `gemini-pro`** that:

- is multimodal — it **sees screenshots natively**, no vision-routing needed;
- owns the `browser_*` (Playwright MCP) tools;
- drives the browser: navigate, read, click, type, **fill forms**;
- returns a concise text summary of what it did.

The primary **narrates the sub-agent's progress** back to the user, and the browser runs
**visible (non-headless)** so she can literally watch Chrome.

**Honesty note baked into the UX:** the local 80B is text-only, so "seeing what's going on"
means (a) the visible Chrome window and (b) the web-driver's **step-by-step text summaries**
surfaced in her session — *not* the 80B itself viewing pixels. Only Gemini sees the images.

Rejected alternatives: local-model-drives + auto-vision-routing (unreliable on long/complex
flows — the whole reason we pivoted); manual `/model` switch (whole session flips to Gemini,
user-driven).

### Components / changes

1. **Turn the browser on, visible (config).** Add `mcp_servers.playwright` to
   `~/.spark/config.yaml` per `docs/browser-mcp.md`, but **drop `--headless`** so the window
   shows; keep `--isolated` + `--output-dir /tmp/spark-playwright-mcp`. Needs Node/npx on
   Mac PATH (pre-warm the MCP package once).

2. **Per-sub-agent provider support (core new capability).** Today
   `AgentDef.model_hint` accepts only `primary`/`utility` (`agents_registry.py:63`), both
   local. Extend agent definitions to name an **arbitrary configured provider** (e.g.
   `provider: gemini-pro`), and have `dispatch.py` build that sub-agent's `ModelClient` from
   the named provider block — reusing cli.py's `_model_factory` pattern
   (`resolve_provider_key(name, pconf, load_keys())` for the key, `pconf` for the rest,
   `supports_vision=pconf.get("vision", False)`). Set `vision: true` on the `gemini-pro`
   provider block.

3. **A `web-driver` agent definition** (`.spark/agents/web-driver.md`): `provider: gemini-pro`,
   a `base_type` that permits browser *actions* (not read-only), `tools:` = the `browser_*`
   set, and a system prompt instructing it to accomplish the given web task and report each
   step it takes.

4. **Orchestration + narration.** The primary dispatches `web-driver` for web tasks and
   surfaces its returned step summaries to the user. No auto-submit anything.

### Testing / proof

- **Unit:** an agent def with `provider: gemini-pro` parses and resolves to a `ModelClient`
  on the gemini provider with `supports_vision=True`; an unknown provider name degrades
  safely (falls back to primary, logs a warning — never crashes dispatch).
- **Smoke (must observe before "done"):** from a normal Spark Code session on the local
  80B, ask it to fill a public demo form (e.g. a login/contact form). Confirm: it dispatches
  the `web-driver`, **Gemini** (not the local model) drives a **visible** Chrome that fills
  the fields, and the primary narrates the steps back. Verify via the provider actually used.

### Out of scope (Layer 1)

No app submission, no credentials, no auto-submit. Purely: Gemini-driven browsing +
form-filling with local-model narration.

## Layer 2 — App submission capability (FUTURE — its own spec/plan)

- **Core = the API path** (what Claude Code uses; *not* browser-driving the console): JWT +
  App Store Connect REST API (iOS) / Play Developer API (Android) — create version, attach
  build, set metadata / age rating / encryption, submit. Reuses the existing ASC key + Play
  service account. The **local 80B assembles the submission data** (coding-shaped, its
  strength); deterministic code makes the calls.
- **Layer 1 web-driver = helper** for web-only ASC screens the API can't reach.
- **Hard guardrail:** assemble the submission, show the exact payload, **require explicit
  human confirmation before the final submit call.** Never auto-fires. Signed keys sidestep
  the Apple-2FA-in-browser problem — which is *why* submission stays on the API even though
  Gemini can fill forms.

## Risks / notes

- `:30000` serves a text-only coder model under the `qwen3.5:122b` name (silent model-swap
  pattern). Irrelevant to seeing now — Gemini does all the seeing.
- **Browser agents need iteration.** The first `web-driver` version will fumble on some
  pages; expect a couple of watch-and-tighten rounds. Not a failure — normal.
- Gemini *driving* (not just seeing) spends more Gemini tokens per web task than seeing
  alone. Billed to the paid Gemini key; web tasks are occasional.
- Everything is additive and opt-in: browser runs in an isolated throwaway profile, image
  failures degrade gracefully, and nothing can submit until Layer 2's confirm-gate exists.
- Changes land in the **live** `~/spark-code`, not the `~/CodingProjects` duplicate.
