# Spark Code — Local-model vision via Gemini, browser automation, and app-submission

- **Date:** 2026-07-07
- **Status:** Design approved (verbal); spec awaiting user review before writing-plans

## Problem / Goal

Spark Code's daily-driver model is **text-only**. The active `llm` provider points at
the DGX Spark endpoint `:30000`, which currently serves
`Intel/Qwen3-Coder-Next-int4-AutoRound` under the `qwen3.5:122b` label. A direct
image probe returns `400 … is not a multimodal model` — it literally rejects images.

We want Spark Code to:

- **Layer 1 (this spec):** *see* web pages / screenshots so it can drive a browser the
  way Claude-in-Chrome does — navigate, read, fill forms — using **Gemini** as its eyes
  while the local model stays the primary driver.
- **Layer 2 (future, separate spec):** fill in and **submit Alexandra's apps** to App
  Store Connect / Google Play, the way Claude Code does for her.

### Runtime topology (confirmed)

- The `spark` CLI runs on the **Mac** — editable install loading `spark_code` from
  `~/spark-code` (HEAD `47dd1d5`). **Code + config changes land here.**
- Config: `~/.spark/config.yaml` (Mac).
- Models run on the **DGX Spark** (`:30000` SGLang/vLLM, `:11434` Ollama).
- Browser automation (Playwright MCP) therefore runs **on the Mac against Mac Chrome** —
  ideal, since the Mac is where Alexandra is logged into things.
- A second, non-live clone exists at `~/CodingProjects/spark-code` (same commit). Not
  used by the CLI; flagged for later cleanup.

## Layer 1 — Vision routing + browser (implement now)

### Approach: automatic per-image vision routing

Keep the local model as the primary driver. When an image (browser/simulator screenshot
or pasted image) needs to be *seen* and the primary model lacks vision, Spark makes a
**side-call to a configured vision provider (Gemini)** to describe/answer, then injects
that text back into the primary model's context. The local model stays primary and cheap;
Gemini is a momentary side-trip; control returns automatically ("switch over, switch
back").

Rejected alternatives:
- *Manual `/model` switch* — works today with config only, but the whole session runs on
  Gemini and the user must flip it by hand. Doesn't match the "automatic" intent.
- *Gemini-backed browser subagent* — cleaner separation but heavier; needs per-subagent
  provider support built first. Deferred.

### Components / changes

1. **Turn the browser on (config only).** Add an `mcp_servers.playwright` block to
   `~/.spark/config.yaml` per `docs/browser-mcp.md` (system Chrome, `--isolated`,
   `--output-dir /tmp/spark-playwright-mcp`). Requires Node/npx on the Mac PATH; first run
   downloads the MCP server via npx (pre-warm once). For most page work the model reads the
   accessibility-tree **text** via `browser_snapshot` — no vision needed there.

2. **Declare Gemini as multimodal + name it as the eyes (config).** Add `vision: true` to
   the `gemini` / `gemini-pro` provider blocks (both genuinely multimodal). Add a key that
   tells the primary provider which provider to use for images, e.g.
   `providers.llm.vision_provider: gemini-pro` (falling back to a global
   `model.vision_provider`).

3. **Vision-routing code** (`spark_code/agent.py`, in `_resolve_mcp_images` and the shared
   image path that `simulator_screenshot` / pasted images use). When
   `self.model.supports_vision` is False **and** a `vision_provider` is configured:
   - resolve a `ModelClient` for the vision provider (reuse existing provider resolution);
   - call it with the image + a fixed "you are the eyes for another agent — describe all
     visible text, form fields with labels, buttons, links, and their approximate
     positions" prompt (plus current task context when available);
   - replace the `[Image: <mime>]` placeholder with an injected
     `[Vision (gemini): <description>]` text block the primary model reads directly.
   - **On any failure, degrade to the existing `[Image: <mime>]` placeholder** — no
     regression versus today.

4. **Cost / caps.** Vision side-calls fire only on actual images (occasional), billed to
   the paid Gemini key. Reuse existing browser result budgets; no new caps needed.

### Testing / proof

- **Unit:** feed a known PNG through the image path with a stub vision client → assert the
  description text is injected; assert a vision-capable primary still embeds the image
  directly (no double-routing); assert failure degrades to the `[Image:]` placeholder.
- **Smoke (must observe before "done"):** real run — navigate to a login/signup form,
  `browser_take_screenshot`, confirm Gemini reports the actual fields, and the local model
  can then target them (e.g. "type into the email field"). Not complete until watched
  working end-to-end.

### Out of scope (Layer 1)

No app submission, no credential handling, no auto-submit. Purely: see pages + drive a
browser.

## Layer 2 — App submission capability (FUTURE — its own spec/plan)

Sketch only; gets a dedicated spec before any build.

- **Core = the API path** (what Claude Code actually uses — *not* browser-driving the
  console): JWT + App Store Connect REST API for iOS, Play Developer API for Android —
  create version, attach build, set metadata / age rating / encryption flag, submit.
  Reuses the existing ASC API key + Play service account.
- **Browser + Gemini (Layer 1) = helper** for the few web-only ASC screens the API can't
  reach (e.g. IAP-attach-to-version).
- **Hard guardrail:** assemble the full submission, show the exact payload, and require
  **explicit human confirmation before the final submit call**. Never auto-fires. Using
  signed keys sidesteps the Apple-2FA-in-browser problem entirely.

## Risks / notes

- `:30000` serves a text-only coder model under the `qwen3.5:122b` name (the silent
  model-swap pattern). Vision routing sidesteps this by never relying on local vision.
- Playwright MCP first run downloads via npx — pre-warm on the Mac to avoid a slow first
  handshake.
- Changes land in the **live** `~/spark-code`, not the `~/CodingProjects` duplicate.
