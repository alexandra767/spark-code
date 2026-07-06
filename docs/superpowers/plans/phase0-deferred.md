# Phase 0 — Deferred findings (recorded 2026-07-06)

Verified findings from the Phase 0 audit (3 lenses + skeptic verification +
task/final reviews) that were consciously deferred, with the reason and the
phase that owns each. Everything here was judged non-blocking for the
Phase 0 merge by the final whole-branch review.

## Deferred to Phase 1 (hygiene / consolidation)

- **Startup fetches `/v1/models` three times** (resolve_real_model_name, ping,
  preflight) — consolidate into one fetch shared by all three consumers.
- **Permission policy duplicated in two places** — `agent.py:356-361`'s inline
  parallel-path `authorized` predicate is a second copy of
  `PermissionManager.check()`'s no-prompt policy (currently coherent; verified
  2026-07-06). Extract `PermissionManager.allows_without_prompt()` and use it
  at both sites before the next policy change silently diverges them.
- **`Context.get_messages()` is a mutating getter** (idempotent orphan repair,
  tested) — rename or move the repair to the request-build site.
- **Stale docstring**: `model.py:253-254` says `real_model_name` is "used for
  reasoning-template detection"; it is display-only since the think-filter fix.
- **Untested at their level** (covered one level away): preflight
  api_key→Authorization header; pinned-files cli wiring (class-tested);
  agent-level denial-reason propagation (manager/worker-tested);
  estimate_tokens 3.5 chars/token ratio (reserve behavior tested).
- **Smoke suite polish**: temp-dir cleanup; c5 model-claim regex is advisory
  (authoritative pytest-subprocess gate is the real check).
- **`preflight.py` url f-string outside try** — unreachable via the guarded
  call site; tighten when preflight grows more callers.
- **Schema reserve uses //4 while estimate uses 3.5 chars/token** — reserve
  undercounts ~12%; immaterial (estimate inflation dominates), align when
  server-reported usage lands (Phase 2).

## Deferred to Phase 2 (design work, already in the parity spec)

- **Write sandbox refuses $HOME-sibling writes** (e.g. editing
  `~/CodingProjects/x` while cwd is elsewhere). Intentional tightening; the
  Phase 2 path-scoped permission design (`write_file(./**)`-style rules) owns
  the policy.
- **Workers under an `ask`-mode lead are denied nearly everything**
  (fail-safe non-interactive policy, accepted 2026-07-06). Phase 2's
  dispatch-agent design (read-only agent types, lead-granted write capability)
  replaces this wholesale.
- **`~/.spark/tasks.json` cross-session staleness** — needs owner/heartbeat
  design; cosmetic today (atomic writes already prevent corruption).
- **`stream_options.include_usage` sent unconditionally** — vLLM supports it;
  Ollama behavior version-dependent (no 400 observed live on either engine
  2026-07-06). Phase 2's context engine consumes usage deliberately and adds
  the capability gate.
- **Latent-only streaming hardening** (current engines never emit these):
  non-string tool_call `arguments`; parallel tool_call assembly when the
  server omits `index`.
- **Compaction session label derived from summary text** — cosmetic.

## Explicitly rejected (do not re-raise without new evidence)

- **NVFP4 / server-side changes** — out of scope for this repo entirely.
- **Weakening the `qwen3.5:122b` alias contract** — the masquerade is
  load-bearing for JARVIS + Claude UI; Spark must never depend on unmasking.
