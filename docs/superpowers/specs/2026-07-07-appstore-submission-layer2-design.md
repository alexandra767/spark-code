# Spark Code — iOS App Store submission (Layer 2, safe slice)

- **Date:** 2026-07-07
- **Status:** Design approved (verbal); spec awaiting user review before writing-plans
- **Builds on:** Layer 1 (Gemini browser web-driver, merged `55b9431`) — the web-driver is the future helper for web-only ASC screens, NOT used in this slice.

## Goal

Spark Code can, for an iOS app under Alexandra's team:
1. **(read-only)** inspect an app's App Store submission **state + readiness** ("is version X ready to submit? what's missing?");
2. **(write, hard-gated)** **submit an already-prepared version for review**.

This is the SAFE FIRST SLICE of "Spark Code submits my apps." **No build uploads, no metadata/screenshot writes** — those risky steps are deferred to later slices. Follows the ASC REST flow documented in the `appstore-connect-submission` skill.

## Runtime / auth (confirmed)

- Runs in the Spark Code CLI on the **Mac**. Auth reuses the existing `~/.appstoreconnect/config.json` (`key_id`, `issuer_id`, `key_file` → the `AuthKey_*.p8`) — an **ES256 JWT** (aud `appstoreconnect-v1`, ~20-min exp) for `https://api.appstoreconnect.apple.com/v1`. **No new credential setup.**

## Components

### 1. `spark_code/appstore.py` — ASC API client (no tool surface of its own)
- `_jwt()` — ES256-sign from the config (iss=issuer_id, kid=key_id, exp≤20min, aud `appstoreconnect-v1`). Cache ~15 min.
- `_get/_post/_patch(path, ...)` — bearer auth, JSON, surfaces ASC error bodies, honors 429.
- `resolve_app(identifier) -> app_id` — by `filter[bundleId]` or by name; error if 0 or >1 match.
- `get_submission_state(app_id) -> {version, state, build_number, build_processing_state}` — the editable `appStoreVersion` + its `appStoreState`, attached build + `processingState`.
- `readiness(app_id) -> list[{check, ok, hint}]` — the skill's checklist: build attached & `processingState=VALID` (`GET /appStoreVersions/{id}/build`); `usesNonExemptEncryption` set (`GET /builds/{id}`); required screenshots present (`GET /appStoreVersionLocalizations/{id}/appScreenshotSets`); review test-creds present IF the app needs login (`GET /appStoreVersions/{id}/appStoreReviewDetail`).

### 2. `appstore_status` tool (read-only, `is_read_only=True`, no gate)
- Input: `app` (bundle id or name). Output: current version + `appStoreState`, attached build number, and the readiness checklist rendered as ✓/✗ + hints. Safe to run anytime.

### 3. `appstore_submit` tool (write, `requires_permission=True`)
- Input: `app` + `version` (must match the editable version).
- Steps: (a) run `readiness`; **refuse with the blocker list if not ready**; (b) render the exact summary — **app name, version, build number, platform IOS**; (c) require **explicit typed confirmation** (`"submit"`); **in non-interactive/headless mode, refuse** unless an explicit `confirm="submit"` argument is passed; (d) only then run the submit sequence:
  `POST /v1/reviewSubmissions` `{type:reviewSubmissions, attributes:{platform:IOS}, relationships:{app}}` → `POST /v1/reviewSubmissionItems` linking the `appStoreVersion` → finalize (`PATCH /v1/reviewSubmissions/{id}` `{submitted:true}`). Return the submission id + state.
- **Never** create a `reviewSubmission` before confirmation.

## Safety properties

- Submit refuses on any readiness blocker (can't submit a not-ready version).
- Submit requires a typed human confirmation; headless refuses without an explicit confirm arg — no path auto-fires.
- Scoped to Alexandra's team; the app must resolve to exactly one app under the account.
- No build uploads / metadata writes in this slice.
- `reviewSubmissions` cannot be DELETEd — if a wrong one is created, recover via `PATCH {canceled:true}`. The tool's error/help text states this.

## Testing

- **`appstore.py`:** unit-test the JWT (decode header+payload; assert `alg=ES256`, `iss`, `aud=appstoreconnect-v1`, `exp` in range). Unit-test `readiness` parsing against mocked GETs (ready / missing-build / missing-encryption). Unit-test `resolve_app` (0/1/many matches).
- **`appstore_status`:** unit-test rendering from a mocked client; then a **LIVE read** against a real app (e.g. GigLedger) as acceptance — safe, read-only.
- **`appstore_submit`:** unit-test the gate — refuses without confirmation, refuses when not ready, and issues the POST sequence **in order only after** confirmation (mocked client asserting call order). **No live test-submit** — a real submit runs only when Alexandra has a genuinely-ready version and says go.

## Out of scope (future slices)

Build upload (`altool`), metadata/screenshot writes, age-rating/encryption writes, Android/Play submission, subscription price setup. Each is its own slice.

## Risks / notes

- ASC token expiry / rate limits → cache JWT ~15 min, honor 429.
- Model must pass the correct version → the tool validates the version exists and is editable before submitting.
- `reviewSubmissions` is non-deletable → the readiness pre-check + typed confirmation are the guardrails; cancel path documented.
- Changes land on branch `feat/appstore-submission` in the live `~/spark-code`.
