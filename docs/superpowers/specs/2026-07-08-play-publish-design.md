# Spark Code — Android/Play publish (build → upload → testing track)

- **Date:** 2026-07-08
- **Status:** Design approved (verbal); spec awaiting user review before writing-plans
- **Builds on:** the `play_status` read-only tool + `spark_code/play.py` client (merged `b96b262`) whose SA has confirmed publishing access. Android parallel to the iOS `appstore_build_upload` slice.

## Goal

A `play_publish` tool that: builds a signed AAB (`gradle bundleRelease`), uploads it to a Google Play edit, assigns it to a **testing track** (internal/alpha/beta), validates, and — **only with `confirm="publish"`** — commits the edit (releasing to that testing track). **Production is refused.**

## Safety model (the crux)

- **`track` must be one of `internal` / `alpha` / `beta`.** `production` (and anything else) is refused up front, before any build or API call.
- Steps 3–6 (create edit → upload → assign track → validate) happen inside an **uncommitted edit** — nothing is live. Only **commit** (step 8) releases anything, and it is gated behind `confirm="publish"` + `requires_permission=True` + headless refusal.
- **Any failure before commit deletes the edit** (in a `finally`) — nothing is published, no partial state.

## Runtime / prerequisites

- Mac + Android SDK/JDK; the Gradle project at `~/CodingProjects/gigledger-android` (root has `gradlew`). Release signing reads `keystore.properties`; the release build also requires `GOOGLE_PLAY_BILLING_PUBLIC_KEY` in `local.properties` (the build throws without it). If either is missing, the gradle step fails — surfaced as an environment/setup issue, not a code defect.
- Play SA (already resolved by `PlayClient`), publishing access confirmed.
- No new deps (PyJWT + httpx). Reuses `spark_code.xcbuild.run_step` (a generic streamed subprocess runner) for the gradle invocation.

## Components

### 1. `spark_code/play.py` — add publish methods
- `TESTING_TRACKS = frozenset({"internal", "alpha", "beta"})`.
- `async create_edit(package) -> str` — `POST .../edits`, returns edit id.
- `async delete_edit(package, edit_id)` — best-effort `DELETE`.
- `async upload_bundle(package, edit_id, aab_path) -> int` — POST the AAB bytes to
  `https://androidpublisher.googleapis.com/upload/androidpublisher/v3/applications/{package}/edits/{edit_id}/bundles?uploadType=media`
  with `Content-Type: application/octet-stream` + bearer, long timeout (AABs are tens of MB); returns `versionCode` from the response. Raises `PlayError` on ≥400 (e.g. duplicate versionCode).
- `async assign_track(package, edit_id, track, version_code) -> None` — `PUT .../tracks/{track}` body `{"track": track, "releases": [{"versionCodes": [str(version_code)], "status": "completed"}]}`.
- `async validate_edit(package, edit_id) -> dict` — `POST .../edits/{edit_id}:validate`.
- `async commit_edit(package, edit_id) -> dict` — `POST .../edits/{edit_id}:commit`.
- (`track_status` from the previous slice is unchanged.)

### 2. `play_publish` tool (`spark_code/tools/play_publish.py`)
`name="play_publish"`, `is_read_only=False`, `requires_permission=True`, params `project` (gradle root), `package`, `track`, `confirm`, optional `module` (default `:androidApp`).

Pipeline in `execute`:
1. `track` not in `TESTING_TRACKS` → refuse ("production is not allowed by this tool; testing tracks only"). Missing `project`/`package`/`track` → error.
2. `run_step(["./gradlew", f"{module}:bundleRelease"], cwd=project)` → on non-zero rc, return the log tail. Find the `.aab` under `{project}/{module-path}/build/outputs/bundle/release/*.aab` (module `:androidApp` → `androidApp/...`); none found → error.
3. `edit_id = await create_edit(package)`, wrapped so every subsequent failure `delete_edit`s in a `finally` (nothing published on error).
4. `vc = await upload_bundle(package, edit_id, aab)`.
5. `await assign_track(package, edit_id, track, vc)`.
6. `await validate_edit(package, edit_id)`.
7. Gate: if `confirm != "publish"` → return summary (package, versionCode, track, "validated ✓") + "re-invoke with confirm=\"publish\"" (headless: refuse). **Delete the edit** (nothing committed) and return.
8. `await commit_edit(package, edit_id)` → success message ("released version {vc} to {track}").

## Testing

- **Unit (mock `run_step`, `PlayClient` methods):** production/invalid track refused (no build, no API call); missing params; full happy-path sequence order `create_edit → upload_bundle → assign_track → validate_edit → commit_edit` **only** when `confirm="publish"`; without confirm → stops after validate, `delete_edit` called, `commit_edit` NOT called; headless refuses without confirm; a mid-pipeline error (e.g. upload raises) → `delete_edit` called, `commit_edit` NOT called; AAB-path discovery + gradle argv.
- **Live acceptance (controller-run):** build GigLedger AAB → create edit → upload → validate → **stop at the gate (no confirm)** → delete edit. Proves the pipeline **without publishing**. Current versionCode 45 already exists on alpha, so upload may reject the duplicate — which itself proves upload works (as with iOS Wanderlink). **Never pass `confirm="publish"` during acceptance.**

## Out of scope (future)

Production releases + staged rollout (a separate, extra-gated slice), release-notes/"what's new" text, staged-rollout percentages, promoting an existing versionCode between tracks (no rebuild).

## Risks / notes

- **Signing/build fragility** (keystore + billing key + multi-minute build) — the headline risk, surfaced clearly.
- AAB upload is a large media POST → long timeout; duplicate versionCode → `PlayError` (caught, edit deleted).
- Edit is per-app transactional; a concurrent edit → 409 `PlayError` (caught, nothing published).
- Changes land on branch `feat/play-publish` in the live `~/spark-code`.
