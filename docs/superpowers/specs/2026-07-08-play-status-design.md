# Spark Code — Android/Play status (read-only)

- **Date:** 2026-07-08
- **Status:** Design approved (verbal); spec awaiting user review before writing-plans
- **First Android slice.** Sibling to the iOS `appstore_status` tool. Foundation + diagnostic for a future gated Play-upload slice.

## Goal

A read-only `play_status` tool showing an Android app's Google Play tracks (production / beta / alpha / internal, plus `wear:*`), their current versionCodes, release names, and status — via the Play Developer API. **The service account's publishing access was already confirmed by a live probe** (it can create edits), so this both delivers value and confirms the foundation for a later upload slice.

## Runtime / auth (confirmed by probe)

- Runs on the Mac. Service-account JSON → **RS256 jwt-bearer OAuth** (scope `https://www.googleapis.com/auth/androidpublisher`), token cached ~55 min.
- **No new dependencies** — reuses `PyJWT` (RS256) + `httpx`.
- **SA resolution order:** env `PLAY_SA_JSON` → `~/.playconsole/service-account.json` → `~/CodingProjects/revdash/secrets/play-rtdn-sa.json` (where it currently lives). Note: Alexandra may want to copy the SA to a dedicated path so Spark Code doesn't reach into revdash's secrets.

## Components

### 1. `spark_code/play.py` — Play Developer API client (no tool surface)
- `PlayError(Exception)`.
- `PlayClient(sa_path=None)`:
  - resolves the SA JSON via the order above (error if none found);
  - `_token()` — RS256 jwt-bearer assertion `{iss: client_email, scope: androidpublisher, aud: token_uri, iat, exp ≤ iat+3600}` → POST `token_uri` (`grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer`) → access token; cached with a skew margin;
  - `get/post/delete(path)` — httpx against `https://androidpublisher.googleapis.com/androidpublisher/v3`, bearer auth, raises `PlayError` on ≥400 (body truncated, token never echoed);
  - `track_status(package) -> list[dict]` — `POST .../applications/{pkg}/edits` (create) → `GET .../edits/{id}/tracks` → **`DELETE .../edits/{id}` in a `finally`** → returns `[{track, status, version_name, version_codes}]` per track/latest-release. The edit is **ephemeral, never committed** → read-only in effect.

### 2. `play_status` tool (`spark_code/tools/play_status.py`)
`name="play_status"`, `is_read_only=True`, `requires_permission=False`, param `package`. Resolves via `PlayClient().track_status(package)` and renders each track → version name + versionCode(s) + status (e.g. `production: 1.0.34 (vc 44) — completed`). Registered in `build_tools`.

## Safety

- **Read-only in effect:** the only write is creating an edit, which is deleted in a `finally` and never committed — no track/release/rollout change is possible from this tool.
- SA publishing access already confirmed; no new deps; token never logged.

## Testing

- **Unit (mock httpx/SA):** `_token()` assertion shape (RS256, `iss=client_email`, `scope=androidpublisher`, `aud=token_uri`, `exp` in range); SA resolution order (env > `~/.playconsole` > revdash path; error when none); `track_status` parsing from mocked responses (tracks with releases → `version_codes`/`status`/`name`; empty tracks handled); **the ephemeral edit is DELETEd even when the tracks GET raises** (assert delete called in the error path).
- **Live acceptance (controller-run):** `play_status` against real `com.gigledger.android` → shows production `1.0.34` (vc 44), alpha `1.0.35 (45)` (vc 45), internal 44, plus wear tracks — already demonstrated by the design probe.

## Out of scope (future slices)

Play upload pipeline (`gradle bundleRelease` → `edits.bundles.upload` → `edits.tracks.update` → validate → commit), staged rollout / promotion, and any Play write/submission — each its own gated slice, now unblocked by the confirmed SA access.

## Risks / notes

- Ephemeral-edit read: if `DELETE` fails, an uncommitted edit lingers harmlessly (Play auto-expires it); `track_status` deletes in a `finally`.
- Play edits are per-app transactions; a concurrent edit elsewhere could 409, surfaced as a `PlayError` (read fails cleanly, no writes).
- Changes land on branch `feat/play-status` in the live `~/spark-code`.
