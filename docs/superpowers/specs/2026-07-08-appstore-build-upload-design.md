# Spark Code — iOS build upload (full pipeline)

- **Date:** 2026-07-08
- **Status:** Design approved (verbal); spec awaiting user review before writing-plans
- **Builds on:** Layer 2 (`appstore_status` + `appstore_submit`, merged `8f19baa`). This adds getting a build *into* App Store Connect so there's something to submit.

## Goal

A `appstore_build_upload` tool that runs the full iOS pipeline **archive → export (signed .ipa) → validate → upload**, driving the exact `xcodebuild`/`altool` sequence the `appstore-connect-submission` skill proved on this Mac. The **upload** is the only outward, gated step; archive/export/validate are local/dry.

## Runtime / prerequisites (confirmed present)

- Mac with Xcode: `xcodebuild` + `xcrun altool` both on `PATH`.
- ASC key: `~/.appstoreconnect/config.json` (`key_id`, `issuer_id`) + `private_keys/AuthKey_<key_id>.p8` — the standard location `altool --apiKey <key_id>` reads.
- **Signing quirk (machine-specific, from the skill):** this Mac has only a Development cert; app-store signing resolves via the Xcode-signed-in account + `-allowProvisioningUpdates` and **no `-authentication*` flags** (adding them triggers a Distribution-cert lookup that fails). Archiving therefore depends on Xcode being signed in and provisioning resolving — fragile and environment-heavy.

## Components

### 1. `spark_code/xcbuild.py` — pipeline runner (no tool surface)
- `async run_step(argv, cwd, timeout) -> (rc, stdout, stderr)` — streamed subprocess with a generous timeout (archive can take minutes).
- `archive(project, scheme, workspace, archive_path)` — builds the `xcodebuild … archive` argv and runs it.
- `export_ipa(archive_path, export_path, export_options_path)` — `xcodebuild -exportArchive …`.
- `write_default_export_options(path, team_id=None)` — generate `ExportOptions.plist` (`method=app-store`, `destination=export`, `signingStyle=automatic`, `uploadSymbols=true`, `teamID` if known) when none supplied.
- `altool(action, ipa, key_id, issuer_id)` — `xcrun altool --{validate,upload}-app -f <ipa> -t ios --apiKey <key_id> --apiIssuer <issuer_id> --output-format json`; returns parsed JSON.
- `read_archive_props(archive_path) -> {bundle_id, marketing_version, build_number}` — from the `.xcarchive`'s `Info.plist` `ApplicationProperties`.

### 2. `appstore_build_upload` tool (write, hard-gated on the upload)
`name="appstore_build_upload"`, `is_read_only=False`, `requires_permission=True`, params `project`, `scheme`, `confirm`.

## Pipeline (tool `execute`)

1. **Resolve inputs:** `project`+`scheme` from args → else `.spark/appstore.json` (`{project, scheme, export_options?}`) → else auto-discover a single `.xcodeproj`/`.xcworkspace` + scheme; error if ambiguous/missing. ASC `key_id`/`issuer_id` from config.
2. **Archive** to a fresh temp dir: `xcodebuild [-project|-workspace] <p> -scheme <s> -configuration Release -destination 'generic/platform=iOS' -archivePath <tmp>/App.xcarchive -allowProvisioningUpdates archive`. Stream; on failure return the tail of the log. (local)
3. **Export:** generate `ExportOptions.plist` (unless one is configured) → `xcodebuild -exportArchive -archivePath <archive> -exportPath <tmp>/export -exportOptionsPlist <opts> -allowProvisioningUpdates` → `.ipa`. (local)
4. **Validate:** `altool --validate-app … --output-format json`. On validation errors, stop and report them (no build created). (dry — no build)
5. **Collision check:** read `{bundle_id, marketing_version, build_number}` from the archive; `AscClient.resolve_app(bundle_id)` → app_id; `GET /v1/builds?filter[app]={app_id}&filter[preReleaseVersion.version]={marketing}&filter[version]={build_number}`. If a build already exists → **refuse** with "build {n} for {marketing} already exists — bump CURRENT_PROJECT_VERSION." **Never edit project files.**
6. **Confirm gate:** if `confirm != "upload"` → return a summary (app, marketing version, build number, `.ipa` path, "validated OK") + "re-run with confirm=\"upload\" to upload." Non-interactive (`interactive=False`) without `confirm="upload"` → refuse. No upload happens.
7. **Upload:** `altool --upload-app … --output-format json`. Parse success/failure. On success report the build + "run `appstore_status` to watch it process to VALID, then `appstore_submit` when ready." (outward — the one step that creates a build)

## Safety

- Steps 2–5 create **no** App Store build (`--validate-app` checks without uploading). Only step 7 (`--upload-app`) does — and it's behind the `confirm="upload"` gate + `requires_permission=True` + headless refusal.
- Build-number collision → refuse, no surprise `pbxproj` edits.
- Archive/export/validate write only into a **temp dir**, never the project or repo tree.
- Generous subprocess timeouts (archive/export minutes); temp dirs cleaned up.

## Testing

- **Unit (mock subprocess + ASC):** argv construction for archive/export/validate/upload; `ExportOptions.plist` contents (`method=app-store`, no auth flags); `altool` JSON parsing (success + validation-error); confirm-gate (refuse without `confirm="upload"`; headless refuse — no upload argv issued); collision-refuse (existing build → no upload). `read_archive_props` parsing from a fixture Info.plist.
- **Live acceptance (SAFE — no build created):** run **archive → export → validate** against one of Alexandra's real app projects (e.g. `~/CodingProjects/GigLedger` or `~/WANDERLINK`). Proves the pipeline up to Apple's validation. **Do NOT run `--upload-app` in testing.** If archive fails on code-signing, that's an environment/setup surface (Xcode account, provisioning) to report — not necessarily a code defect.

## Out of scope (future)

Metadata/screenshot writes, Android/Play build upload, TestFlight group assignment, automatic build-number bump, xcodegen/Capacitor project generation.

## Risks / notes

- **Signing fragility** is the headline risk: archiving is machine-specific and may fail non-interactively; the live acceptance may need the Mac's Xcode signing to be ready.
- `xcodebuild` output is verbose → rely on exit code + `altool --output-format json` for machine-readable success/failure.
- `generic/platform=iOS` requires a device-buildable scheme (not simulator-only).
- Changes land on branch `feat/appstore-build-upload` in the live `~/spark-code`.
