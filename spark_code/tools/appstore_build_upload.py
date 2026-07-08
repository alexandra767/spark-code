"""Hard-gated iOS archive/export/validate/upload pipeline tool.

Orchestrates the full App Store Connect build-upload pipeline: archive the
project, export a signed .ipa, validate it against Apple, check for a build
collision, and (only with ``confirm="upload"``) upload it. This is a real,
outward action against Apple's API (a successful upload creates a new build
in App Store Connect that cannot be un-created), so it is gated hard:

  1. Validation (``altool --validate-app``) must pass.
  2. No build already exists in App Store Connect for this app's
     marketing-version + build-number pair (a collision means the project's
     ``CURRENT_PROJECT_VERSION`` needs bumping — this tool never edits
     project files to fix that for you).
  3. The caller passes ``confirm="upload"`` exactly.

Without ``confirm="upload"`` this runs archive/export/validate (no upload)
and returns a summary — an interactive session can call it once to preview,
then again with ``confirm="upload"`` to proceed. A headless
(``interactive=False``) run additionally refuses to upload if ``confirm`` is
omitted, exactly like ``AppStoreSubmitTool``.

All archive/export/validate work happens inside a ``tempfile.
TemporaryDirectory`` that is removed when the tool returns (success or
failure) — nothing is ever written into the project/repo tree.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ..appstore import AscClient, AscError
from ..xcbuild import (
    XcBuildError,
    altool_argv,
    archive_argv,
    export_argv,
    parse_altool_json,
    read_archive_props,
    run_step,
    write_default_export_options,
)
from .base import Tool


class AppStoreBuildUploadTool(Tool):
    def __init__(self, interactive: bool = True):
        self._interactive = interactive

    @property
    def name(self) -> str:
        return "appstore_build_upload"

    @property
    def description(self) -> str:
        return ("Archive, export a signed .ipa, validate it against Apple, and (only with "
                "confirm='upload') upload the build to App Store Connect. Input: project "
                "(.xcodeproj/.xcworkspace path), scheme, confirm. Archive/export/validate create "
                "no build; only upload does.")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "project": {"type": "string", "description": ".xcodeproj or .xcworkspace path"},
            "scheme": {"type": "string", "description": "Xcode scheme to archive"},
            "confirm": {"type": "string", "description": "Pass exactly 'upload' to upload; omit to preview"}},
            "required": ["project", "scheme"]}

    @property
    def is_read_only(self) -> bool:
        return False

    @property
    def requires_permission(self) -> bool:
        return True

    async def execute(self, project: str = "", scheme: str = "", confirm: str = "", **kwargs) -> str:
        if not project or not scheme:
            return "Error: 'project' and 'scheme' are required."
        is_ws = str(project).endswith(".xcworkspace")
        try:
            with tempfile.TemporaryDirectory(prefix="spark-xcbuild-") as td:
                arch = str(Path(td) / "App.xcarchive")
                exp = str(Path(td) / "export")
                opts = str(Path(td) / "ExportOptions.plist")
                # 1. archive
                rc, out = await run_step(archive_argv(project, scheme, arch, is_ws))
                if rc != 0:
                    return f"Archive failed (rc {rc}). Last output:\n{out[-1500:]}"
                # 2. export
                write_default_export_options(opts)
                rc, out = await run_step(export_argv(arch, exp, opts))
                if rc != 0:
                    return f"Export failed (rc {rc}). Last output:\n{out[-1500:]}"
                # The real xcodebuild -exportArchive writes the .ipa into `exp`
                # as a side effect of the (real) subprocess; find it if present.
                # Fall back to the conventional <scheme>.ipa path otherwise —
                # this keeps the pipeline moving under a monkeypatched run_step
                # (unit tests never touch the filesystem) while still picking
                # up the actual exported file name in real runs.
                ipas = list(Path(exp).glob("*.ipa"))
                ipa = str(ipas[0]) if ipas else str(Path(exp) / f"{scheme}.ipa")
                props = read_archive_props(arch)
                # 3. validate (creates no build)
                c = AscClient()
                key_id = c._config().get("key_id")
                issuer_id = c._config().get("issuer_id")
                rc, out = await run_step(altool_argv("validate", ipa, key_id, issuer_id))
                ok, errs = parse_altool_json(out)
                if rc != 0 or not ok:
                    return "Validation failed:\n" + "\n".join(f"  - {e}" for e in (errs or [f"rc {rc}"]))
                # 4. collision check
                app_id = await c.resolve_app(props["bundle_id"])
                q = (f"/v1/builds?filter[app]={app_id}"
                     f"&filter[preReleaseVersion.version]={props['marketing_version']}"
                     f"&filter[version]={props['build_number']}")
                existing = (await c.get(q)).get("data", [])
                if existing:
                    return (f"Build {props['build_number']} for {props['marketing_version']} already "
                            f"exists in App Store Connect — bump CURRENT_PROJECT_VERSION and re-run. "
                            f"(Not uploading; your project files were not modified.)")
                summary = (f"Validated ✓ — ready to upload:\n"
                           f"  App: {props['bundle_id']}\n"
                           f"  Version {props['marketing_version']} (build {props['build_number']})\n"
                           f"  IPA: {ipa}")
                # 5. gate
                if confirm != "upload":
                    if not self._interactive:
                        return summary + "\n\nHeadless: refusing. Re-invoke with confirm=\"upload\" to upload."
                    return summary + "\n\nThis uploads a real build to Apple. Re-invoke with confirm=\"upload\"."
                # 6. upload (the one outward step)
                rc, out = await run_step(altool_argv("upload", ipa, key_id, issuer_id))
                ok, errs = parse_altool_json(out)
                if rc != 0 or not ok:
                    return "Upload failed:\n" + "\n".join(f"  - {e}" for e in (errs or [f"rc {rc}"]))
                return (f"✅ Uploaded {props['bundle_id']} {props['marketing_version']} "
                        f"(build {props['build_number']}). It will take a few minutes to process — "
                        f"run appstore_status to watch it reach VALID, then appstore_submit when ready.")
        except (AscError, XcBuildError) as e:
            return f"Build upload error: {e}"
