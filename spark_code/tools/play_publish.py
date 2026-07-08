"""Hard-gated Android gradle/build -> upload -> testing-track publish pipeline tool.

Orchestrates the full Google Play publish pipeline: build a signed .aab via
gradle (``bundleRelease``), upload it to Google Play, assign it to a TESTING
track, and (only with ``confirm="publish"``) commit the edit so it actually
releases. This is a real, outward action against Google's API (a successful
commit releases a build to real testers), so it is gated hard:

  1. ``track`` must be one of the TESTING tracks (``internal``/``alpha``/
     ``beta``) — production is out of scope for this tool and is refused
     BEFORE any build or API call happens.
  2. Validation (``edits.validate``) must pass.
  3. The caller passes ``confirm="publish"`` exactly.

Without ``confirm="publish"`` this builds/uploads/assigns/validates (no
commit), deletes the edit (nothing is ever left half-published), and returns
a summary — an interactive session can call it once to preview, then again
with ``confirm="publish"`` to proceed. A headless (``interactive=False``) run
additionally refuses to commit if ``confirm`` is omitted, exactly like
``AppStoreBuildUploadTool``/``AppStoreSubmitTool``.

Any failure once the edit is created (upload, assign, validate) deletes the
edit — nothing is ever left published or committed on an error path.
"""
from __future__ import annotations

from pathlib import Path

from ..play import TESTING_TRACKS, PlayClient, PlayError
from ..xcbuild import XcBuildError, run_step
from .base import Tool


class PlayPublishTool(Tool):
    def __init__(self, interactive: bool = True):
        self._interactive = interactive

    @property
    def name(self) -> str:
        return "play_publish"

    @property
    def description(self) -> str:
        return ("Build a signed Android AAB (gradle bundleRelease), upload it to Google Play, "
                "and (only with confirm='publish') release it to a TESTING track "
                "(internal/alpha/beta — production is not allowed). Input: project (gradle root), "
                "package, track, confirm, optional module (default :androidApp).")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "project": {"type": "string", "description": "Gradle project root (has ./gradlew)"},
            "package": {"type": "string", "description": "Android package, e.g. com.gigledger.android"},
            "track": {"type": "string", "description": "Testing track: internal | alpha | beta"},
            "module": {"type": "string", "description": "Gradle module (default :androidApp)"},
            "confirm": {"type": "string", "description": "Pass 'publish' to release; omit to preview"}},
            "required": ["project", "package", "track"]}

    @property
    def is_read_only(self) -> bool:
        return False

    @property
    def requires_permission(self) -> bool:
        return True

    async def execute(self, project: str = "", package: str = "", track: str = "",
                      module: str = ":androidApp", confirm: str = "", **kwargs) -> str:
        if not project or not package or not track:
            return "Error: 'project', 'package', and 'track' are required."
        if track not in TESTING_TRACKS:
            return (f"Refusing: track '{track}' is not allowed. This tool only releases to "
                    f"testing tracks ({', '.join(sorted(TESTING_TRACKS))}) — production is out of scope.")
        # 1. build the signed AAB
        try:
            rc, out, err = await run_step(["./gradlew", f"{module}:bundleRelease"], cwd=project)
        except XcBuildError as e:
            return f"Gradle build error: {e}"
        if rc != 0:
            return f"Gradle bundleRelease failed (rc {rc}). Last output:\n{(out + err)[-1500:]}"
        mod_dir = module.lstrip(":").replace(":", "/")
        aabs = list((Path(project) / mod_dir / "build" / "outputs" / "bundle" / "release").glob("*.aab"))
        if not aabs:
            return "Build succeeded but no .aab was found under build/outputs/bundle/release/."
        aab = str(aabs[0])
        # 2. drive the edit; any pre-commit failure deletes it (nothing published)
        c = PlayClient()
        try:
            edit_id = await c.create_edit(package)
        except PlayError as e:
            return f"Google Play error: {e}"
        try:
            vc = await c.upload_bundle(package, edit_id, aab)
            await c.assign_track(package, edit_id, track, vc)
            await c.validate_edit(package, edit_id)
            summary = (f"Built + validated ✓ — ready to release:\n"
                       f"  {package}  version code {vc}  →  track {track}")
            if confirm != "publish":
                await c.delete_edit(package, edit_id)  # nothing committed
                tail = ("\n\nHeadless: refusing. Re-invoke with confirm=\"publish\" to release."
                        if not self._interactive
                        else f"\n\nThis RELEASES to your {track} testers. Re-invoke with confirm=\"publish\".")
                return summary + tail
            await c.commit_edit(package, edit_id)
            return f"✅ Released version code {vc} to the {track} track for {package}."
        except Exception as e:
            # ANY post-create failure (PlayError, a transport error, a bad
            # response, a file-read error) must discard the edit so no orphaned
            # uncommitted edit is left behind. commit is never reached here.
            try:
                await c.delete_edit(package, edit_id)
            except Exception:
                pass
            return f"Google Play error (nothing published, edit discarded): {e}"
