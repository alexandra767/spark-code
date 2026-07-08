"""Hard-gated App Store Connect submit-for-review tool.

Submits an ALREADY-PREPARED iOS App Store version for review. This is a
real, one-way action against Apple's API, so it is gated on two independent
conditions and refuses unless BOTH hold:

  1. ``readiness()`` (spark_code.appstore) reports no blockers for the
     app/version pair.
  2. The caller passes ``confirm="submit"`` exactly.

Without ``confirm="submit"`` this returns a summary only (no API calls) —
an interactive session can call it once to preview, then again with
``confirm="submit"`` to proceed. A headless (``interactive=False``) run
additionally refuses to submit if ``confirm`` is omitted (rather than
silently previewing), since there's no human present to read the preview
and decide; a headless caller that explicitly passes ``confirm="submit"``
(e.g. a scripted release step that already made the decision) is still
honored, exactly like an interactive one, provided readiness passes.
"""
from __future__ import annotations

from ..appstore import AscClient, AscError, blockers, readiness
from .base import Tool


class AppStoreSubmitTool(Tool):
    def __init__(self, interactive: bool = True):
        self._interactive = interactive

    @property
    def name(self) -> str:
        return "appstore_submit"

    @property
    def description(self) -> str:
        return ("Submit an ALREADY-PREPARED iOS App Store version for review. Refuses if the "
                "version isn't ready. Requires confirm='submit' to actually submit; without it, "
                "returns a summary only. Input: app (bundle id/name), version, confirm.")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "app": {"type": "string", "description": "App bundle id (preferred) or exact name."},
            "version": {"type": "string", "description": "Version string to submit, e.g. 1.3.22."},
            "confirm": {"type": "string",
                "description": "Pass exactly 'submit' to actually submit for review. Omit to preview."}},
            "required": ["app", "version"]}

    @property
    def is_read_only(self) -> bool:
        return False

    @property
    def requires_permission(self) -> bool:
        return True

    async def execute(self, app: str = "", version: str = "", confirm: str = "", **kwargs) -> str:
        if not app or not version:
            return "Error: 'app' and 'version' are required."
        try:
            c = AscClient()
            app_id = await c.resolve_app(app)
            st = await c.get_submission_state(app_id)
            if st["version"] != version:
                return (f"Refusing: the editable version is {st['version']} (state {st['state']}), "
                        f"not {version}. Re-check with appstore_status.")
            checks = await readiness(c, app_id)
            blk = blockers(checks)
            if blk:
                return ("NOT ready to submit — fix these first:\n"
                        + "\n".join(f"  ✗ {b['check']} — {b['hint']}" for b in blk))
            summary = (f"About to submit for App Review:\n"
                       f"  App: {app} (id {app_id})\n"
                       f"  Version: {st['version']}  |  Build: {st['build_number']}  |  Platform: iOS")
            if confirm != "submit":
                if not self._interactive:
                    return summary + "\n\nHeadless: refusing. Re-invoke with confirm=\"submit\" to proceed."
                return summary + "\n\nThis is a real submission to Apple. Re-invoke with confirm=\"submit\" to proceed."
            sub = (await c.post("/v1/reviewSubmissions", {"data": {"type": "reviewSubmissions",
                "attributes": {"platform": "IOS"},
                "relationships": {"app": {"data": {"type": "apps", "id": app_id}}}}})).get("data", {})
            sid = sub.get("id")
            if not sid:
                return ("App Store Connect error: submission was created but returned no id — "
                        "check App Store Connect; nothing was finalized.")
            await c.post("/v1/reviewSubmissionItems", {"data": {"type": "reviewSubmissionItems",
                "relationships": {
                    "reviewSubmission": {"data": {"type": "reviewSubmissions", "id": sid}},
                    "appStoreVersion": {"data": {"type": "appStoreVersions", "id": st["version_id"]}}}}})
            res = (await c.patch(f"/v1/reviewSubmissions/{sid}", {"data": {"type": "reviewSubmissions",
                "id": sid, "attributes": {"submitted": True}}})).get("data", {})
            state = res.get("attributes", {}).get("state", "SUBMITTED")
            return (f"✅ Submitted {app} {st['version']} (build {st['build_number']}) for review. "
                    f"reviewSubmission {sid} → state {state}. "
                    f"(A wrong submission can be canceled via PATCH reviewSubmissions/{sid} {{canceled:true}}.)")
        except AscError as e:
            return f"App Store Connect error: {e}"
