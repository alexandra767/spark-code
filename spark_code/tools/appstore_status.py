"""Read-only App Store Connect status tool.

Reports an iOS app's current App Store version state, attached build, and a
submission-readiness checklist (see spark_code.appstore.readiness) — no
writes, no submissions. Safe to expose without a permission prompt.
"""
from __future__ import annotations

from ..appstore import AscClient, AscError, readiness
from .base import Tool


class AppStoreStatusTool(Tool):
    @property
    def name(self) -> str:
        return "appstore_status"

    @property
    def description(self) -> str:
        return ("Read-only: show an iOS app's current App Store version state, attached "
                "build, and a submission-readiness checklist. Input: app (bundle id or name).")

    @property
    def parameters(self) -> dict:
        return {"type": "object",
                "properties": {"app": {"type": "string",
                    "description": "App bundle id (preferred) or exact name."}},
                "required": ["app"]}

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def requires_permission(self) -> bool:
        return False

    async def execute(self, app: str = "", **kwargs) -> str:
        if not app:
            return "Error: 'app' (bundle id or name) is required."
        try:
            c = AscClient()
            app_id = await c.resolve_app(app)
            st = await c.get_submission_state(app_id)
            checks = await readiness(c, app_id)
        except AscError as e:
            return f"App Store Connect error: {e}"
        lines = [f"App: {app}  (id {app_id})",
                 f"Version {st['version']} — state: {st['state']}",
                 f"Build: {st['build_number'] or '(none attached)'} "
                 f"[{st['build_processing_state'] or '-'}]",
                 "Readiness:"]
        for c_ in checks:
            lines.append(f"  {'✓' if c_['ok'] else '✗'} {c_['check']}"
                         + (f" — {c_['hint']}" if not c_['ok'] and c_['hint'] else ""))
        blk = [c_ for c_ in checks if not c_['ok']]
        lines.append("READY to submit." if not blk else f"NOT ready — {len(blk)} blocker(s).")
        return "\n".join(lines)
