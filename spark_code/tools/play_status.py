"""Read-only Google Play status tool.

Reports an Android app's current Google Play track/version state (production,
beta, alpha, internal, plus wear:* tracks) — no writes, no submissions. Safe
to expose without a permission prompt.
"""
from __future__ import annotations

from ..play import PlayClient, PlayError
from .base import Tool


class PlayStatusTool(Tool):
    @property
    def name(self) -> str: return "play_status"

    @property
    def description(self) -> str:
        return ("Read-only: show a Google Play app's tracks (production/beta/alpha/internal, "
                "plus wear:*), each with its current version name, versionCode(s), and release "
                "status. Input: package (e.g. com.gigledger.android).")

    @property
    def parameters(self) -> dict:
        return {"type": "object",
                "properties": {"package": {"type": "string",
                    "description": "Android package name, e.g. com.gigledger.android"}},
                "required": ["package"]}

    @property
    def is_read_only(self) -> bool: return True

    @property
    def requires_permission(self) -> bool: return False

    async def execute(self, package: str = "", **kwargs) -> str:
        if not package:
            return "Error: 'package' (e.g. com.gigledger.android) is required."
        try:
            tracks = await PlayClient().track_status(package)
        except PlayError as e:
            return f"Google Play error: {e}"
        if not tracks:
            return f"{package}: no tracks found."
        lines = [f"Google Play — {package}:"]
        for t in tracks:
            vc = ", ".join(t["version_codes"]) if t["version_codes"] else "-"
            if t["version_name"] or t["version_codes"]:
                lines.append(f"  {t['track']}: {t['version_name'] or '?'} (vc {vc}) — {t['status'] or '?'}")
            else:
                lines.append(f"  {t['track']}: (no release)")
        return "\n".join(lines)
