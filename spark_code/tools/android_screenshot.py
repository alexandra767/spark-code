"""Android screenshot + vision-describe tool.

Captures the screen of a connected Android device (USB debugging enabled)
via ``adb exec-out screencap -p`` (raw PNG on stdout — no on-device temp
file, no pull step) and feeds the bytes into ``vision.describe_image`` so a
text-only primary model can "see" what's on screen.

Read-only, no gate: it only captures pixels and asks Gemini to describe
them — never touches app/device state. No device or no ``adb`` binary
returns a clean, actionable message and never calls ``describe_image``.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from shutil import which

from ..vision import VisionError, describe_image
from .base import Tool


def _resolve_adb():
    """Locate the adb binary: PATH first, then common SDK install locations.

    Returns the resolved path as a string, or None if adb can't be found
    anywhere. Factored out as a module-level function (rather than inlined)
    so tests can monkeypatch it directly without needing a real Android SDK
    installed.
    """
    p = which("adb")
    if p:
        return p
    candidates = [Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb"]
    ah = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if ah:
        candidates.append(Path(ah) / "platform-tools" / "adb")
    for c in candidates:
        if c.exists():
            return str(c)
    return None


class AndroidScreenshotTool(Tool):
    @property
    def name(self) -> str:
        return "android_screenshot"

    @property
    def description(self) -> str:
        return ("Capture a screenshot from a connected Android device (USB debugging) and return "
                 "a Gemini-generated text description of what's on screen. Optional: device (serial), "
                 "prompt (what to look for).")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "device": {"type": "string", "description": "adb serial (omit if only one device)"},
            "prompt": {"type": "string", "description": "What to look for / ask about the screen"}}}

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def requires_permission(self) -> bool:
        return False

    async def execute(self, device: str = "", prompt: str = "", **kwargs) -> str:
        adb = _resolve_adb()
        if not adb:
            return "Error: adb not found. Install Android platform-tools or set ANDROID_HOME."
        argv = [adb] + (["-s", device] if device else []) + ["exec-out", "screencap", "-p"]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode != 0 or not out:
            detail = (err.decode("utf-8", "replace").strip() or "no output")[:300]
            return (f"No Android device captured — {detail}. "
                    f"Enable USB debugging and plug in the phone (check `adb devices`).")
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(out)
                path = f.name
            desc = await describe_image(path, prompt or None)
        except VisionError as e:
            return f"Captured the screen but vision failed: {e}"
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
        return f"📱 Android screenshot — Gemini sees:\n{desc}"
