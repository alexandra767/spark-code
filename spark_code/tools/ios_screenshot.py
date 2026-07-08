"""Real-iPhone screenshot + vision-describe tool.

Captures the screen of a paired, trusted physical iPhone via
``idevicescreenshot`` (from ``libimobiledevice`` — ``brew install
libimobiledevice``) and feeds the resulting PNG into
``vision.describe_image`` so a text-only primary model can "see" what's on
screen. This is the real-device counterpart to ``simulator.
SimulatorScreenshotTool`` (which only targets the iOS Simulator).

Read-only, no gate: it only captures pixels and asks Gemini to describe
them. No ``idevicescreenshot`` binary, or no paired/trusted device (rc != 0,
or the output file is missing/0 bytes), returns a clean, actionable message
and never calls ``describe_image``.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from shutil import which

from ..vision import VisionError, describe_image
from .base import Tool


class IosScreenshotTool(Tool):
    @property
    def name(self) -> str:
        return "ios_screenshot"

    @property
    def description(self) -> str:
        return ("Capture a screenshot from a paired, trusted real iPhone (via idevicescreenshot) and "
                 "return a Gemini text description of the screen. Optional: prompt (what to look for).")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "prompt": {"type": "string", "description": "What to look for / ask about the screen"}}}

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def requires_permission(self) -> bool:
        return False

    async def execute(self, prompt: str = "", **kwargs) -> str:
        if not which("idevicescreenshot"):
            return ("Error: idevicescreenshot not installed "
                    "(brew install libimobiledevice) — needed for real-iPhone capture.")
        path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        proc = await asyncio.create_subprocess_exec(
            "idevicescreenshot", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
            try:
                os.remove(path)
            except OSError:
                pass
            detail = (err.decode("utf-8", "replace").strip() or "no output")[:300]
            return (f"No iPhone captured — {detail}. "
                    f"Pair the iPhone and trust this Mac (unlock the phone), then retry.")
        try:
            desc = await describe_image(path, prompt or None)
        except VisionError as e:
            return f"Captured the screen but vision failed: {e}"
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return f"📱 iPhone screenshot — Gemini sees:\n{desc}"
