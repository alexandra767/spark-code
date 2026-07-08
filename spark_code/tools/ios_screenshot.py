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
                 "return a Gemini text description of the screen. Works over USB (default) or, with "
                 "network=true, over Wi-Fi (needs an initial USB pairing + Wi-Fi sync enabled). "
                 "Optional: prompt (what to look for), network (Wi-Fi), udid (target a specific device).")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "prompt": {"type": "string", "description": "What to look for / ask about the screen"},
            "network": {"type": "boolean", "description": "Capture over Wi-Fi instead of USB (default false)"},
            "udid": {"type": "string", "description": "Target a specific device by UDID (omit if only one)"}}}

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def requires_permission(self) -> bool:
        return False

    async def execute(self, prompt: str = "", network=False, udid: str = "", **kwargs) -> str:
        if not which("idevicescreenshot"):
            return ("Error: idevicescreenshot not installed "
                    "(brew install libimobiledevice) — needed for real-iPhone capture.")
        # `network` may arrive as a real bool or a "true"/"false" string.
        wifi = str(network).strip().lower() in ("true", "1", "yes", "on")
        path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        argv = (["idevicescreenshot"]
                + (["-n"] if wifi else [])
                + (["-u", udid] if udid else [])
                + [path])
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
            try:
                os.remove(path)
            except OSError:
                pass
            detail = (err.decode("utf-8", "replace").strip() or "no output")[:300]
            hint = ("Pair the iPhone over USB and trust this Mac (unlock the phone). For Wi-Fi "
                    "(network=true), also enable Wi-Fi sync and keep the phone on this network. "
                    "idevicescreenshot also needs a mounted Developer Disk Image (connect the phone "
                    "to Xcode once).")
            return f"No iPhone captured — {detail}. {hint}"
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
