"""Real-iPhone screenshot + vision-describe tool.

Captures the screen of a paired, trusted physical iPhone and feeds the PNG into
``vision.describe_image`` so a text-only primary model can "see" what's on
screen. Real-device counterpart to ``simulator.SimulatorScreenshotTool``.

iOS 17+ (incl. iOS 26) moved developer services behind a RemoteXPC tunnel, so
the classic ``idevicescreenshot`` (libimobiledevice) can no longer screenshot
them ("Could not start screenshotr service: Invalid service"). This uses
``pymobiledevice3 developer dvt screenshot``, which reaches the service over a
running ``tunneld``. ``tunneld`` requires a one-time ``sudo`` the user starts;
this tool detects when it isn't running and returns the exact command. USB and
Wi-Fi both work transparently — ``tunneld`` discovers the device either way.

NOTE: ``pymobiledevice3`` can exit 0 even when it fails to capture (it logs an
error and writes no file), so success is judged by the output file existing and
being non-empty, not by the return code alone.

Read-only, no gate.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

from ..vision import VisionError, describe_image
from .base import Tool

_TUNNEL_HINT = (
    "iOS 17+ needs a running RemoteXPC tunnel. Start it once in a Terminal and leave it open:\n"
    f"  sudo {sys.executable} -m pymobiledevice3 remote tunneld\n"
    "(the pairing consent on the phone is one-time), then retry.")


class IosScreenshotTool(Tool):
    @property
    def name(self) -> str:
        return "ios_screenshot"

    @property
    def description(self) -> str:
        return ("Capture a screenshot from a paired, trusted real iPhone (iOS 17+, via "
                "pymobiledevice3 over a running tunneld) and return a Gemini text description of the "
                "screen. Works over USB or Wi-Fi. Optional: prompt (what to look for), udid.")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "prompt": {"type": "string", "description": "What to look for / ask about the screen"},
            "udid": {"type": "string", "description": "Target a specific device by UDID (omit if only one)"}}}

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def requires_permission(self) -> bool:
        return False

    async def execute(self, prompt: str = "", udid: str = "", **kwargs) -> str:
        path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        argv = [sys.executable, "-m", "pymobiledevice3", "developer", "dvt", "screenshot", path]
        if udid:
            argv += ["--udid", udid]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _out, err = await proc.communicate()
        except OSError as e:
            self._cleanup(path)
            return f"Error launching pymobiledevice3: {e}"
        errtext = err.decode("utf-8", "replace")
        # pymobiledevice3 can exit 0 yet write no file on failure — judge by the file.
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            self._cleanup(path)
            low = errtext.lower()
            if "tunnel" in low:
                return f"No iPhone captured — the tunnel isn't running. {_TUNNEL_HINT}"
            if "no device" in low or "not found" in low or "muxdevice" in low:
                return ("No iPhone captured — no paired device found. Plug the iPhone in via USB "
                        "(or Wi-Fi with the tunnel up), unlock it, and trust this Mac.")
            detail = (errtext.strip() or "no output")[-300:]
            return f"No iPhone captured — {detail}\n{_TUNNEL_HINT}"
        try:
            desc = await describe_image(path, prompt or None)
        except VisionError as e:
            return f"Captured the screen but vision failed: {e}"
        finally:
            self._cleanup(path)
        return f"📱 iPhone screenshot — Gemini sees:\n{desc}"

    @staticmethod
    def _cleanup(path) -> None:
        try:
            os.remove(path)
        except OSError:
            pass
