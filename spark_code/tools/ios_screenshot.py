"""Real-iPhone screenshot + vision-describe tool.

Captures the screen of a paired, trusted physical iPhone and feeds the PNG into
``vision.describe_image`` so a text-only primary model can "see" what's on
screen. Real-device counterpart to ``simulator.SimulatorScreenshotTool``.

iOS 17+ (incl. iOS 26) put developer services behind a RemoteXPC tunnel, so the
classic ``idevicescreenshot`` is dead ("Invalid service"). This uses
``pymobiledevice3 developer dvt screenshot --tunnel <udid>``, which reaches the
service through a running ``tunneld``. Going through ``--tunnel`` (rather than a
bare invocation) is what makes **Wi-Fi** work as well as USB: ``tunneld``
discovers the device over either transport and this command uses whichever
tunnel exists, instead of a USB-only lockdown bootstrap that fails cordless.

Flow: query ``tunneld`` (its HTTP API at 127.0.0.1:49151) for tunneled devices,
pick one, then capture through its tunnel. ``tunneld`` needs a one-time ``sudo``
to start (a launchd service can make it boot-persistent); this tool detects when
it isn't running and says so. Success is judged by the output file, since
``pymobiledevice3`` can exit 0 yet write nothing.

Read-only, no gate.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import httpx

from ..vision import VisionError, describe_image
from .base import Tool

TUNNELD_URL = "http://127.0.0.1:49151/"
_TUNNEL_HINT = (
    "iOS 17+ needs a running RemoteXPC tunnel (tunneld). Start it once and leave it running:\n"
    f"  sudo {sys.executable} -m pymobiledevice3 remote tunneld\n"
    "(pairing consent on the phone is one-time), then retry. USB and Wi-Fi both work once it's up.")


class IosScreenshotTool(Tool):
    @property
    def name(self) -> str:
        return "ios_screenshot"

    @property
    def description(self) -> str:
        return ("Capture a screenshot from a paired, trusted real iPhone (iOS 17+, via pymobiledevice3 "
                "over a running tunneld) and return a Gemini text description of the screen. Works over "
                "USB or Wi-Fi (same network). Optional: prompt (what to look for), udid.")

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

    async def _tunneled_devices(self) -> dict | None:
        """UDIDs tunneld currently has a tunnel for, or None if tunneld is unreachable."""
        try:
            async with httpx.AsyncClient(timeout=8.0) as c:
                r = await c.get(TUNNELD_URL)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return None

    async def execute(self, prompt: str = "", udid: str = "", **kwargs) -> str:
        devices = await self._tunneled_devices()
        if devices is None:
            return f"No iPhone captured — tunneld isn't running. {_TUNNEL_HINT}"
        if not devices:
            return ("No iPhone captured — the tunnel is up but sees no device. Connect the iPhone via "
                    "USB, or for Wi-Fi make sure it's unlocked, trusted, and on the same network as this Mac.")
        target = udid or next(iter(devices))
        if target not in devices:
            return (f"No iPhone captured — device '{target}' isn't in the tunnel. "
                    f"Available: {', '.join(devices)}.")

        path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        argv = [sys.executable, "-m", "pymobiledevice3", "developer", "dvt", "screenshot",
                "--tunnel", target, path]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _out, err = await proc.communicate()
        except OSError as e:
            self._cleanup(path)
            return f"Error launching pymobiledevice3: {e}"
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            self._cleanup(path)
            detail = (err.decode("utf-8", "replace").strip() or "no output")[-300:]
            return (f"No iPhone captured — {detail}. If on Wi-Fi, wake/unlock the phone and keep it on "
                    f"this network; otherwise reconnect USB.")
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
