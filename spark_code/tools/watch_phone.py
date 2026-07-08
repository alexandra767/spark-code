"""Watch a phone screen on an interval until a condition is met.

Polls the device (Android via adb, iPhone via the pymobiledevice3 tunnel),
asks Gemini after each capture whether the watched condition is now true, and
stops the moment it is (showing that screen) — or gives up after a bounded
number of checks. Useful for "watch my Dasher app and tell me when an offer
pops" while doing something else.

Read-only. Bounded by max_checks so it can't run forever; each check costs one
Gemini call, so keep intervals sane.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

import httpx

from ..vision import DISPLAY_IMAGE_SENTINEL, VisionError, describe_image
from .android_screenshot import _resolve_adb
from .base import Tool

_TUNNELD_URL = "http://127.0.0.1:49151/"


def _toint(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


async def _capture_android(device: str):
    adb = _resolve_adb()
    if not adb:
        return None, "adb not found (install Android platform-tools)."
    argv = [adb] + (["-s", device] if device else []) + ["exec-out", "screencap", "-p"]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0 or not out:
        return None, (err.decode("utf-8", "replace").strip() or "no output")
    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    with open(path, "wb") as f:
        f.write(out)
    return path, None


async def _capture_ios(udid: str):
    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.get(_TUNNELD_URL)
        devices = r.json() if r.status_code == 200 else {}
    except Exception:
        return None, "tunneld isn't running (start: sudo <venv>/python -m pymobiledevice3 remote tunneld)."
    if not devices:
        return None, "tunnel is up but sees no iPhone (USB, or Wi-Fi unlocked + same network)."
    target = udid or next(iter(devices))
    if target not in devices:
        return None, f"device '{target}' not in the tunnel (available: {', '.join(devices)})."
    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    argv = [sys.executable, "-m", "pymobiledevice3", "developer", "dvt", "screenshot", "--tunnel", target, path]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _out, err = await proc.communicate()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        _rm(path)
        return None, (err.decode("utf-8", "replace").strip() or "no output")
    return path, None


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


class WatchPhoneTool(Tool):
    @property
    def name(self) -> str:
        return "watch_phone"

    @property
    def description(self) -> str:
        return ("Watch a phone screen on an interval until a condition happens, then report it. "
                "Captures the screen (Android via adb, iPhone via the tunnel), and Gemini checks each "
                "shot for your condition. Input: watch_for (what to wait for), platform (android|ios), "
                "interval seconds, max_checks. Stops early when the condition is met. Each check costs "
                "one Gemini call, so it's bounded by max_checks.")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "watch_for": {"type": "string", "description": "The condition to wait for, e.g. 'a DoorDash offer appears'"},
            "platform": {"type": "string", "description": "android (default) or ios"},
            "interval": {"type": "integer", "description": "Seconds between checks (default 15, 5-120)"},
            "max_checks": {"type": "integer", "description": "Max checks before giving up (default 20, 1-60)"},
            "device": {"type": "string", "description": "Android adb serial (optional)"},
            "udid": {"type": "string", "description": "iPhone UDID (optional)"}},
            "required": ["watch_for"]}

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def requires_permission(self) -> bool:
        return False

    async def execute(self, watch_for: str = "", platform: str = "android", interval=15,
                      max_checks=20, device: str = "", udid: str = "", **kwargs) -> str:
        if not watch_for:
            return "Error: 'watch_for' (what to watch the screen for) is required."
        interval = max(5, min(_toint(interval, 15), 120))
        max_checks = max(1, min(_toint(max_checks, 20), 60))
        is_ios = (platform or "android").lower().startswith(("ios", "iphone"))
        prompt = (f"You are watching a phone screen for this condition: '{watch_for}'. On the FIRST "
                  "line reply with exactly YES if that condition is true on this screen right now, "
                  "otherwise NO. Then one short line saying what's on screen.")
        last = ""
        for i in range(max_checks):
            path, err = await (_capture_ios(udid) if is_ios else _capture_android(device))
            if err:
                return f"Watch stopped — capture failed on check {i + 1}: {err}"
            try:
                desc = await describe_image(path, prompt)
            except VisionError as e:
                _rm(path)
                return f"Watch stopped — vision failed: {e}"
            last = desc
            if desc.strip().upper().startswith("YES"):
                return (f"✅ '{watch_for}' — detected on check {i + 1} (~{i * interval}s in):\n"
                        f"{desc}{DISPLAY_IMAGE_SENTINEL}{path}")
            _rm(path)
            if i < max_checks - 1:
                await asyncio.sleep(interval)
        return (f"Watched {max_checks}× over ~{max_checks * interval}s — '{watch_for}' not detected. "
                f"Last screen: {last}")
