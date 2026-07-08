"""Launch an app on a connected REAL iOS device via Apple's devicectl (CoreDevice).

Launch-only: it brings an app to the foreground on the physical iPhone/iPad. It
does NOT tap/type/drive the UI (that needs a WebDriverAgent harness, out of
scope) — pair it with ``ios_screenshot`` to SEE the result. Uses
``xcrun devicectl``, which works on iOS 17+ over USB or Wi-Fi and manages its own
connection (no tunneld needed). This targets the real device, never the
Simulator (that's ``simulator_screenshot`` / ``xcrun simctl``).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from shutil import which

from .base import Tool

_APP_TYPES = ("iPhone", "iPad")


async def _run(argv):
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


async def _devicectl_json(args):
    """Run a devicectl subcommand with --json-output to a temp file → (parsed, err).

    devicectl writes structured JSON only to a file path (not stdout), so we use
    a temp file and read it back — far more robust than scraping the table output.
    """
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        rc, out, err = await _run(["xcrun", "devicectl", *args, "--json-output", path])
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None, (err.strip() or out.strip() or "devicectl produced no output")
        if rc != 0:
            return None, (err.strip() or "devicectl returned nonzero")
        return data, None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _pick_device(devices, udid=""):
    """Choose the target: explicit udid, else prefer a reachable iPhone over iPad.

    `tunnelState` fluctuates ("connected"/"disconnected") because devicectl
    connects on demand; only "unavailable" means genuinely unreachable, so we
    exclude that and otherwise prefer iPhone, then a currently-connected one.
    """
    if udid:
        for d in devices:
            if d.get("identifier") == udid:
                return d, udid
        return None, ""
    usable = []
    for d in devices:
        dtype = (d.get("hardwareProperties") or {}).get("deviceType", "")
        state = (d.get("connectionProperties") or {}).get("tunnelState", "")
        if dtype in _APP_TYPES and state != "unavailable":
            usable.append((d, d.get("identifier", ""), dtype, state))
    usable.sort(key=lambda t: (t[2] != "iPhone", t[3] != "connected"))
    if usable:
        return usable[0][0], usable[0][1]
    return None, ""


def _resolve_bundle(apps, query):
    """Fuzzy-match an app name / bundle id against installed apps → (app, alternatives).

    Matches on BOTH the display name and the bundle id (so 'dometic' finds
    'Mobile Cooling' = com.dometic.cfx3). Exact bundle id wins; then name matches
    rank before bundle-only matches; visible/non-internal and shorter names first.
    """
    q = (query or "").strip().lower()
    if not q:
        return None, []
    for a in apps:
        if (a.get("bundleIdentifier") or "").lower() == q:
            return a, []
    matches = [a for a in apps
               if q in (a.get("name") or "").lower()
               or q in (a.get("bundleIdentifier") or "").lower()]
    if not matches:
        return None, []
    matches.sort(key=lambda a: (q not in (a.get("name") or "").lower(),
                                bool(a.get("hidden")), bool(a.get("internalApp")),
                                len(a.get("name") or "")))
    return matches[0], matches[1:]


class IosLaunchTool(Tool):
    @property
    def name(self) -> str:
        return "ios_launch"

    @property
    def description(self) -> str:
        return ("Open/launch an app on a connected REAL iPhone/iPad (NOT the Simulator). "
                "app = app name or bundle id (e.g. 'dometic', 'gigledger'). Brings it to the "
                "foreground on the physical device via Apple devicectl (USB or Wi-Fi, iOS 17+). "
                "This is how you OPEN an app on iOS. It only LAUNCHES — it cannot tap/type/scroll; "
                "use ios_screenshot to see the screen afterward.")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "app": {"type": "string", "description": "App name or bundle id to open (e.g. 'dometic')"},
            "udid": {"type": "string", "description": "Device UDID (omit to auto-pick the connected iPhone)"}},
            "required": ["app"]}

    @property
    def is_read_only(self) -> bool:
        return False

    @property
    def requires_permission(self) -> bool:
        return True

    async def execute(self, app: str = "", udid: str = "", **kwargs) -> str:
        if not which("xcrun"):
            return "Error: `xcrun` not found. Xcode command-line tools are required for iOS device control."
        if not (app or "").strip():
            return "ios_launch needs 'app' (an app name like 'dometic' or a bundle id)."
        devdata, err = await _devicectl_json(["list", "devices"])
        if err:
            return f"Couldn't list iOS devices: {err}."
        devices = (devdata.get("result", {}) or {}).get("devices", [])
        dev, dudid = _pick_device(devices, udid.strip())
        if not dev:
            return ("No connected iPhone/iPad found. Plug it in (or join the same Wi-Fi), unlock it, "
                    "and trust this Mac — `xcrun devicectl list devices` should show it.")
        appdata, err = await _devicectl_json(
            ["device", "info", "apps", "--device", dudid, "--include-default-apps"])
        if err:
            return f"Couldn't list apps on the device: {err}."
        apps = (appdata.get("result", {}) or {}).get("apps", [])
        match, alts = _resolve_bundle(apps, app)
        if not match:
            return f"No app matching '{app}' on the device. Try the full bundle id, or a different name."
        bundle = match.get("bundleIdentifier")
        rc, out, err = await _run(
            ["xcrun", "devicectl", "device", "process", "launch", "--device", dudid, bundle])
        if rc != 0 or "Launched application" not in (out + err):
            reason = (err.strip() or out.strip() or "launch failed")[:300]
            return f"Couldn't launch {match.get('name')} ({bundle}): {reason}."
        dname = (dev.get("deviceProperties") or {}).get("name", "device")
        note = f" (other matches: {', '.join(a.get('name', '') for a in alts[:3])})" if alts else ""
        return f"✅ Opened {match.get('name')} ({bundle}) on {dname}.{note}"
