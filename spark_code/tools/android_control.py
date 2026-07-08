"""Control a connected Android phone — tap, type, swipe, key — with a hard gate.

Every acting action (tap/type/swipe/key) is gated: it resolves EXACTLY what it
will do (for a tap, it finds the target element via `uiautomator` so it hits the
real element, not a guessed pixel), and does nothing unless ``confirm="yes"`` is
passed. Combined with ``requires_permission=True`` (the human approves each call
in interactive mode), this is "confirm every action" on a real, non-sandboxed
phone. The ``find`` action is read-only (dump the tappable elements) so the model
can see what's on screen before proposing an action.

iOS is intentionally NOT supported here — arbitrary tap injection on iOS needs a
WebDriverAgent/Appium harness, out of scope. iOS stays view-only (ios_screenshot).
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET

from .android_screenshot import _resolve_adb
from .base import Tool

_ACTING = ("tap", "type", "swipe", "key")
_KEYS = {"back": "KEYCODE_BACK", "home": "KEYCODE_HOME", "enter": "KEYCODE_ENTER",
         "recent": "KEYCODE_APP_SWITCH", "power": "KEYCODE_POWER", "volup": "KEYCODE_VOLUME_UP",
         "voldown": "KEYCODE_VOLUME_DOWN", "delete": "KEYCODE_DEL"}
_BOUNDS = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def _escape_adb_text(text: str) -> str:
    """`adb shell input text` is re-parsed by the DEVICE shell: spaces must be
    %s and shell metacharacters escaped, or multi-word/special text is mangled."""
    out = []
    for ch in text:
        if ch == " ":
            out.append("%s")
        elif ch in "\\\"'`$&|;<>()!*?[]{}~#":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


async def _run(argv):
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


async def _dump_ui(base) -> tuple[str | None, str | None]:
    rc, _o, err = await _run(base + ["shell", "uiautomator", "dump", "/sdcard/spark_ui.xml"])
    if rc != 0:
        return None, (err.strip() or "uiautomator dump failed")
    rc, out, err = await _run(base + ["exec-out", "cat", "/sdcard/spark_ui.xml"])
    if rc != 0 or not out.strip():
        return None, (err.strip() or "could not read UI dump")
    return out, None


def _elements(xml_text):
    """Parse uiautomator XML → clickable/labeled nodes with center coords."""
    els = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return els
    for node in root.iter("node"):
        a = node.attrib
        label = (a.get("text") or "").strip() or (a.get("content-desc") or "").strip()
        m = _BOUNDS.match(a.get("bounds", ""))
        if not m or not label:
            continue
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        els.append({"label": label, "clickable": a.get("clickable") == "true",
                    "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2})
    return els


def _find(els, target):
    t = target.strip().lower()
    # exact clickable match first, then substring, preferring clickable
    ranked = sorted(els, key=lambda e: (not e["clickable"], len(e["label"])))
    for e in ranked:
        if e["label"].lower() == t and e["clickable"]:
            return e
    for e in ranked:
        if t in e["label"].lower():
            return e
    return None


class AndroidControlTool(Tool):
    @property
    def name(self) -> str:
        return "android_control"

    @property
    def description(self) -> str:
        return ("Control a connected Android phone. action='find' lists tappable on-screen elements "
                "(read-only). action='tap'/'type'/'swipe'/'key' ACT on the phone and require confirm='yes' "
                "(nothing happens without it). tap: target (element text) or x,y. type: text. swipe: "
                "direction up/down/left/right. key: back/home/enter/recent/delete. This is your REAL phone.")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "action": {"type": "string", "description": "find | tap | type | swipe | key"},
            "target": {"type": "string", "description": "For tap: visible text of the element to tap"},
            "text": {"type": "string", "description": "For type: the text to enter"},
            "x": {"type": "integer", "description": "For tap: explicit x coordinate (with y)"},
            "y": {"type": "integer", "description": "For tap: explicit y coordinate (with x)"},
            "direction": {"type": "string", "description": "For swipe: up | down | left | right"},
            "key": {"type": "string", "description": "For key: back | home | enter | recent | delete"},
            "confirm": {"type": "string", "description": "Pass 'yes' to actually perform the action"},
            "device": {"type": "string", "description": "adb serial (omit if only one device)"}},
            "required": ["action"]}

    @property
    def is_read_only(self) -> bool:
        return False

    @property
    def requires_permission(self) -> bool:
        return True

    async def execute(self, action: str = "", target: str = "", text: str = "", x=None, y=None,
                      direction: str = "", key: str = "", confirm: str = "", device: str = "",
                      **kwargs) -> str:
        adb = _resolve_adb()
        if not adb:
            return "Error: adb not found. Install Android platform-tools or set ANDROID_HOME."
        base = [adb] + (["-s", device] if device else [])
        act = (action or "").strip().lower()

        if act in ("find", "look", "screen", ""):
            xml_text, err = await _dump_ui(base)
            if err:
                return f"No Android device / UI dump failed: {err}. Plug in + enable USB debugging."
            els = _elements(xml_text)
            if not els:
                return "No labeled/tappable elements found on the current screen."
            lines = [f"  {'•' if e['clickable'] else '·'} {e['label']}  @({e['cx']},{e['cy']})"
                     for e in els[:40]]
            return "Tappable elements on screen (• = clickable):\n" + "\n".join(lines)

        if act not in _ACTING:
            return "action must be one of: find, tap, type, swipe, key."

        # Resolve the exact command + a human-readable summary of what it WILL do.
        if act == "tap":
            if isinstance(x, int) and isinstance(y, int):
                tx, ty, what = x, y, f"tap at ({x},{y})"
            elif target:
                xml_text, err = await _dump_ui(base)
                if err:
                    return f"No Android device / UI dump failed: {err}."
                el = _find(_elements(xml_text), target)
                if not el:
                    return (f"No on-screen element matching '{target}'. "
                            f"Run action='find' to see what's tappable.")
                tx, ty, what = el["cx"], el["cy"], f"tap '{el['label']}' at ({el['cx']},{el['cy']})"
            else:
                return "tap needs 'target' (element text) or both 'x' and 'y'."
            cmd = base + ["shell", "input", "tap", str(tx), str(ty)]
        elif act == "type":
            if not text:
                return "type needs 'text'."
            cmd = base + ["shell", "input", "text", _escape_adb_text(text)]
            what = f"type {text!r}"
        elif act == "swipe":
            dirs = {"up": (540, 1500, 540, 500), "down": (540, 500, 540, 1500),
                    "left": (900, 1000, 200, 1000), "right": (200, 1000, 900, 1000)}
            d = dirs.get(direction.strip().lower())
            if not d:
                return "swipe needs 'direction': up, down, left, or right."
            cmd = base + ["shell", "input", "swipe", *(str(v) for v in d), "300"]
            what = f"swipe {direction.lower()}"
        else:  # key
            kc = _KEYS.get(key.strip().lower())
            if not kc:
                return f"key must be one of: {', '.join(sorted(_KEYS))}."
            cmd = base + ["shell", "input", "keyevent", kc]
            what = f"press {key.lower()}"

        # HARD GATE — nothing touches the phone without confirm="yes".
        if confirm != "yes":
            return (f"About to {what} on your REAL phone. Nothing done yet — "
                    f"re-invoke with confirm=\"yes\" to perform it.")

        rc, _out, err = await _run(cmd)
        if rc != 0:
            return f"Action failed ({what}): {err.strip() or 'adb returned nonzero'}."
        return f"✅ Done: {what}."
