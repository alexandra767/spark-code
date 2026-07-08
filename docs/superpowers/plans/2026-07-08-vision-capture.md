# Phone Screenshot Capture + Gemini Vision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `describe_image()` (image → Gemini → text) + `android_screenshot` / `ios_screenshot` tools that capture a real phone screen and return a Gemini description the text-only 80B can use.

**Architecture:** `spark_code/vision.py` builds a Gemini `ModelClient` and sends the image as an OpenAI `image_url` part, returning the text. The two capture tools grab a PNG (`adb exec-out screencap -p` / `idevicescreenshot`) and call `describe_image`.

**Tech Stack:** Python 3.12 (asyncio subprocess, base64), pytest. No new deps.

## Global Constraints

- Changes land in the live `~/spark-code`, branch `feat/vision-capture`.
- `asyncio_mode = "auto"` (async tests no decorator); monkeypatch, not respx.
- `describe_image` uses provider `gemini-pro` from `~/.spark/config.yaml` (`config.load_config()`), builds a `ModelClient(supports_vision=True)`, sends `[{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/png;base64,<b64>"}}]`, returns collected text; raises `VisionError` on missing provider / unreadable image / empty response.
- Capture tools are read-only (`is_read_only=True`, `requires_permission=False`); no device connected → clean message, `describe_image` NOT called; temp PNG deleted after.
- Android PNG via `adb exec-out screencap -p` (exec-out avoids CRLF corruption).

---

### Task 1: `vision.py` — Gemini image describer

**Files:**
- Create: `spark_code/vision.py`
- Test: `tests/test_vision.py`

**Interfaces produced:** `VisionError`; `async describe_image(image_path, prompt=None, provider_name="gemini-pro", config=None) -> str`; `DEFAULT_VISION_PROMPT`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vision.py
import pytest
from pathlib import Path
from spark_code.vision import describe_image, VisionError, DEFAULT_VISION_PROMPT

CFG = {"providers": {"gemini-pro": {"endpoint": "https://gen.example/v1beta/openai",
    "model": "gemini-2.5-pro", "max_tokens": 1024, "api_key": "k"}}}

async def test_describe_image_sends_image_and_returns_text(tmp_path, monkeypatch):
    img = tmp_path / "s.png"; img.write_bytes(b"\x89PNG_fake_bytes")
    seen = {}
    class FakeClient:
        def __init__(self, **kw): seen["init"] = kw
        async def chat(self, messages, tools=None, stream=True):
            seen["messages"] = messages
            yield {"type": "text", "content": "A login screen with "}
            yield {"type": "text", "content": "an email field and a blue Submit button."}
            yield {"type": "done", "usage": {}}
        async def close(self): seen["closed"] = True
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    out = await describe_image(str(img), config=CFG)
    assert "blue Submit button" in out and out == out.strip()
    assert seen["init"]["supports_vision"] is True and seen["init"]["provider"] == "gemini-pro"
    content = seen["messages"][0]["content"]
    assert content[0]["type"] == "text" and DEFAULT_VISION_PROMPT[:10] in content[0]["text"]
    assert content[1]["type"] == "image_url" and content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["closed"] is True

async def test_describe_image_custom_prompt(tmp_path, monkeypatch):
    img = tmp_path / "s.png"; img.write_bytes(b"x")
    class FakeClient:
        def __init__(self, **kw): pass
        async def chat(self, messages, tools=None, stream=True):
            FakeClient.msgs = messages
            yield {"type": "text", "content": "ok"}
        async def close(self): pass
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    await describe_image(str(img), prompt="Just read the error text.", config=CFG)
    assert FakeClient.msgs[0]["content"][0]["text"] == "Just read the error text."

async def test_missing_provider_raises(tmp_path):
    img = tmp_path / "s.png"; img.write_bytes(b"x")
    with pytest.raises(VisionError):
        await describe_image(str(img), config={"providers": {}})

async def test_unreadable_image_raises(tmp_path):
    with pytest.raises(VisionError):
        await describe_image(str(tmp_path / "nope.png"), config=CFG)

async def test_empty_description_raises(tmp_path, monkeypatch):
    img = tmp_path / "s.png"; img.write_bytes(b"x")
    class FakeClient:
        def __init__(self, **kw): pass
        async def chat(self, messages, tools=None, stream=True):
            yield {"type": "done", "usage": {}}
        async def close(self): pass
    monkeypatch.setattr("spark_code.vision.ModelClient", FakeClient)
    with pytest.raises(VisionError):
        await describe_image(str(img), config=CFG)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_vision.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `vision.py`**

```python
# spark_code/vision.py
"""Describe an image via a vision-capable provider (Gemini) so a text-only
primary model can work with what's on screen."""
from __future__ import annotations
import base64
from pathlib import Path

from .config import load_config, load_keys, resolve_provider_key
from .model import ModelClient

DEFAULT_VISION_PROMPT = (
    "You are the eyes for another AI. Describe this screen in detail: all visible "
    "text, UI elements, buttons, layout, current state, and anything notable. Be "
    "thorough and literal.")


class VisionError(Exception):
    pass


async def describe_image(image_path, prompt=None, provider_name="gemini-pro", config=None) -> str:
    cfg = config or load_config()
    pconf = (cfg.get("providers", {}) or {}).get(provider_name)
    if not pconf:
        raise VisionError(f"vision provider '{provider_name}' is not configured in ~/.spark/config.yaml")
    try:
        data = Path(image_path).read_bytes()
    except OSError as e:
        raise VisionError(f"could not read image {image_path}: {e}") from e
    client = ModelClient(
        endpoint=pconf.get("endpoint", ""),
        model=pconf.get("model", ""),
        temperature=0.2,
        max_tokens=pconf.get("max_tokens", 1024),
        api_key=resolve_provider_key(provider_name, pconf, load_keys()),
        provider=provider_name,
        supports_vision=True)
    b64 = base64.b64encode(data).decode("utf-8")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt or DEFAULT_VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
    text = ""
    try:
        async for chunk in client.chat(messages, stream=False):
            if chunk.get("type") == "text":
                text += chunk.get("content", "")
    finally:
        await client.close()
    text = text.strip()
    if not text:
        raise VisionError("vision provider returned no description")
    return text
```

- [ ] **Step 4: Run tests + ruff**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_vision.py -v` (PASS, 5) then `.venv/bin/python -m ruff check spark_code/vision.py tests/test_vision.py`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/vision.py tests/test_vision.py
git commit -m "feat: vision.describe_image — send an image to Gemini, return a text description"
```

---

### Task 2: `android_screenshot` + `ios_screenshot` tools + registration

**Files:**
- Create: `spark_code/tools/android_screenshot.py`, `spark_code/tools/ios_screenshot.py`
- Modify: `spark_code/cli.py` (register both in `build_tools`)
- Test: `tests/test_screenshot_tools.py`

**Interfaces:** `AndroidScreenshotTool` / `IosScreenshotTool` — `is_read_only=True`, `requires_permission=False`. Android params `device`, `prompt`; iOS param `prompt`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_screenshot_tools.py
import pytest
from spark_code.tools.android_screenshot import AndroidScreenshotTool, _resolve_adb
from spark_code.tools.ios_screenshot import IosScreenshotTool

class _Proc:
    def __init__(self, rc, out=b"", err=b""): self._rc, self._out, self._err = rc, out, err
    async def communicate(self): return self._out, self._err
    @property
    def returncode(self): return self._rc

async def test_android_no_device_clean_message(monkeypatch):
    monkeypatch.setattr("spark_code.tools.android_screenshot._resolve_adb", lambda: "/adb")
    async def fake_exec(*a, **k): return _Proc(1, b"", b"adb: no devices/emulators found")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    called = {"describe": False}
    async def fake_desc(*a, **k): called["describe"] = True; return "x"
    monkeypatch.setattr("spark_code.tools.android_screenshot.describe_image", fake_desc)
    out = await AndroidScreenshotTool().execute()
    assert "no" in out.lower() and "device" in out.lower() and called["describe"] is False

async def test_android_no_adb(monkeypatch):
    monkeypatch.setattr("spark_code.tools.android_screenshot._resolve_adb", lambda: None)
    out = await AndroidScreenshotTool().execute()
    assert "adb" in out.lower()

async def test_android_captures_and_describes(monkeypatch):
    monkeypatch.setattr("spark_code.tools.android_screenshot._resolve_adb", lambda: "/adb")
    async def fake_exec(*a, **k): return _Proc(0, b"\x89PNGdata", b"")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    seen = {}
    async def fake_desc(path, prompt=None, **k):
        seen["path"] = path; seen["prompt"] = prompt
        return "Home screen with a map and a Start button."
    monkeypatch.setattr("spark_code.tools.android_screenshot.describe_image", fake_desc)
    out = await AndroidScreenshotTool().execute(prompt="what's on screen?")
    assert "Start button" in out and seen["prompt"] == "what's on screen?"
    assert AndroidScreenshotTool().is_read_only is True

async def test_ios_no_binary(monkeypatch):
    monkeypatch.setattr("spark_code.tools.ios_screenshot.which", lambda n: None)
    out = await IosScreenshotTool().execute()
    assert "idevicescreenshot" in out.lower()

async def test_ios_no_device_clean_message(monkeypatch):
    monkeypatch.setattr("spark_code.tools.ios_screenshot.which", lambda n: "/idevicescreenshot")
    async def fake_exec(*a, **k): return _Proc(1, b"", b"No device found")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    called = {"d": False}
    async def fake_desc(*a, **k): called["d"] = True; return "x"
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)
    out = await IosScreenshotTool().execute()
    assert "device" in out.lower() and called["d"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_screenshot_tools.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the tools**

```python
# spark_code/tools/android_screenshot.py
from __future__ import annotations
import asyncio
import os
import tempfile
from pathlib import Path
from shutil import which
from .base import Tool
from ..vision import describe_image, VisionError


def _resolve_adb():
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
    def name(self) -> str: return "android_screenshot"

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
    def is_read_only(self) -> bool: return True

    @property
    def requires_permission(self) -> bool: return False

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
```
```python
# spark_code/tools/ios_screenshot.py
from __future__ import annotations
import asyncio
import os
import tempfile
from shutil import which
from .base import Tool
from ..vision import describe_image, VisionError


class IosScreenshotTool(Tool):
    @property
    def name(self) -> str: return "ios_screenshot"

    @property
    def description(self) -> str:
        return ("Capture a screenshot from a paired, trusted real iPhone (via idevicescreenshot) and "
                "return a Gemini text description of the screen. Optional: prompt (what to look for).")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "prompt": {"type": "string", "description": "What to look for / ask about the screen"}}}

    @property
    def is_read_only(self) -> bool: return True

    @property
    def requires_permission(self) -> bool: return False

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
```
Register in `cli.py` `build_tools` (near `SimulatorScreenshotTool`):
```python
    from .tools.android_screenshot import AndroidScreenshotTool
    from .tools.ios_screenshot import IosScreenshotTool
    registry.register(AndroidScreenshotTool())
    registry.register(IosScreenshotTool())
```

- [ ] **Step 4: Run tests**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_screenshot_tools.py -v` (PASS, 5) then `.venv/bin/python -m pytest tests/ -q | tail -2` and `.venv/bin/python -m ruff check spark_code/tools/android_screenshot.py spark_code/tools/ios_screenshot.py spark_code/cli.py tests/test_screenshot_tools.py`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/tools/android_screenshot.py spark_code/tools/ios_screenshot.py spark_code/cli.py tests/test_screenshot_tools.py
git commit -m "feat: android_screenshot + ios_screenshot tools — capture a phone screen, describe via Gemini"
```

## Live acceptance (controller-run, after Task 2 — no phone needed)

1. **Vision core (ground truth):** write a PNG containing a known random token, then:
```bash
cd ~/spark-code && .venv/bin/python -c "
import asyncio; from spark_code.vision import describe_image
print(asyncio.run(describe_image('<token.png>', prompt='Read the exact text shown.')))"
```
Expected: Gemini's description contains the token → proves image→Gemini→text end to end.
2. **No-device path:** run `android_screenshot` / `ios_screenshot` with nothing attached → expect the clean "no device" messages (real, since none is connected).
Full real-screen capture is deferred until a device is plugged in.

## Self-Review

- **Spec coverage:** `describe_image` core (T1) + both capture tools with no-device clean paths + registration (T2); live acceptance = token-image vision proof + no-device path. ✓
- **Placeholders:** none. ✓
- **Type consistency:** `describe_image(path, prompt)` signature used identically by both tools and tests; `_resolve_adb` returns path-or-None consumed in the tool. ✓
