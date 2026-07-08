from spark_code.tools.android_screenshot import AndroidScreenshotTool
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
    async def fake_desc(*a, **k):
        called["describe"] = True
        return "x"
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
        seen["path"] = path
        seen["prompt"] = prompt
        return "Home screen with a map and a Start button."
    monkeypatch.setattr("spark_code.tools.android_screenshot.describe_image", fake_desc)
    out = await AndroidScreenshotTool().execute(prompt="what's on screen?")
    assert "Start button" in out and seen["prompt"] == "what's on screen?"
    assert AndroidScreenshotTool().is_read_only is True

def _patch_tunneld(monkeypatch, devices):
    async def fake(self):
        return devices
    monkeypatch.setattr("spark_code.tools.ios_screenshot.IosScreenshotTool._tunneled_devices", fake)


async def test_ios_no_tunneld_actionable_message(monkeypatch):
    _patch_tunneld(monkeypatch, None)  # tunneld unreachable
    called = {"d": False}

    async def fake_desc(*a, **k):
        called["d"] = True
        return "x"
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)
    out = await IosScreenshotTool().execute()
    assert "tunneld" in out.lower() and called["d"] is False


async def test_ios_tunnel_up_no_device(monkeypatch):
    _patch_tunneld(monkeypatch, {})  # tunneld up, no devices
    out = await IosScreenshotTool().execute()
    assert "no device" in out.lower()


async def test_ios_captures_via_tunnel_and_describes(monkeypatch):
    _patch_tunneld(monkeypatch, {"UDID-1": [{"tunnel-address": "fda9::1", "tunnel-port": 5, "interface": "wifi"}]})
    seen = {}

    async def fake_exec(*a, **k):
        seen["argv"] = a
        path = a[-1]
        from pathlib import Path as _P
        _P(path).write_bytes(b"\x89PNGrealbytes")
        return _Proc(0, b"", b"")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    async def fake_desc(path, prompt=None, **k):
        seen["prompt"] = prompt
        return "DoorDash Dasher home screen."
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)

    out = await IosScreenshotTool().execute(prompt="what app?")
    assert "DoorDash" in out and seen["prompt"] == "what app?"
    # goes through --tunnel with the discovered UDID (this is what makes Wi-Fi work)
    assert "--tunnel" in seen["argv"] and "UDID-1" in seen["argv"]
    assert IosScreenshotTool().is_read_only is True


async def test_ios_udid_targets_specific_device(monkeypatch):
    _patch_tunneld(monkeypatch, {"UDID-1": [{}], "UDID-2": [{}]})
    seen = {}

    async def fake_exec(*a, **k):
        seen["argv"] = a
        path = a[-1]
        from pathlib import Path as _P
        _P(path).write_bytes(b"PNG")
        return _Proc(0, b"", b"")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    async def fake_desc(path, prompt=None, **k):
        return "x"
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)

    await IosScreenshotTool().execute(udid="UDID-2")
    assert "--tunnel" in seen["argv"] and "UDID-2" in seen["argv"]


async def test_ios_unknown_udid_refused(monkeypatch):
    _patch_tunneld(monkeypatch, {"UDID-1": [{}]})
    out = await IosScreenshotTool().execute(udid="NOPE")
    assert "NOPE" in out and "available" in out.lower()
