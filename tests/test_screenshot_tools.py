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

async def test_ios_captures_and_describes(monkeypatch):
    # success is judged by the output FILE (pymobiledevice3 can exit 0 + warn on
    # stderr yet still succeed), not by rc or stderr content.
    seen = {}

    async def fake_exec(*a, **k):
        seen["argv"] = a
        path = a[a.index("screenshot") + 1]
        from pathlib import Path as _P
        _P(path).write_bytes(b"\x89PNGrealbytes")
        return _Proc(0, b"", b"WARNING trying again over tunneld")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    async def fake_desc(path, prompt=None, **k):
        seen["prompt"] = prompt
        return "DoorDash Dasher home screen."
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)

    out = await IosScreenshotTool().execute(prompt="what app?")
    assert "DoorDash" in out and seen["prompt"] == "what app?"
    assert "pymobiledevice3" in seen["argv"] and "screenshot" in seen["argv"]
    assert IosScreenshotTool().is_read_only is True


async def test_ios_no_tunnel_actionable_message(monkeypatch):
    # pymobiledevice3 exits 0 but writes no bytes when the tunnel is down.
    called = {"d": False}

    async def fake_exec(*a, **k):
        return _Proc(0, b"", b"ERROR Unable to connect to Tunneld. sudo ... remote tunneld")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    async def fake_desc(*a, **k):
        called["d"] = True
        return "x"
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)

    out = await IosScreenshotTool().execute()
    assert "tunnel" in out.lower() and "tunneld" in out.lower() and called["d"] is False


async def test_ios_udid_passed_in_argv(monkeypatch):
    seen = {}

    async def fake_exec(*a, **k):
        seen["argv"] = a
        path = a[a.index("screenshot") + 1]
        from pathlib import Path as _P
        _P(path).write_bytes(b"PNG")
        return _Proc(0, b"", b"")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    async def fake_desc(path, prompt=None, **k):
        return "x"
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)

    await IosScreenshotTool().execute(udid="ABC123")
    assert "--udid" in seen["argv"] and "ABC123" in seen["argv"]
