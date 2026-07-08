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

async def test_ios_no_binary(monkeypatch):
    monkeypatch.setattr("spark_code.tools.ios_screenshot.which", lambda n: None)
    out = await IosScreenshotTool().execute()
    assert "idevicescreenshot" in out.lower()

async def test_ios_no_device_clean_message(monkeypatch):
    monkeypatch.setattr("spark_code.tools.ios_screenshot.which", lambda n: "/idevicescreenshot")
    async def fake_exec(*a, **k): return _Proc(1, b"", b"No device found")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    called = {"d": False}
    async def fake_desc(*a, **k):
        called["d"] = True
        return "x"
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)
    out = await IosScreenshotTool().execute()
    assert "device" in out.lower() and called["d"] is False


async def test_ios_network_and_udid_build_argv(monkeypatch):
    monkeypatch.setattr("spark_code.tools.ios_screenshot.which", lambda n: "/idevicescreenshot")
    seen = {}

    async def fake_exec(*a, **k):
        seen["argv"] = a
        from pathlib import Path as _P
        _P(a[-1]).write_bytes(b"PNGDATA")  # last arg is the output path
        return _Proc(0, b"", b"")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    async def fake_desc(path, prompt=None, **k):
        return "home screen with icons"
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)

    out = await IosScreenshotTool().execute(network=True, udid="ABC123")
    assert "-n" in seen["argv"] and "-u" in seen["argv"] and "ABC123" in seen["argv"]
    assert "home screen" in out


async def test_ios_usb_default_omits_network_flag(monkeypatch):
    monkeypatch.setattr("spark_code.tools.ios_screenshot.which", lambda n: "/idevicescreenshot")
    seen = {}

    async def fake_exec(*a, **k):
        seen["argv"] = a
        from pathlib import Path as _P
        _P(a[-1]).write_bytes(b"PNGDATA")
        return _Proc(0, b"", b"")
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    async def fake_desc(path, prompt=None, **k):
        return "x"
    monkeypatch.setattr("spark_code.tools.ios_screenshot.describe_image", fake_desc)

    await IosScreenshotTool().execute()
    assert "-n" not in seen["argv"] and "-u" not in seen["argv"]
