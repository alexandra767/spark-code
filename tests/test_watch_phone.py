from spark_code.tools.watch_phone import WatchPhoneTool
from spark_code.vision import DISPLAY_IMAGE_SENTINEL


async def test_watch_requires_watch_for():
    out = await WatchPhoneTool().execute(watch_for="")
    assert "required" in out.lower()


async def test_watch_detects_and_stops_early(monkeypatch, tmp_path):
    calls = {"n": 0}

    async def fake_cap(device):
        calls["n"] += 1
        p = tmp_path / f"c{calls['n']}.png"
        p.write_bytes(b"PNG")
        return str(p), None
    monkeypatch.setattr("spark_code.tools.watch_phone._capture_android", fake_cap)

    async def fake_desc(path, prompt=None, **k):
        return "YES\nan offer is showing"
    monkeypatch.setattr("spark_code.tools.watch_phone.describe_image", fake_desc)

    out = await WatchPhoneTool().execute(watch_for="an offer", max_checks=5, interval=5)
    assert "detected on check 1" in out.lower() and DISPLAY_IMAGE_SENTINEL in out
    assert calls["n"] == 1  # stopped the instant the condition was met


async def test_watch_not_detected_after_max(monkeypatch, tmp_path):
    calls = {"n": 0}

    async def fake_cap(device):
        calls["n"] += 1
        p = tmp_path / f"c{calls['n']}.png"
        p.write_bytes(b"PNG")
        return str(p), None
    monkeypatch.setattr("spark_code.tools.watch_phone._capture_android", fake_cap)

    async def fake_desc(path, prompt=None, **k):
        return "NO\nstill waiting"
    monkeypatch.setattr("spark_code.tools.watch_phone.describe_image", fake_desc)

    async def no_sleep(_s):
        return None
    monkeypatch.setattr("asyncio.sleep", no_sleep)

    out = await WatchPhoneTool().execute(watch_for="x", max_checks=3, interval=5)
    assert "not detected" in out.lower() and calls["n"] == 3


async def test_watch_capture_failure_reported(monkeypatch):
    async def fake_cap(device):
        return None, "adb not found"
    monkeypatch.setattr("spark_code.tools.watch_phone._capture_android", fake_cap)
    out = await WatchPhoneTool().execute(watch_for="x")
    assert "capture failed" in out.lower() and "adb not found" in out


async def test_watch_ios_platform_uses_ios_capture(monkeypatch, tmp_path):
    called = {"ios": False}

    async def fake_ios(udid):
        called["ios"] = True
        p = tmp_path / "i.png"
        p.write_bytes(b"PNG")
        return str(p), None
    monkeypatch.setattr("spark_code.tools.watch_phone._capture_ios", fake_ios)

    async def fake_desc(path, prompt=None, **k):
        return "YES\nhome screen"
    monkeypatch.setattr("spark_code.tools.watch_phone.describe_image", fake_desc)

    out = await WatchPhoneTool().execute(watch_for="x", platform="ios", max_checks=2)
    assert called["ios"] is True and "detected" in out.lower()
