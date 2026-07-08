from spark_code.tools.ios_launch import IosLaunchTool, _pick_device, _resolve_bundle

APPS = [
    {"name": "Mobile Cooling", "bundleIdentifier": "com.dometic.cfx3", "hidden": False, "internalApp": False},
    {"name": "GigLedger", "bundleIdentifier": "com.alexandratitus.GigLedger", "hidden": False, "internalApp": False},
    {"name": "Calendar", "bundleIdentifier": "com.apple.mobilecal", "hidden": False, "internalApp": True},
]

DEVICES = [
    {"identifier": "WATCH", "hardwareProperties": {"deviceType": "appleWatch"},
     "connectionProperties": {"tunnelState": "disconnected"}, "deviceProperties": {"name": "Watch"}},
    {"identifier": "IPAD", "hardwareProperties": {"deviceType": "iPad"},
     "connectionProperties": {"tunnelState": "unavailable"}, "deviceProperties": {"name": "iPad"}},
    {"identifier": "IPHONE", "hardwareProperties": {"deviceType": "iPhone"},
     "connectionProperties": {"tunnelState": "disconnected"}, "deviceProperties": {"name": "iPhone"}},
]


def test_resolve_bundle_matches_name_and_bundle_id():
    # 'dometic' is not in the display name — only the bundle id — must still match
    assert _resolve_bundle(APPS, "dometic")[0]["bundleIdentifier"] == "com.dometic.cfx3"
    assert _resolve_bundle(APPS, "gigledger")[0]["name"] == "GigLedger"
    assert _resolve_bundle(APPS, "com.apple.mobilecal")[0]["name"] == "Calendar"
    assert _resolve_bundle(APPS, "spotify") == (None, [])


def test_pick_device_prefers_reachable_iphone():
    dev, udid = _pick_device(DEVICES)
    assert udid == "IPHONE"  # iPhone over iPad(unavailable)/Watch
    # explicit udid honored
    assert _pick_device(DEVICES, "IPAD")[1] == "IPAD"
    assert _pick_device(DEVICES, "NOPE") == (None, "")


def _patch(monkeypatch, launched, *, devices=DEVICES, apps=APPS, launch_ok=True):
    async def fake_json(args):
        if "devices" in args:
            return {"result": {"devices": devices}}, None
        return {"result": {"apps": apps}}, None

    async def fake_run(argv):
        if "launch" in argv:
            launched.append(argv)
            if launch_ok:
                return 0, "Launched application with %s bundle identifier." % argv[-1], ""
            return 1, "", "error: application not found"
        return 0, "", ""
    monkeypatch.setattr("spark_code.tools.ios_launch._devicectl_json", fake_json)
    monkeypatch.setattr("spark_code.tools.ios_launch._run", fake_run)
    monkeypatch.setattr("spark_code.tools.ios_launch.which", lambda _x: "/usr/bin/xcrun")


async def test_launch_opens_by_name(monkeypatch):
    launched = []
    _patch(monkeypatch, launched)
    out = await IosLaunchTool().execute(app="dometic")
    assert "opened" in out.lower() and "com.dometic.cfx3" in out
    assert launched and launched[0][-1] == "com.dometic.cfx3" and "IPHONE" in launched[0]


async def test_launch_unknown_app_launches_nothing(monkeypatch):
    launched = []
    _patch(monkeypatch, launched)
    out = await IosLaunchTool().execute(app="spotify")
    assert "no app matching" in out.lower()
    assert launched == []


async def test_launch_failure_is_reported(monkeypatch):
    launched = []
    _patch(monkeypatch, launched, launch_ok=False)
    out = await IosLaunchTool().execute(app="dometic")
    assert "couldn't launch" in out.lower()


async def test_no_device_reported(monkeypatch):
    launched = []
    _patch(monkeypatch, launched, devices=[])
    out = await IosLaunchTool().execute(app="dometic")
    assert "no connected iphone" in out.lower()
    assert launched == []


async def test_missing_xcrun(monkeypatch):
    monkeypatch.setattr("spark_code.tools.ios_launch.which", lambda _x: None)
    out = await IosLaunchTool().execute(app="dometic")
    assert "xcrun" in out.lower()


async def test_flags():
    t = IosLaunchTool()
    assert t.is_read_only is False and t.requires_permission is True
