from spark_code.tools.android_control import AndroidControlTool, _elements, _find, _resolve_package

INSTALLED = ["com.android.settings", "com.gigledger.android.debug",
             "com.boonpoint.android.debug", "com.sec.android.app.launcher"]

SAMPLE_XML = (
    '<?xml version="1.0"?><hierarchy>'
    '<node text="Accept" content-desc="" bounds="[100,200][300,260]" clickable="true"/>'
    '<node text="Decline" content-desc="" bounds="[100,300][300,360]" clickable="true"/>'
    '</hierarchy>')


def _patch(monkeypatch, cmds):
    async def fake_run(argv):
        cmds.append(argv)
        if "uiautomator" in argv:
            return 0, "", ""
        if "cat" in argv:
            return 0, SAMPLE_XML, ""
        return 0, "", ""
    monkeypatch.setattr("spark_code.tools.android_control._run", fake_run)
    monkeypatch.setattr("spark_code.tools.android_control._resolve_adb", lambda: "/adb")


def _issued_input(cmds):
    return [a for a in cmds if "input" in a]


def _patch_launch(monkeypatch, cmds, monkey_out="Events injected: 1"):
    async def fake_run(argv):
        cmds.append(argv)
        if "packages" in argv:
            return 0, "".join(f"package:{p}\n" for p in INSTALLED), ""
        if "monkey" in argv:
            return 0, monkey_out, ""
        return 0, "", ""
    monkeypatch.setattr("spark_code.tools.android_control._run", fake_run)
    monkeypatch.setattr("spark_code.tools.android_control._resolve_adb", lambda: "/adb")


def _monkey_cmds(cmds):
    return [a for a in cmds if "monkey" in a]


def test_resolve_package_fuzzy_and_debug_preference():
    # segment match + only-debug-installed → picks the debug build
    assert _resolve_package(INSTALLED, "gigledger")[0] == "com.gigledger.android.debug"
    assert _resolve_package(INSTALLED, "boonpoint")[0] == "com.boonpoint.android.debug"
    # exact package id wins outright
    assert _resolve_package(INSTALLED, "com.android.settings")[0] == "com.android.settings"
    # prefer .debug when both a base and its .debug variant match
    both = ["com.gigledger.android", "com.gigledger.android.debug"]
    assert _resolve_package(both, "gigledger")[0] == "com.gigledger.android.debug"
    # no match
    assert _resolve_package(INSTALLED, "spotify") == (None, [])


async def test_launch_opens_immediately_without_confirm(monkeypatch):
    cmds = []
    _patch_launch(monkeypatch, cmds)
    out = await AndroidControlTool().execute(action="launch", app="gigledger")
    assert "opened" in out.lower() and "com.gigledger.android.debug" in out
    assert "confirm" not in out.lower()  # launch does NOT require the gate
    mk = _monkey_cmds(cmds)
    assert mk and "com.gigledger.android.debug" in mk[0] and "LAUNCHER" in " ".join(mk[0])


async def test_launch_unknown_app_launches_nothing(monkeypatch):
    cmds = []
    _patch_launch(monkeypatch, cmds)
    out = await AndroidControlTool().execute(action="launch", app="spotify")
    assert "no installed app matching" in out.lower()
    assert _monkey_cmds(cmds) == []  # nothing launched


async def test_launch_no_launchable_activity_reported(monkeypatch):
    cmds = []
    _patch_launch(monkeypatch, cmds, monkey_out="** No activities found to run, monkey aborted.")
    out = await AndroidControlTool().execute(action="launch", app="settings")
    assert "couldn't launch" in out.lower()


def test_elements_and_find_compute_center():
    els = _elements(SAMPLE_XML)
    acc = next(e for e in els if e["label"] == "Accept")
    assert acc["cx"] == 200 and acc["cy"] == 230 and acc["clickable"] is True
    assert _find(els, "accept")["label"] == "Accept"


async def test_find_is_read_only(monkeypatch):
    cmds = []
    _patch(monkeypatch, cmds)
    out = await AndroidControlTool().execute(action="find")
    assert "Accept" in out and "Decline" in out
    assert _issued_input(cmds) == []  # never acts


async def test_tap_without_confirm_previews_and_does_nothing(monkeypatch):
    cmds = []
    _patch(monkeypatch, cmds)
    out = await AndroidControlTool().execute(action="tap", target="Accept")
    assert "confirm" in out.lower() and "nothing done" in out.lower()
    assert _issued_input(cmds) == []  # THE gate: no input command reached the phone
    assert any("uiautomator" in a for a in cmds)  # it did resolve the element (read-only)


async def test_tap_with_confirm_executes_input_tap(monkeypatch):
    cmds = []
    _patch(monkeypatch, cmds)
    out = await AndroidControlTool().execute(action="tap", target="Accept", confirm="yes")
    assert "done" in out.lower()
    tap = _issued_input(cmds)
    assert tap and "tap" in tap[0] and "200" in tap[0] and "230" in tap[0]


async def test_type_gated(monkeypatch):
    cmds = []
    _patch(monkeypatch, cmds)
    assert "confirm" in (await AndroidControlTool().execute(action="type", text="hi")).lower()
    assert _issued_input(cmds) == []
    await AndroidControlTool().execute(action="type", text="hi", confirm="yes")
    assert any("text" in a and "hi" in a for a in _issued_input(cmds))


async def test_swipe_gated(monkeypatch):
    cmds = []
    _patch(monkeypatch, cmds)
    assert "confirm" in (await AndroidControlTool().execute(action="swipe", direction="up")).lower()
    assert _issued_input(cmds) == []  # the gate: no swipe reached the phone
    await AndroidControlTool().execute(action="swipe", direction="up", confirm="yes")
    assert any("swipe" in a for a in _issued_input(cmds))


async def test_type_multiword_is_escaped_for_device_shell(monkeypatch):
    cmds = []
    _patch(monkeypatch, cmds)
    await AndroidControlTool().execute(action="type", text="hello world", confirm="yes")
    typed = _issued_input(cmds)[0]
    # the device shell re-parses argv, so spaces must be %s or only "hello" types
    assert "hello%sworld" in typed and "hello world" not in typed


async def test_key_gated(monkeypatch):
    cmds = []
    _patch(monkeypatch, cmds)
    assert "confirm" in (await AndroidControlTool().execute(action="key", key="back")).lower()
    assert _issued_input(cmds) == []
    await AndroidControlTool().execute(action="key", key="back", confirm="yes")
    assert any("keyevent" in a and "KEYCODE_BACK" in a for a in _issued_input(cmds))


async def test_tap_unknown_target_never_acts(monkeypatch):
    cmds = []
    _patch(monkeypatch, cmds)
    out = await AndroidControlTool().execute(action="tap", target="Nope", confirm="yes")
    assert "no on-screen element" in out.lower()
    assert _issued_input(cmds) == []


async def test_no_adb(monkeypatch):
    monkeypatch.setattr("spark_code.tools.android_control._resolve_adb", lambda: None)
    assert "adb" in (await AndroidControlTool().execute(action="find")).lower()


async def test_flags_are_write_and_gated():
    t = AndroidControlTool()
    assert t.is_read_only is False and t.requires_permission is True
