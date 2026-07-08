from spark_code.tools.android_control import AndroidControlTool, _elements, _find

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
