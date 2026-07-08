from pathlib import Path

from spark_code.tools.appstore_build_upload import AppStoreBuildUploadTool


def _patch(monkeypatch, *, validate_ok=True, existing_build=False, steps=None, create_ipa=True):
    # record which pipeline steps ran, in order
    async def fake_run_step(argv, cwd=None, timeout=1800.0):
        if steps is not None:
            if "archive" in argv:
                steps.append("archive")
            elif "-exportArchive" in argv:
                steps.append("export")
                if create_ipa:
                    ep = argv[argv.index("-exportPath") + 1]
                    Path(ep).mkdir(parents=True, exist_ok=True)
                    (Path(ep) / "App.ipa").touch()
            elif "--validate-app" in argv:
                steps.append("validate")
            elif "--upload-app" in argv:
                steps.append("upload")
        return 0, "{}"

    def fake_parse(out):
        return (validate_ok, [] if validate_ok else ["bad binary"])

    def fake_props(p):
        return {"bundle_id": "com.x.App", "marketing_version": "1.2.3", "build_number": "42"}

    async def fake_resolve(self, ident):
        return "APP1"

    async def fake_get(self, path):
        return {"data": [{"id": "B1"}]} if existing_build else {"data": []}

    monkeypatch.setattr("spark_code.tools.appstore_build_upload.run_step", fake_run_step)
    monkeypatch.setattr("spark_code.tools.appstore_build_upload.parse_altool_json", fake_parse)
    monkeypatch.setattr("spark_code.tools.appstore_build_upload.read_archive_props", fake_props)
    monkeypatch.setattr("spark_code.appstore.AscClient.resolve_app", fake_resolve)
    monkeypatch.setattr("spark_code.appstore.AscClient.get", fake_get)


async def test_without_confirm_stops_before_upload(monkeypatch):
    steps = []
    _patch(monkeypatch, steps=steps)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App")
    assert "upload" not in steps  # archive/export/validate may run; upload must NOT
    assert "confirm" in out.lower() and "1.2.3" in out and "42" in out


async def test_validation_failure_stops_before_upload(monkeypatch):
    steps = []
    _patch(monkeypatch, validate_ok=False, steps=steps)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App", confirm="upload")
    assert "upload" not in steps and "bad binary" in out


async def test_collision_refuses_before_upload(monkeypatch):
    steps = []
    _patch(monkeypatch, existing_build=True, steps=steps)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App", confirm="upload")
    assert "upload" not in steps and "already exists" in out.lower()


async def test_confirmed_runs_full_pipeline_in_order(monkeypatch):
    steps = []
    _patch(monkeypatch, steps=steps)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App", confirm="upload")
    assert steps == ["archive", "export", "validate", "upload"]
    assert "appstore_status" in out


async def test_headless_refuses_without_confirm(monkeypatch):
    steps = []
    _patch(monkeypatch, steps=steps)
    out = await AppStoreBuildUploadTool(interactive=False).execute(project="/p/App.xcodeproj", scheme="App")
    assert "upload" not in steps and "confirm=" in out


async def test_export_with_no_ipa_stops_before_validate(monkeypatch):
    steps = []
    _patch(monkeypatch, steps=steps, create_ipa=False)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App", confirm="upload")
    assert steps == ["archive", "export"]  # validate/upload must NOT run
    assert "no .ipa" in out.lower()
