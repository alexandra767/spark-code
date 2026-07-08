import plistlib

import pytest

from spark_code.xcbuild import (
    XcBuildError,
    altool_argv,
    archive_argv,
    export_argv,
    parse_altool_json,
    read_archive_props,
    run_step,
    write_default_export_options,
)


def test_archive_argv_project_and_workspace():
    a = archive_argv("App.xcodeproj", "App", "/tmp/A.xcarchive")
    assert a[:3] == ["xcodebuild", "-project", "App.xcodeproj"]
    assert "-allowProvisioningUpdates" in a and a[-1] == "archive"
    assert "generic/platform=iOS" in a
    assert "-authenticationKeyPath" not in a  # signing quirk: no auth flags
    w = archive_argv("App.xcworkspace", "App", "/tmp/A.xcarchive", is_workspace=True)
    assert w[1] == "-workspace"

def test_export_argv():
    e = export_argv("/tmp/A.xcarchive", "/tmp/exp", "/tmp/opts.plist")
    assert e[:2] == ["xcodebuild", "-exportArchive"]
    assert "-allowProvisioningUpdates" in e and "-authenticationKeyPath" not in e

def test_altool_argv_validate_and_upload():
    v = altool_argv("validate", "/tmp/a.ipa", "KID", "ISS")
    assert v[:3] == ["xcrun", "altool", "--validate-app"]
    assert v[-2:] == ["--output-format", "json"]
    assert "--apiKey" in v and "KID" in v and "--apiIssuer" in v and "ISS" in v
    u = altool_argv("upload", "/tmp/a.ipa", "KID", "ISS")
    assert "--upload-app" in u

def test_write_default_export_options(tmp_path):
    p = tmp_path / "ExportOptions.plist"
    write_default_export_options(str(p), team_id="TEAM1")
    d = plistlib.loads(p.read_bytes())
    assert d["method"] == "app-store" and d["destination"] == "export"
    assert d["signingStyle"] == "automatic" and d["teamID"] == "TEAM1"

def test_parse_altool_json_success_and_errors():
    assert parse_altool_json('{"tool-version":"5"}') == (True, [])
    ok, errs = parse_altool_json('{"product-errors":[{"message":"bad binary"}]}')
    assert ok is False and errs == ["bad binary"]
    ok2, errs2 = parse_altool_json("not json")
    assert ok2 is False and errs2
    # non-dict top-level JSON must not crash — fails safe
    for blob in ("[]", "null", '"oops"'):
        okn, errsn = parse_altool_json(blob)
        assert okn is False and errsn


def test_parse_altool_json_surfaces_userinfo_reason():
    import json as _json
    blob = _json.dumps({"product-errors": [{
        "message": "The provided entity includes an attribute with a value that has already been used",
        "user-info": {
            "NSLocalizedFailureReason": "The bundle version must be higher than the previously uploaded version: '8'.",
            "detail": "The bundle version must be higher than the previously uploaded version."}}]})
    ok, errs = parse_altool_json(blob)
    assert ok is False and len(errs) == 1
    assert "bundle version must be higher" in errs[0]  # actionable reason surfaced, not just generic message

def test_parse_altool_json_generic_message_without_userinfo():
    ok, errs = parse_altool_json('{"product-errors":[{"message":"Validation failed"}]}')
    assert ok is False and errs == ["Validation failed"]

def test_read_archive_props(tmp_path):
    arch = tmp_path / "A.xcarchive"
    arch.mkdir()
    (arch / "Info.plist").write_bytes(plistlib.dumps({"ApplicationProperties": {
        "CFBundleIdentifier": "com.x.App", "CFBundleShortVersionString": "1.2.3",
        "CFBundleVersion": "42"}}))
    props = read_archive_props(str(arch))
    assert props == {"bundle_id": "com.x.App", "marketing_version": "1.2.3", "build_number": "42"}

def test_read_archive_props_missing_archive_raises_xcbuilderror(tmp_path):
    missing = tmp_path / "NoSuchArchive.xcarchive"
    with pytest.raises(XcBuildError):
        read_archive_props(str(missing))

async def test_run_step_captures_output_and_rc():
    rc, out, err = await run_step(["/bin/sh", "-c", "echo hello; echo oops >&2; exit 3"])
    assert rc == 3 and "hello" in out
    assert "oops" in err and "oops" not in out  # streams captured separately

async def test_run_step_timeout_raises():
    with pytest.raises(XcBuildError):
        await run_step(["/bin/sh", "-c", "sleep 5"], timeout=0.3)
