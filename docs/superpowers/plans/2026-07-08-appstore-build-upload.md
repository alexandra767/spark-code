# iOS Build Upload — Implementation Plan (full pipeline)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `appstore_build_upload` tool that runs archive → export (signed .ipa) → validate → gated upload for an iOS app, driving the proven `xcodebuild`/`altool` sequence.

**Architecture:** `spark_code/xcbuild.py` holds pure argv-builders + a streamed subprocess runner + `altool` JSON parsing + archive-plist reading. The tool orchestrates them, checks ASC for a build-number collision (reusing `AscClient`), gates the upload behind `confirm="upload"`, and refuses uploads in non-interactive mode without it.

**Tech Stack:** Python 3.12 (asyncio subprocess, plistlib), pytest. External: `xcodebuild`, `xcrun altool` (Xcode).

## Global Constraints

- Changes land in the live `~/spark-code`, branch `feat/appstore-build-upload`.
- Repo uses `asyncio_mode = "auto"` (async tests need NO decorator); use `monkeypatch`, not respx.
- ASC key/issuer from `~/.appstoreconnect/config.json`; `altool` reads the `.p8` from `~/.appstoreconnect/private_keys/` given `--apiKey <key_id> --apiIssuer <issuer_id>`.
- Signing quirk (do NOT deviate): archive/export use `-allowProvisioningUpdates` and **no `-authentication*` flags**.
- `--validate-app` creates NO build; only `--upload-app` does. The upload is the ONLY outward step and is gated on `confirm == "upload"` (exact) AND (interactive OR explicit confirm).
- Never edit the user's project files; a build-number collision → refuse with guidance.
- Archive/export write only into a temp dir, never the project/repo tree.
- `ExportOptions.plist`: `method=app-store`, `destination=export`, `signingStyle=automatic`, `uploadSymbols=true` (+ `teamID` if known).

---

### Task 1: `xcbuild.py` — pipeline primitives

**Files:**
- Create: `spark_code/xcbuild.py`
- Test: `tests/test_xcbuild.py`

**Interfaces produced (Task 2 consumes these exact signatures):**
- `XcBuildError(Exception)`
- `archive_argv(project, scheme, archive_path, is_workspace=False) -> list[str]`
- `export_argv(archive_path, export_path, export_options_path) -> list[str]`
- `altool_argv(action, ipa_path, key_id, issuer_id) -> list[str]` (action `"validate"`/`"upload"`)
- `write_default_export_options(path, team_id=None) -> None`
- `parse_altool_json(stdout) -> tuple[bool, list[str]]` → `(ok, error_messages)`
- `read_archive_props(archive_path) -> dict` keys `bundle_id, marketing_version, build_number`
- `async run_step(argv, cwd=None, timeout=1800.0) -> tuple[int, str]` → `(returncode, combined_output)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_xcbuild.py
import plistlib, pytest
from pathlib import Path
from spark_code.xcbuild import (
    archive_argv, export_argv, altool_argv, write_default_export_options,
    parse_altool_json, read_archive_props, run_step, XcBuildError)

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

def test_read_archive_props(tmp_path):
    arch = tmp_path / "A.xcarchive"; arch.mkdir()
    (arch / "Info.plist").write_bytes(plistlib.dumps({"ApplicationProperties": {
        "CFBundleIdentifier": "com.x.App", "CFBundleShortVersionString": "1.2.3",
        "CFBundleVersion": "42"}}))
    props = read_archive_props(str(arch))
    assert props == {"bundle_id": "com.x.App", "marketing_version": "1.2.3", "build_number": "42"}

async def test_run_step_captures_output_and_rc():
    rc, out = await run_step(["/bin/sh", "-c", "echo hello; exit 3"])
    assert rc == 3 and "hello" in out

async def test_run_step_timeout_raises():
    with pytest.raises(XcBuildError):
        await run_step(["/bin/sh", "-c", "sleep 5"], timeout=0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_xcbuild.py -v`
Expected: FAIL — `ModuleNotFoundError: spark_code.xcbuild`.

- [ ] **Step 3: Implement `xcbuild.py`**

```python
# spark_code/xcbuild.py
"""iOS build pipeline primitives (archive/export/validate/upload) for Spark Code."""
from __future__ import annotations
import asyncio
import json
import plistlib
from pathlib import Path


class XcBuildError(Exception):
    pass


def archive_argv(project, scheme, archive_path, is_workspace=False) -> list[str]:
    flag = "-workspace" if is_workspace else "-project"
    return ["xcodebuild", flag, str(project), "-scheme", scheme,
            "-configuration", "Release", "-destination", "generic/platform=iOS",
            "-archivePath", str(archive_path), "-allowProvisioningUpdates", "archive"]


def export_argv(archive_path, export_path, export_options_path) -> list[str]:
    return ["xcodebuild", "-exportArchive", "-archivePath", str(archive_path),
            "-exportPath", str(export_path), "-exportOptionsPlist", str(export_options_path),
            "-allowProvisioningUpdates"]


def altool_argv(action, ipa_path, key_id, issuer_id) -> list[str]:
    return ["xcrun", "altool", f"--{action}-app", "-f", str(ipa_path), "-t", "ios",
            "--apiKey", key_id, "--apiIssuer", issuer_id, "--output-format", "json"]


def write_default_export_options(path, team_id=None) -> None:
    opts = {"method": "app-store", "destination": "export",
            "signingStyle": "automatic", "uploadSymbols": True}
    if team_id:
        opts["teamID"] = team_id
    Path(path).write_bytes(plistlib.dumps(opts))


def parse_altool_json(stdout) -> tuple[bool, list[str]]:
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return False, ["Could not parse altool output"]
    errs = data.get("product-errors") or []
    if errs:
        return False, [e.get("message", str(e)) if isinstance(e, dict) else str(e) for e in errs]
    return True, []


def read_archive_props(archive_path) -> dict:
    p = Path(archive_path) / "Info.plist"
    data = plistlib.loads(p.read_bytes())
    props = data.get("ApplicationProperties", {})
    return {"bundle_id": props.get("CFBundleIdentifier"),
            "marketing_version": props.get("CFBundleShortVersionString"),
            "build_number": props.get("CFBundleVersion")}


async def run_step(argv, cwd=None, timeout=1800.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise XcBuildError(f"step timed out after {timeout}s: {' '.join(map(str, argv[:3]))} …")
    return proc.returncode, out.decode("utf-8", "replace")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_xcbuild.py -v` (Expected: PASS) then `.venv/bin/python -m ruff check spark_code/xcbuild.py tests/test_xcbuild.py`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/xcbuild.py tests/test_xcbuild.py
git commit -m "feat: xcbuild pipeline primitives (archive/export/altool argv, plist, run_step)"
```

---

### Task 2: `appstore_build_upload` tool + registration

**Files:**
- Create: `spark_code/tools/appstore_build_upload.py`
- Modify: `spark_code/cli.py` (register in `build_tools`, passing the existing `interactive` param)
- Test: `tests/test_appstore_build_upload.py`

**Interfaces:**
- Consumes: `spark_code.xcbuild.*` (Task 1); `spark_code.appstore.AscClient` (`resolve_app`, `get`); `Tool` base.
- Produces: `AppStoreBuildUploadTool(interactive: bool = True)` — `name="appstore_build_upload"`, `is_read_only=False`, `requires_permission=True`, params `project`, `scheme`, `confirm`.

- [ ] **Step 1: Write the failing tests (gate + collision + pipeline order)**

```python
# tests/test_appstore_build_upload.py
import pytest
from spark_code.tools.appstore_build_upload import AppStoreBuildUploadTool

def _patch(monkeypatch, *, validate_ok=True, existing_build=False, steps=None):
    # record which pipeline steps ran, in order
    async def fake_run_step(argv, cwd=None, timeout=1800.0):
        if steps is not None:
            if "archive" in argv: steps.append("archive")
            elif "-exportArchive" in argv: steps.append("export")
            elif "--validate-app" in argv: steps.append("validate")
            elif "--upload-app" in argv: steps.append("upload")
        return 0, "{}"
    def fake_parse(out): return (validate_ok, [] if validate_ok else ["bad binary"])
    def fake_props(p): return {"bundle_id": "com.x.App", "marketing_version": "1.2.3", "build_number": "42"}
    async def fake_resolve(self, ident): return "APP1"
    async def fake_get(self, path):
        return {"data": [{"id": "B1"}]} if existing_build else {"data": []}
    monkeypatch.setattr("spark_code.tools.appstore_build_upload.run_step", fake_run_step)
    monkeypatch.setattr("spark_code.tools.appstore_build_upload.parse_altool_json", fake_parse)
    monkeypatch.setattr("spark_code.tools.appstore_build_upload.read_archive_props", fake_props)
    monkeypatch.setattr("spark_code.appstore.AscClient.resolve_app", fake_resolve)
    monkeypatch.setattr("spark_code.appstore.AscClient.get", fake_get)

async def test_without_confirm_stops_before_upload(monkeypatch):
    steps = []; _patch(monkeypatch, steps=steps)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App")
    assert "upload" not in steps  # archive/export/validate may run; upload must NOT
    assert "confirm" in out.lower() and "1.2.3" in out and "42" in out

async def test_validation_failure_stops_before_upload(monkeypatch):
    steps = []; _patch(monkeypatch, validate_ok=False, steps=steps)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App", confirm="upload")
    assert "upload" not in steps and "bad binary" in out

async def test_collision_refuses_before_upload(monkeypatch):
    steps = []; _patch(monkeypatch, existing_build=True, steps=steps)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App", confirm="upload")
    assert "upload" not in steps and "already exists" in out.lower()

async def test_confirmed_runs_full_pipeline_in_order(monkeypatch):
    steps = []; _patch(monkeypatch, steps=steps)
    out = await AppStoreBuildUploadTool().execute(project="/p/App.xcodeproj", scheme="App", confirm="upload")
    assert steps == ["archive", "export", "validate", "upload"]
    assert "appstore_status" in out

async def test_headless_refuses_without_confirm(monkeypatch):
    steps = []; _patch(monkeypatch, steps=steps)
    out = await AppStoreBuildUploadTool(interactive=False).execute(project="/p/App.xcodeproj", scheme="App")
    assert "upload" not in steps and "confirm=" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore_build_upload.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the tool**

```python
# spark_code/tools/appstore_build_upload.py
from __future__ import annotations
import tempfile
from pathlib import Path
from .base import Tool
from ..appstore import AscClient, AscError
from ..xcbuild import (archive_argv, export_argv, altool_argv, write_default_export_options,
                       parse_altool_json, read_archive_props, run_step, XcBuildError)


class AppStoreBuildUploadTool(Tool):
    def __init__(self, interactive: bool = True):
        self._interactive = interactive

    @property
    def name(self) -> str: return "appstore_build_upload"

    @property
    def description(self) -> str:
        return ("Archive, export a signed .ipa, validate it against Apple, and (only with "
                "confirm='upload') upload the build to App Store Connect. Input: project "
                "(.xcodeproj/.xcworkspace path), scheme, confirm. Archive/export/validate create "
                "no build; only upload does.")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "project": {"type": "string", "description": ".xcodeproj or .xcworkspace path"},
            "scheme": {"type": "string", "description": "Xcode scheme to archive"},
            "confirm": {"type": "string", "description": "Pass exactly 'upload' to upload; omit to preview"}},
            "required": ["project", "scheme"]}

    @property
    def is_read_only(self) -> bool: return False

    @property
    def requires_permission(self) -> bool: return True

    async def execute(self, project: str = "", scheme: str = "", confirm: str = "", **kwargs) -> str:
        if not project or not scheme:
            return "Error: 'project' and 'scheme' are required."
        is_ws = str(project).endswith(".xcworkspace")
        try:
            with tempfile.TemporaryDirectory(prefix="spark-xcbuild-") as td:
                arch = str(Path(td) / "App.xcarchive")
                exp = str(Path(td) / "export")
                opts = str(Path(td) / "ExportOptions.plist")
                # 1. archive
                rc, out = await run_step(archive_argv(project, scheme, arch, is_ws))
                if rc != 0:
                    return f"Archive failed (rc {rc}). Last output:\n{out[-1500:]}"
                # 2. export
                write_default_export_options(opts)
                rc, out = await run_step(export_argv(arch, exp, opts))
                if rc != 0:
                    return f"Export failed (rc {rc}). Last output:\n{out[-1500:]}"
                ipas = list(Path(exp).glob("*.ipa"))
                if not ipas:
                    return "Export produced no .ipa."
                ipa = str(ipas[0])
                props = read_archive_props(arch)
                # 3. validate (creates no build)
                c = AscClient()
                key_id = c._config().get("key_id"); issuer_id = c._config().get("issuer_id")
                rc, out = await run_step(altool_argv("validate", ipa, key_id, issuer_id))
                ok, errs = parse_altool_json(out)
                if rc != 0 or not ok:
                    return "Validation failed:\n" + "\n".join(f"  - {e}" for e in (errs or [f"rc {rc}"]))
                # 4. collision check
                app_id = await c.resolve_app(props["bundle_id"])
                q = (f"/v1/builds?filter[app]={app_id}"
                     f"&filter[preReleaseVersion.version]={props['marketing_version']}"
                     f"&filter[version]={props['build_number']}")
                existing = (await c.get(q)).get("data", [])
                if existing:
                    return (f"Build {props['build_number']} for {props['marketing_version']} already "
                            f"exists in App Store Connect — bump CURRENT_PROJECT_VERSION and re-run. "
                            f"(Not uploading; your project files were not modified.)")
                summary = (f"Validated ✓ — ready to upload:\n"
                           f"  App: {props['bundle_id']}\n"
                           f"  Version {props['marketing_version']} (build {props['build_number']})\n"
                           f"  IPA: {ipa}")
                # 5. gate
                if confirm != "upload":
                    if not self._interactive:
                        return summary + "\n\nHeadless: refusing. Re-invoke with confirm=\"upload\" to upload."
                    return summary + "\n\nThis uploads a real build to Apple. Re-invoke with confirm=\"upload\"."
                # 6. upload (the one outward step)
                rc, out = await run_step(altool_argv("upload", ipa, key_id, issuer_id))
                ok, errs = parse_altool_json(out)
                if rc != 0 or not ok:
                    return "Upload failed:\n" + "\n".join(f"  - {e}" for e in (errs or [f"rc {rc}"]))
                return (f"✅ Uploaded {props['bundle_id']} {props['marketing_version']} "
                        f"(build {props['build_number']}). It will take a few minutes to process — "
                        f"run appstore_status to watch it reach VALID, then appstore_submit when ready.")
        except (AscError, XcBuildError) as e:
            return f"Build upload error: {e}"
```
Register in `cli.py` `build_tools` (next to `AppStoreSubmitTool`):
```python
    from .tools.appstore_build_upload import AppStoreBuildUploadTool
    registry.register(AppStoreBuildUploadTool(interactive=interactive))
```

- [ ] **Step 4: Run tests**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore_build_upload.py -v` (Expected: PASS, 5 tests) then `.venv/bin/python -m pytest tests/ -q | tail -2` and `.venv/bin/python -m ruff check spark_code/tools/appstore_build_upload.py spark_code/cli.py tests/test_appstore_build_upload.py`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/tools/appstore_build_upload.py spark_code/cli.py tests/test_appstore_build_upload.py
git commit -m "feat: appstore_build_upload tool — archive/export/validate + gated upload + collision refuse"
```

---

## Live acceptance (controller-run, after Task 2 — NOT a subagent task)

Against a real app project, run the pipeline **up to validate only** (patch/omit the upload) or invoke with NO confirm so it stops at the gate — proving archive → export → validate works end-to-end without uploading. Suggested:
```bash
cd ~/spark-code && .venv/bin/python -c "
import asyncio; from spark_code.tools.appstore_build_upload import AppStoreBuildUploadTool
print(asyncio.run(AppStoreBuildUploadTool().execute(project='<real .xcodeproj>', scheme='<scheme>')))"
```
Expected: archive + export + validate run, then a "Validated ✓ — ready to upload … re-invoke with confirm" summary — and NO build appears in App Store Connect. If archive fails on code-signing, surface it as an environment/signing setup issue (Xcode account / provisioning), not a code defect. **Do not pass `confirm="upload"` during acceptance.**

## Self-Review

- **Spec coverage:** xcbuild primitives + run_step (T1); tool orchestration, ExportOptions, validate-no-build, collision-refuse, confirm gate + headless refuse, upload-last (T2); temp-dir isolation (T2 `TemporaryDirectory`); live acceptance stops at validate. ✓
- **Placeholders:** none — full code per step. ✓
- **Type consistency:** `read_archive_props` keys (`bundle_id/marketing_version/build_number`) produced in T1, consumed in T2; `run_step`→`(rc, out)` and `parse_altool_json`→`(ok, errs)` used consistently; pipeline-order test asserts `["archive","export","validate","upload"]`. ✓
