# Android/Play Publish — Implementation Plan (build → upload → testing track)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `play_publish` tool that builds a signed AAB, uploads it to a Google Play edit, assigns it to a **testing** track (internal/alpha/beta), validates, and — only with `confirm="publish"` — commits (releasing to that testing track). Production is refused.

**Architecture:** Add publish methods to `spark_code/play.py` (create/upload/assign/validate/commit/delete edit). The tool runs `gradle bundleRelease` (via the existing `xcbuild.run_step`), then drives the edit; any pre-commit failure deletes the edit; commit is gated.

**Tech Stack:** Python 3.12 (async httpx, PyJWT), pytest. No new deps. External: `gradlew`.

## Global Constraints

- Changes land in the live `~/spark-code`, branch `feat/play-publish`.
- `asyncio_mode = "auto"` (async tests no decorator); monkeypatch, not respx.
- `track` MUST be in `TESTING_TRACKS = {"internal","alpha","beta"}` — production/other refused before any build or API call.
- `commit_edit` is reachable ONLY with `confirm == "publish"` (exact). Non-interactive without confirm → refuse. Any failure before commit → `delete_edit` (nothing published).
- AAB upload posts to the `/upload/androidpublisher/v3/...?uploadType=media` URL with `Content-Type: application/octet-stream`; long timeout.
- Never log/echo the token. `play_publish`: `is_read_only=False`, `requires_permission=True`.

---

### Task 1: `play.py` publish methods

**Files:**
- Modify: `spark_code/play.py`
- Test: `tests/test_play_publish_client.py`

**Interfaces produced (Task 2 consumes):**
- `TESTING_TRACKS: frozenset[str]`
- `PlayClient` async methods: `create_edit(package)->str`, `delete_edit(package, edit_id)`, `upload_bundle(package, edit_id, aab_path)->int`, `assign_track(package, edit_id, track, version_code)`, `validate_edit(package, edit_id)->dict`, `commit_edit(package, edit_id)->dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_play_publish_client.py
import json, pytest
from pathlib import Path
from spark_code.play import PlayClient, PlayError, TESTING_TRACKS

def _sa_file(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    p = tmp_path / "sa.json"
    p.write_text(json.dumps({"client_email": "b@p.iam.gserviceaccount.com",
        "private_key": pem, "token_uri": "https://oauth2.googleapis.com/token"}))
    return str(p)

def test_testing_tracks_excludes_production():
    assert "production" not in TESTING_TRACKS
    assert {"internal", "alpha", "beta"} <= TESTING_TRACKS

async def test_edit_json_methods_hit_right_paths(tmp_path, monkeypatch):
    c = PlayClient(sa_path=_sa_file(tmp_path))
    calls = []
    async def fake_req(method, path, body=None):
        calls.append((method, path, body))
        if path.endswith("/edits"): return {"id": "E9"}
        return {"ok": True}
    monkeypatch.setattr(c, "_req", fake_req)
    assert await c.create_edit("com.x") == "E9"
    await c.assign_track("com.x", "E9", "internal", 46)
    await c.validate_edit("com.x", "E9")
    await c.commit_edit("com.x", "E9")
    await c.delete_edit("com.x", "E9")
    paths = [(m, p) for m, p, _ in calls]
    assert ("POST", "/applications/com.x/edits") in paths
    assert ("PUT", "/applications/com.x/edits/E9/tracks/internal") in paths
    assert ("POST", "/applications/com.x/edits/E9:validate") in paths
    assert ("POST", "/applications/com.x/edits/E9:commit") in paths
    assert ("DELETE", "/applications/com.x/edits/E9") in paths
    # the assign_track release body targets the chosen track, status completed
    body = next(b for m, p, b in calls if p.endswith("/tracks/internal"))
    assert body["releases"][0]["versionCodes"] == ["46"]
    assert body["releases"][0]["status"] == "completed"

async def test_upload_bundle_posts_aab_bytes_and_returns_versioncode(tmp_path, monkeypatch):
    c = PlayClient(sa_path=_sa_file(tmp_path))
    aab = tmp_path / "app.aab"; aab.write_bytes(b"AABDATA")
    async def fake_token(): return "TOK"
    monkeypatch.setattr(c, "_token", fake_token)
    seen = {}
    class FakeResp:
        status_code = 200
        text = '{"versionCode": 46}'
        def json(self): return {"versionCode": 46}
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None, content=None):
            seen["url"] = url; seen["headers"] = headers; seen["content"] = content
            return FakeResp()
    monkeypatch.setattr("spark_code.play.httpx.AsyncClient", FakeClient)
    vc = await c.upload_bundle("com.x", "E9", str(aab))
    assert vc == 46
    assert "uploadType=media" in seen["url"] and "/upload/androidpublisher/" in seen["url"]
    assert seen["headers"]["Content-Type"] == "application/octet-stream"
    assert seen["content"] == b"AABDATA"

async def test_upload_bundle_raises_on_error(tmp_path, monkeypatch):
    c = PlayClient(sa_path=_sa_file(tmp_path))
    aab = tmp_path / "app.aab"; aab.write_bytes(b"X")
    async def fake_token(): return "TOK"
    monkeypatch.setattr(c, "_token", fake_token)
    class FakeResp:
        status_code = 403; text = '{"error":"duplicate versionCode"}'
        def json(self): return {}
    class FakeClient:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, headers=None, content=None): return FakeResp()
    monkeypatch.setattr("spark_code.play.httpx.AsyncClient", FakeClient)
    with pytest.raises(PlayError):
        await c.upload_bundle("com.x", "E9", str(aab))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_play_publish_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'TESTING_TRACKS'` (and missing methods).

- [ ] **Step 3: Implement (append to `spark_code/play.py`)**

Add near the top (module level):
```python
TESTING_TRACKS = frozenset({"internal", "alpha", "beta"})
```
Add methods to `PlayClient`:
```python
    async def create_edit(self, package: str) -> str:
        return (await self.post(f"/applications/{package}/edits", {}))["id"]

    async def delete_edit(self, package: str, edit_id: str) -> None:
        await self.delete(f"/applications/{package}/edits/{edit_id}")

    async def upload_bundle(self, package: str, edit_id: str, aab_path: str) -> int:
        tok = await self._token()
        url = ("https://androidpublisher.googleapis.com/upload/androidpublisher/v3/"
               f"applications/{package}/edits/{edit_id}/bundles?uploadType=media")
        data = Path(aab_path).read_bytes()
        async with httpx.AsyncClient(timeout=600.0) as c:
            r = await c.post(url, headers={"Authorization": f"Bearer {tok}",
                                           "Content-Type": "application/octet-stream"}, content=data)
        if r.status_code >= 400:
            raise PlayError(f"Play bundle upload → {r.status_code}: {r.text[:300]}")
        return int(r.json()["versionCode"])

    async def assign_track(self, package: str, edit_id: str, track: str, version_code: int) -> None:
        body = {"track": track,
                "releases": [{"versionCodes": [str(version_code)], "status": "completed"}]}
        await self._req("PUT", f"/applications/{package}/edits/{edit_id}/tracks/{track}", body)

    async def validate_edit(self, package: str, edit_id: str) -> dict:
        return await self.post(f"/applications/{package}/edits/{edit_id}:validate", {})

    async def commit_edit(self, package: str, edit_id: str) -> dict:
        return await self.post(f"/applications/{package}/edits/{edit_id}:commit", {})
```

- [ ] **Step 4: Run tests + ruff**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_play_publish_client.py -v` (Expected: PASS, 5) then `.venv/bin/python -m ruff check spark_code/play.py tests/test_play_publish_client.py`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/play.py tests/test_play_publish_client.py
git commit -m "feat: Play publish methods (create/upload/assign-track/validate/commit edit)"
```

---

### Task 2: `play_publish` tool + registration

**Files:**
- Create: `spark_code/tools/play_publish.py`
- Modify: `spark_code/cli.py` (register in `build_tools` with `interactive`, near `AppStoreBuildUploadTool`)
- Test: `tests/test_play_publish_tool.py`

**Interfaces:**
- Consumes: `spark_code.play` (`PlayClient`, `PlayError`, `TESTING_TRACKS`), `spark_code.xcbuild.run_step`, `Tool` base.
- Produces: `PlayPublishTool(interactive: bool = True)` — `name="play_publish"`, `is_read_only=False`, `requires_permission=True`, params `project`, `package`, `track`, `confirm`, `module` (default `:androidApp`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_play_publish_tool.py
import pytest
from pathlib import Path
from spark_code.tools.play_publish import PlayPublishTool

def _patch(monkeypatch, tmp_path, *, upload_ok=True, steps=None):
    # a real .aab so glob finds it
    out = tmp_path / "androidApp" / "build" / "outputs" / "bundle" / "release"
    out.mkdir(parents=True, exist_ok=True)
    (out / "androidApp-release.aab").write_bytes(b"AAB")
    async def fake_run_step(argv, cwd=None, timeout=1800.0):
        if steps is not None: steps.append("gradle")
        return 0, "BUILD SUCCESSFUL", ""
    monkeypatch.setattr("spark_code.tools.play_publish.run_step", fake_run_step)
    async def fake_create(self, pkg):
        if steps is not None: steps.append("create"); 
        return "E1"
    async def fake_upload(self, pkg, eid, aab):
        if steps is not None: steps.append("upload")
        if not upload_ok: from spark_code.play import PlayError; raise PlayError("dup vc")
        return 46
    async def fake_assign(self, pkg, eid, track, vc):
        if steps is not None: steps.append("assign")
    async def fake_validate(self, pkg, eid):
        if steps is not None: steps.append("validate"); 
        return {}
    async def fake_commit(self, pkg, eid):
        if steps is not None: steps.append("commit"); 
        return {}
    async def fake_delete(self, pkg, eid):
        if steps is not None: steps.append("delete")
    monkeypatch.setattr("spark_code.play.PlayClient.create_edit", fake_create)
    monkeypatch.setattr("spark_code.play.PlayClient.upload_bundle", fake_upload)
    monkeypatch.setattr("spark_code.play.PlayClient.assign_track", fake_assign)
    monkeypatch.setattr("spark_code.play.PlayClient.validate_edit", fake_validate)
    monkeypatch.setattr("spark_code.play.PlayClient.commit_edit", fake_commit)
    monkeypatch.setattr("spark_code.play.PlayClient.delete_edit", fake_delete)

async def test_production_refused_before_anything(monkeypatch, tmp_path):
    steps = []; _patch(monkeypatch, tmp_path, steps=steps)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="production", confirm="publish")
    assert steps == [] and "production" in out.lower()

async def test_without_confirm_stops_and_deletes_edit(monkeypatch, tmp_path):
    steps = []; _patch(monkeypatch, tmp_path, steps=steps)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="internal")
    assert "commit" not in steps and "delete" in steps
    assert "confirm" in out.lower() and "46" in out and "internal" in out

async def test_confirmed_runs_full_sequence(monkeypatch, tmp_path):
    steps = []; _patch(monkeypatch, tmp_path, steps=steps)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="internal", confirm="publish")
    assert steps == ["gradle","create","upload","assign","validate","commit"]
    assert "released" in out.lower() and "internal" in out

async def test_upload_failure_deletes_edit_no_commit(monkeypatch, tmp_path):
    steps = []; _patch(monkeypatch, tmp_path, upload_ok=False, steps=steps)
    out = await PlayPublishTool().execute(project=str(tmp_path), package="com.x", track="internal", confirm="publish")
    assert "commit" not in steps and "delete" in steps and "dup vc" in out

async def test_headless_refuses_without_confirm(monkeypatch, tmp_path):
    steps = []; _patch(monkeypatch, tmp_path, steps=steps)
    out = await PlayPublishTool(interactive=False).execute(project=str(tmp_path), package="com.x", track="internal")
    assert "commit" not in steps and "confirm=" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_play_publish_tool.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the tool**

```python
# spark_code/tools/play_publish.py
from __future__ import annotations
from pathlib import Path
from .base import Tool
from ..play import PlayClient, PlayError, TESTING_TRACKS
from ..xcbuild import run_step, XcBuildError


class PlayPublishTool(Tool):
    def __init__(self, interactive: bool = True):
        self._interactive = interactive

    @property
    def name(self) -> str: return "play_publish"

    @property
    def description(self) -> str:
        return ("Build a signed Android AAB (gradle bundleRelease), upload it to Google Play, "
                "and (only with confirm='publish') release it to a TESTING track "
                "(internal/alpha/beta — production is not allowed). Input: project (gradle root), "
                "package, track, confirm, optional module (default :androidApp).")

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "project": {"type": "string", "description": "Gradle project root (has ./gradlew)"},
            "package": {"type": "string", "description": "Android package, e.g. com.gigledger.android"},
            "track": {"type": "string", "description": "Testing track: internal | alpha | beta"},
            "module": {"type": "string", "description": "Gradle module (default :androidApp)"},
            "confirm": {"type": "string", "description": "Pass 'publish' to release; omit to preview"}},
            "required": ["project", "package", "track"]}

    @property
    def is_read_only(self) -> bool: return False

    @property
    def requires_permission(self) -> bool: return True

    async def execute(self, project: str = "", package: str = "", track: str = "",
                      module: str = ":androidApp", confirm: str = "", **kwargs) -> str:
        if not project or not package or not track:
            return "Error: 'project', 'package', and 'track' are required."
        if track not in TESTING_TRACKS:
            return (f"Refusing: track '{track}' is not allowed. This tool only releases to "
                    f"testing tracks ({', '.join(sorted(TESTING_TRACKS))}) — production is out of scope.")
        # 1. build the signed AAB
        try:
            rc, out, err = await run_step(["./gradlew", f"{module}:bundleRelease"], cwd=project)
        except XcBuildError as e:
            return f"Gradle build error: {e}"
        if rc != 0:
            return f"Gradle bundleRelease failed (rc {rc}). Last output:\n{(out + err)[-1500:]}"
        mod_dir = module.lstrip(":").replace(":", "/")
        aabs = list((Path(project) / mod_dir / "build" / "outputs" / "bundle" / "release").glob("*.aab"))
        if not aabs:
            return "Build succeeded but no .aab was found under build/outputs/bundle/release/."
        aab = str(aabs[0])
        # 2. drive the edit; any pre-commit failure deletes it (nothing published)
        c = PlayClient()
        try:
            edit_id = await c.create_edit(package)
        except PlayError as e:
            return f"Google Play error: {e}"
        try:
            vc = await c.upload_bundle(package, edit_id, aab)
            await c.assign_track(package, edit_id, track, vc)
            await c.validate_edit(package, edit_id)
            summary = (f"Built + validated ✓ — ready to release:\n"
                       f"  {package}  version code {vc}  →  track {track}")
            if confirm != "publish":
                await c.delete_edit(package, edit_id)  # nothing committed
                tail = ("\n\nHeadless: refusing. Re-invoke with confirm=\"publish\" to release."
                        if not self._interactive
                        else f"\n\nThis RELEASES to your {track} testers. Re-invoke with confirm=\"publish\".")
                return summary + tail
            await c.commit_edit(package, edit_id)
            return f"✅ Released version code {vc} to the {track} track for {package}."
        except PlayError as e:
            try:
                await c.delete_edit(package, edit_id)
            except PlayError:
                pass
            return f"Google Play error (nothing published, edit discarded): {e}"
```
Register in `cli.py` `build_tools` (near `AppStoreBuildUploadTool`):
```python
    from .tools.play_publish import PlayPublishTool
    registry.register(PlayPublishTool(interactive=interactive))
```

- [ ] **Step 4: Run tests**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_play_publish_tool.py -v` (Expected: PASS, 5) then `.venv/bin/python -m pytest tests/ -q | tail -2` and `.venv/bin/python -m ruff check spark_code/tools/play_publish.py spark_code/cli.py tests/test_play_publish_tool.py`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/tools/play_publish.py spark_code/cli.py tests/test_play_publish_tool.py
git commit -m "feat: play_publish tool — gradle AAB -> upload -> testing track, gated commit, prod refused"
```

## Live acceptance (controller-run, after Task 2 — NOT a subagent task)

Build GigLedger + upload + validate, stopping at the gate (NO confirm) so nothing is released:
```bash
cd ~/spark-code && .venv/bin/python -c "
import asyncio; from spark_code.tools.play_publish import PlayPublishTool
print(asyncio.run(PlayPublishTool().execute(
    project='/Users/alexandratitus/CodingProjects/gigledger-android',
    package='com.gigledger.android', track='internal')))"
```
Expected: gradle build → create edit → upload (may reject: versionCode 45 already exists on alpha) → if it gets past upload, validate then a "ready to release … confirm" summary with the edit deleted. **Never pass `confirm="publish"`.** If gradle fails on keystore/billing-key, surface it as an environment/setup issue.

## Self-Review

- **Spec coverage:** testing-track-only (T1 constant + T2 refusal), publish edit methods incl. media upload (T1), gradle build + full sequence + gate + delete-on-failure + delete-without-confirm (T2), production refused before any build/API (T2 test), live acceptance stops at gate. ✓
- **Placeholders:** none. ✓
- **Type consistency:** `upload_bundle`→`int` vc used in `assign_track`; sequence-order test asserts `["gradle","create","upload","assign","validate","commit"]`; delete-on-failure + delete-without-confirm both asserted. ✓
