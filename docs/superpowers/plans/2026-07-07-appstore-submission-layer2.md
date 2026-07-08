# iOS App Store Submission — Implementation Plan (Layer 2, safe slice)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Spark Code two tools — `appstore_status` (read-only readiness) and `appstore_submit` (hard-gated submit-for-review of an already-prepared iOS version) — backed by a small ASC REST client.

**Architecture:** `spark_code/appstore.py` signs an ES256 JWT from `~/.appstoreconnect/` and calls `api.appstoreconnect.apple.com/v1`. Two tools wrap it; the submit tool refuses on any readiness blocker, requires an explicit `confirm="submit"` argument, and refuses in non-interactive mode without it.

**Tech Stack:** Python 3.12, PyJWT (ES256), httpx (async), pytest.

## Global Constraints

- Changes land in the live `~/spark-code`, branch `feat/appstore-submission`.
- JWT: `alg=ES256`, headers `{kid: <key_id>, typ: "JWT"}`, payload `{iss: <issuer_id>, iat, exp ≤ iat+1200, aud: "appstoreconnect-v1"}`.
- Auth config: `~/.appstoreconnect/config.json` keys `key_id`, `issuer_id`, `key_file` (path to the `.p8`; may be relative to `~/.appstoreconnect/private_keys/`).
- **Never** issue `POST /v1/reviewSubmissions` (or any submit call) unless readiness passes AND `confirm == "submit"`.
- No build uploads, no metadata/screenshot writes anywhere in this plan.
- Submit sequence (verbatim shape): `POST /v1/reviewSubmissions {data:{type:"reviewSubmissions", attributes:{platform:"IOS"}, relationships:{app:{data:{type:"apps",id:APP}}}}}` → `POST /v1/reviewSubmissionItems {data:{type:"reviewSubmissionItems", relationships:{reviewSubmission:{data:{type:"reviewSubmissions",id:SID}}, appStoreVersion:{data:{type:"appStoreVersions",id:VID}}}}}` → `PATCH /v1/reviewSubmissions/{SID} {data:{type:"reviewSubmissions",id:SID,attributes:{submitted:true}}}`.
- Follow existing tool patterns in `spark_code/tools/` (see `read_file.py` for a read-only tool, `write_file.py` for a gated one).

---

### Task 1: ASC client core — auth + requests + app/version state

**Files:**
- Create: `spark_code/appstore.py`
- Test: `tests/test_appstore.py`

**Interfaces:**
- Produces:
  - `AscClient(config_dir="~/.appstoreconnect")` with `async get(path)`, `async post(path, body)`, `async patch(path, body)`, `async resolve_app(identifier) -> str`, `async get_submission_state(app_id) -> dict`.
  - `AscError(Exception)` raised on config/HTTP errors.
  - `get_submission_state` returns `{"version_id": str, "version": str, "state": str, "build_id": str|None, "build_number": str|None, "build_processing_state": str|None}`.

- [ ] **Step 1: Write the failing test (JWT shape + resolve_app)**

```python
# tests/test_appstore.py
import json, jwt as pyjwt, pytest
from spark_code.appstore import AscClient, AscError

# minimal ES256 key for signing tests
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

def _write_config(tmp_path):
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    (tmp_path / "AuthKey.p8").write_bytes(pem)
    (tmp_path / "config.json").write_text(json.dumps({
        "key_id": "KID123", "issuer_id": "ISS-abc", "key_file": "AuthKey.p8"}))
    return key

def test_jwt_has_required_claims_and_header(tmp_path):
    _write_config(tmp_path)
    c = AscClient(config_dir=str(tmp_path))
    tok = c._token()
    hdr = pyjwt.get_unverified_header(tok)
    assert hdr["kid"] == "KID123" and hdr["typ"] == "JWT" and hdr["alg"] == "ES256"
    payload = pyjwt.decode(tok, options={"verify_signature": False})
    assert payload["iss"] == "ISS-abc"
    assert payload["aud"] == "appstoreconnect-v1"
    assert 0 < payload["exp"] - payload["iat"] <= 1200

def test_missing_config_raises(tmp_path):
    with pytest.raises(AscError):
        AscClient(config_dir=str(tmp_path))._token()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore.py -v`
Expected: FAIL — `ModuleNotFoundError: spark_code.appstore`.

- [ ] **Step 3: Implement `appstore.py` (auth + client)**

```python
# spark_code/appstore.py
"""Minimal App Store Connect REST client for Spark Code (Layer 2, safe slice)."""
from __future__ import annotations
import json, time
from pathlib import Path
import httpx
import jwt as pyjwt

API = "https://api.appstoreconnect.apple.com"

class AscError(Exception):
    pass

class AscClient:
    def __init__(self, config_dir: str = "~/.appstoreconnect"):
        self._dir = Path(config_dir).expanduser()
        self._cfg = None
        self._tok = None
        self._tok_exp = 0.0

    def _config(self) -> dict:
        if self._cfg is None:
            p = self._dir / "config.json"
            if not p.exists():
                raise AscError(f"ASC config not found at {p}")
            self._cfg = json.loads(p.read_text())
        return self._cfg

    def _private_key(self) -> bytes:
        cfg = self._config()
        kf = Path(cfg["key_file"]).expanduser()
        for cand in (kf, self._dir / cfg["key_file"], self._dir / "private_keys" / cfg["key_file"]):
            if cand.exists():
                return cand.read_bytes()
        raise AscError(f"ASC private key not found: {cfg['key_file']}")

    def _token(self) -> str:
        now = time.time()
        if self._tok and now < self._tok_exp - 60:
            return self._tok
        cfg = self._config()
        iat = int(now)
        self._tok = pyjwt.encode(
            {"iss": cfg["issuer_id"], "iat": iat, "exp": iat + 1200, "aud": "appstoreconnect-v1"},
            self._private_key(), algorithm="ES256",
            headers={"kid": cfg["key_id"], "typ": "JWT"})
        self._tok_exp = iat + 1200
        return self._tok

    async def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        url = path if path.startswith("http") else f"{API}{path}"
        headers = {"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.request(method, url, headers=headers,
                                content=json.dumps(body) if body is not None else None)
        if r.status_code == 429:
            raise AscError("ASC rate limited (429) — try again shortly")
        if r.status_code >= 400:
            raise AscError(f"ASC {method} {path} → {r.status_code}: {r.text[:400]}")
        return r.json() if r.text else {}

    async def get(self, path: str) -> dict:
        return await self._req("GET", path)
    async def post(self, path: str, body: dict) -> dict:
        return await self._req("POST", path, body)
    async def patch(self, path: str, body: dict) -> dict:
        return await self._req("PATCH", path, body)

    async def resolve_app(self, identifier: str) -> str:
        data = (await self.get(f"/v1/apps?filter[bundleId]={identifier}&limit=2")).get("data", [])
        if not data:
            data = (await self.get(f"/v1/apps?filter[name]={identifier}&limit=2")).get("data", [])
        if not data:
            raise AscError(f"No app found for '{identifier}'")
        if len(data) > 1:
            raise AscError(f"'{identifier}' matched {len(data)} apps — use the exact bundle id")
        return data[0]["id"]

    async def get_submission_state(self, app_id: str) -> dict:
        versions = (await self.get(
            f"/v1/apps/{app_id}/appStoreVersions?limit=1&sort=-createdDate")).get("data", [])
        if not versions:
            raise AscError("No App Store version found for this app")
        v = versions[0]
        vid = v["id"]
        state = v["attributes"].get("appStoreState")
        version = v["attributes"].get("versionString")
        build_id = build_num = build_state = None
        try:
            b = (await self.get(f"/v1/appStoreVersions/{vid}/build")).get("data")
            if b:
                build_id = b["id"]
                bd = (await self.get(f"/v1/builds/{build_id}")).get("data", {})
                build_num = bd.get("attributes", {}).get("version")
                build_state = bd.get("attributes", {}).get("processingState")
        except AscError:
            pass
        return {"version_id": vid, "version": version, "state": state,
                "build_id": build_id, "build_number": build_num,
                "build_processing_state": build_state}
```

- [ ] **Step 4: Add a `resolve_app` test (mock httpx) and run**

Add to `tests/test_appstore.py`:
```python
import respx, httpx as _httpx  # respx may not be installed; if absent, monkeypatch AscClient._req instead
```
If `respx` is unavailable, monkeypatch instead:
```python
async def test_resolve_app_single_match(tmp_path, monkeypatch):
    _write_config(tmp_path)
    c = AscClient(config_dir=str(tmp_path))
    async def fake(method, path, body=None):
        assert "filter[bundleId]=com.x" in path
        return {"data": [{"id": "APP1"}]}
    monkeypatch.setattr(c, "_req", fake)
    assert await c.resolve_app("com.x") == "APP1"

async def test_resolve_app_ambiguous_raises(tmp_path, monkeypatch):
    _write_config(tmp_path)
    c = AscClient(config_dir=str(tmp_path))
    async def fake(method, path, body=None):
        return {"data": [{"id": "A"}, {"id": "B"}]}
    monkeypatch.setattr(c, "_req", fake)
    with pytest.raises(AscError):
        await c.resolve_app("com.x")
```
Add `asyncio_mode = auto` usage note: mark async tests with `@pytest.mark.asyncio` if the repo isn't in auto mode — check `pyproject.toml`/`pytest.ini`; the existing async tests (e.g. `tests/test_dispatch_provider.py`) show the repo's convention — follow it exactly.

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/appstore.py tests/test_appstore.py
git commit -m "feat: ASC REST client — ES256 JWT auth, resolve_app, submission state"
```

---

### Task 2: Readiness checklist

**Files:**
- Modify: `spark_code/appstore.py` (add `readiness`)
- Test: `tests/test_appstore.py`

**Interfaces:**
- Consumes: `AscClient.get`, `get_submission_state`.
- Produces: `async readiness(client, app_id) -> list[dict]` where each item is `{"check": str, "ok": bool, "hint": str}`. `blockers(checks) -> list[dict]` helper returns the `ok=False` items.

- [ ] **Step 1: Write the failing test**

```python
async def test_readiness_flags_missing_build(tmp_path, monkeypatch):
    _write_config(tmp_path)
    from spark_code.appstore import AscClient, readiness, blockers
    c = AscClient(config_dir=str(tmp_path))
    async def fake_state(app_id):
        return {"version_id": "V", "version": "1.0", "state": "PREPARE_FOR_SUBMISSION",
                "build_id": None, "build_number": None, "build_processing_state": None}
    monkeypatch.setattr(c, "get_submission_state", fake_state)
    async def fake_get(path): return {"data": []}
    monkeypatch.setattr(c, "get", fake_get)
    checks = await readiness(c, "APP")
    assert any(x["check"].lower().startswith("build") and not x["ok"] for x in checks)
    assert blockers(checks)  # at least one blocker

async def test_readiness_all_ok(tmp_path, monkeypatch):
    _write_config(tmp_path)
    from spark_code.appstore import AscClient, readiness, blockers
    c = AscClient(config_dir=str(tmp_path))
    async def fake_state(app_id):
        return {"version_id": "V", "version": "1.0", "state": "PREPARE_FOR_SUBMISSION",
                "build_id": "B", "build_number": "42", "build_processing_state": "VALID"}
    monkeypatch.setattr(c, "get_submission_state", fake_state)
    async def fake_get(path):
        if path.endswith("/builds/B"):
            return {"data": {"attributes": {"usesNonExemptEncryption": False}}}
        if "appScreenshotSets" in path:
            return {"data": [{"id": "s1"}]}
        return {"data": {}}
    monkeypatch.setattr(c, "get", fake_get)
    checks = await readiness(c, "APP")
    assert not blockers(checks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore.py -k readiness -v`
Expected: FAIL — `ImportError: cannot import name 'readiness'`.

- [ ] **Step 3: Implement `readiness` + `blockers`**

```python
# append to spark_code/appstore.py
async def readiness(client: "AscClient", app_id: str) -> list[dict]:
    st = await client.get_submission_state(app_id)
    checks: list[dict] = []
    build_ok = bool(st["build_id"]) and st["build_processing_state"] == "VALID"
    checks.append({"check": "Build attached & processed",
                   "ok": build_ok,
                   "hint": "" if build_ok else "Attach a build with processingState=VALID"})
    enc_ok = False
    if st["build_id"]:
        bd = (await client.get(f"/v1/builds/{st['build_id']}")).get("data", {})
        enc_ok = bd.get("attributes", {}).get("usesNonExemptEncryption") is not None
    checks.append({"check": "Export-compliance / encryption flag set",
                   "ok": enc_ok, "hint": "" if enc_ok else "Set usesNonExemptEncryption on the build"})
    shots = (await client.get(
        f"/v1/appStoreVersions/{st['version_id']}/appStoreVersionLocalizations")).get("data", [])
    shot_ok = False
    if shots:
        sets = (await client.get(
            f"/v1/appStoreVersionLocalizations/{shots[0]['id']}/appScreenshotSets")).get("data", [])
        shot_ok = bool(sets)
    checks.append({"check": "Screenshots present",
                   "ok": shot_ok, "hint": "" if shot_ok else "Upload required screenshots"})
    return checks

def blockers(checks: list[dict]) -> list[dict]:
    return [c for c in checks if not c["ok"]]
```

- [ ] **Step 4: Run tests**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore.py -k readiness -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/appstore.py tests/test_appstore.py
git commit -m "feat: ASC readiness checklist (build/encryption/screenshots) + blockers"
```

---

### Task 3: `appstore_status` tool (read-only) + registration

**Files:**
- Create: `spark_code/tools/appstore_status.py`
- Modify: `spark_code/cli.py` (register in `build_tools`, near the other `registry.register(...)` calls ~692-718)
- Test: `tests/test_appstore_tools.py`

**Interfaces:**
- Consumes: `AscClient`, `readiness`, `get_submission_state`.
- Produces: `AppStoreStatusTool` — `name="appstore_status"`, `is_read_only=True`, `requires_permission=False`, param `app` (string). Returns a text report.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_appstore_tools.py
import pytest
from spark_code.tools.appstore_status import AppStoreStatusTool

async def test_status_reports_state_and_readiness(monkeypatch):
    t = AppStoreStatusTool()
    async def fake_resolve(self, ident): return "APP1"
    async def fake_state(self, app_id):
        return {"version_id":"V","version":"1.3.22","state":"PREPARE_FOR_SUBMISSION",
                "build_id":"B","build_number":"55","build_processing_state":"VALID"}
    async def fake_ready(client, app_id):
        return [{"check":"Build attached & processed","ok":True,"hint":""},
                {"check":"Screenshots present","ok":False,"hint":"Upload required screenshots"}]
    monkeypatch.setattr("spark_code.appstore.AscClient.resolve_app", fake_resolve)
    monkeypatch.setattr("spark_code.appstore.AscClient.get_submission_state", fake_state)
    monkeypatch.setattr("spark_code.tools.appstore_status.readiness", fake_ready)
    out = await t.execute(app="com.gigledger")
    assert "1.3.22" in out and "PREPARE_FOR_SUBMISSION" in out
    assert "55" in out and "Screenshots present" in out and "Upload required screenshots" in out
    assert t.is_read_only is True and t.requires_permission is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore_tools.py -k status -v`
Expected: FAIL — `ModuleNotFoundError: spark_code.tools.appstore_status`.

- [ ] **Step 3: Implement the tool**

```python
# spark_code/tools/appstore_status.py
from __future__ import annotations
from .base import Tool
from ..appstore import AscClient, readiness, AscError

class AppStoreStatusTool(Tool):
    @property
    def name(self) -> str: return "appstore_status"
    @property
    def description(self) -> str:
        return ("Read-only: show an iOS app's current App Store version state, attached "
                "build, and a submission-readiness checklist. Input: app (bundle id or name).")
    @property
    def parameters(self) -> dict:
        return {"type": "object",
                "properties": {"app": {"type": "string",
                    "description": "App bundle id (preferred) or exact name."}},
                "required": ["app"]}
    @property
    def is_read_only(self) -> bool: return True
    @property
    def requires_permission(self) -> bool: return False
    async def execute(self, app: str = "", **kwargs) -> str:
        if not app:
            return "Error: 'app' (bundle id or name) is required."
        try:
            c = AscClient()
            app_id = await c.resolve_app(app)
            st = await c.get_submission_state(app_id)
            checks = await readiness(c, app_id)
        except AscError as e:
            return f"App Store Connect error: {e}"
        lines = [f"App: {app}  (id {app_id})",
                 f"Version {st['version']} — state: {st['state']}",
                 f"Build: {st['build_number'] or '(none attached)'} "
                 f"[{st['build_processing_state'] or '-'}]",
                 "Readiness:"]
        for c_ in checks:
            lines.append(f"  {'✓' if c_['ok'] else '✗'} {c_['check']}"
                         + (f" — {c_['hint']}" if not c_['ok'] and c_['hint'] else ""))
        blk = [c_ for c_ in checks if not c_['ok']]
        lines.append("READY to submit." if not blk else f"NOT ready — {len(blk)} blocker(s).")
        return "\n".join(lines)
```
Register in `cli.py` `build_tools` (next to the other `registry.register(...)` calls):
```python
    from .tools.appstore_status import AppStoreStatusTool
    registry.register(AppStoreStatusTool())
```

- [ ] **Step 4: Run test + a live read acceptance**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore_tools.py -k status -v`
Expected: PASS.
Live acceptance (safe, read-only) — run against a real app and eyeball it:
```bash
cd ~/spark-code && .venv/bin/python -c "
import asyncio; from spark_code.tools.appstore_status import AppStoreStatusTool
print(asyncio.run(AppStoreStatusTool().execute(app='com.alexandratitus.GigLedger')))"
```
Expected: prints GigLedger's real version state + readiness (no error). If auth fails, check `~/.appstoreconnect/config.json` `key_file`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/tools/appstore_status.py spark_code/cli.py tests/test_appstore_tools.py
git commit -m "feat: appstore_status tool — read-only version state + readiness"
```

---

### Task 4: `appstore_submit` tool (hard-gated) + registration

**Files:**
- Create: `spark_code/tools/appstore_submit.py`
- Modify: `spark_code/cli.py` (register in `build_tools`; thread an `interactive: bool` into `build_tools` and pass `interactive=True` from `run_interactive` ~2866 call and `interactive=False` from `_one_shot` ~4395 call)
- Test: `tests/test_appstore_tools.py`

**Interfaces:**
- Consumes: `AscClient`, `readiness`, `blockers`, `get_submission_state`.
- Produces: `AppStoreSubmitTool(interactive: bool = True)` — `name="appstore_submit"`, `is_read_only=False`, `requires_permission=True`, params `app`, `version`, `confirm`.

- [ ] **Step 1: Write the failing tests (the gate)**

```python
from spark_code.tools.appstore_submit import AppStoreSubmitTool

def _client_monkeypatch(monkeypatch, ready=True, calls=None):
    async def fake_resolve(self, ident): return "APP1"
    async def fake_state(self, app_id):
        return {"version_id":"V1","version":"1.3.22","state":"PREPARE_FOR_SUBMISSION",
                "build_id":"B","build_number":"55","build_processing_state":"VALID"}
    async def fake_ready(client, app_id):
        return [{"check":"Build attached & processed","ok":ready,"hint":"" if ready else "attach build"}]
    async def fake_post(self, path, body):
        if calls is not None: calls.append(("POST", path))
        return {"data":{"id":"SID1"}}
    async def fake_patch(self, path, body):
        if calls is not None: calls.append(("PATCH", path))
        return {"data":{"id":"SID1","attributes":{"state":"WAITING_FOR_REVIEW"}}}
    monkeypatch.setattr("spark_code.appstore.AscClient.resolve_app", fake_resolve)
    monkeypatch.setattr("spark_code.appstore.AscClient.get_submission_state", fake_state)
    monkeypatch.setattr("spark_code.tools.appstore_submit.readiness", fake_ready)
    monkeypatch.setattr("spark_code.appstore.AscClient.post", fake_post)
    monkeypatch.setattr("spark_code.appstore.AscClient.patch", fake_patch)

async def test_submit_refuses_when_not_ready(monkeypatch):
    calls=[]; _client_monkeypatch(monkeypatch, ready=False, calls=calls)
    out = await AppStoreSubmitTool().execute(app="com.x", version="1.3.22", confirm="submit")
    assert "not ready" in out.lower() and calls == []  # never hit the API

async def test_submit_without_confirm_shows_summary_only(monkeypatch):
    calls=[]; _client_monkeypatch(monkeypatch, ready=True, calls=calls)
    out = await AppStoreSubmitTool().execute(app="com.x", version="1.3.22")
    assert "1.3.22" in out and "confirm" in out.lower() and calls == []

async def test_submit_confirmed_runs_sequence_in_order(monkeypatch):
    calls=[]; _client_monkeypatch(monkeypatch, ready=True, calls=calls)
    out = await AppStoreSubmitTool().execute(app="com.x", version="1.3.22", confirm="submit")
    assert [m for m,_ in calls] == ["POST","POST","PATCH"]
    assert "reviewSubmissions" in calls[0][1] and "reviewSubmissionItems" in calls[1][1]
    assert "submitted" in out.lower() or "waiting_for_review" in out.lower()

async def test_headless_refuses_without_confirm(monkeypatch):
    calls=[]; _client_monkeypatch(monkeypatch, ready=True, calls=calls)
    out = await AppStoreSubmitTool(interactive=False).execute(app="com.x", version="1.3.22")
    assert "confirm=" in out and calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore_tools.py -k submit -v`
Expected: FAIL — `ModuleNotFoundError: spark_code.tools.appstore_submit`.

- [ ] **Step 3: Implement the tool**

```python
# spark_code/tools/appstore_submit.py
from __future__ import annotations
from .base import Tool
from ..appstore import AscClient, readiness, blockers, AscError

class AppStoreSubmitTool(Tool):
    def __init__(self, interactive: bool = True):
        self._interactive = interactive
    @property
    def name(self) -> str: return "appstore_submit"
    @property
    def description(self) -> str:
        return ("Submit an ALREADY-PREPARED iOS App Store version for review. Refuses if the "
                "version isn't ready. Requires confirm='submit' to actually submit; without it, "
                "returns a summary only. Input: app (bundle id/name), version, confirm.")
    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {
            "app": {"type": "string", "description": "App bundle id (preferred) or exact name."},
            "version": {"type": "string", "description": "Version string to submit, e.g. 1.3.22."},
            "confirm": {"type": "string",
                "description": "Pass exactly 'submit' to actually submit for review. Omit to preview."}},
            "required": ["app", "version"]}
    @property
    def is_read_only(self) -> bool: return False
    @property
    def requires_permission(self) -> bool: return True
    async def execute(self, app: str = "", version: str = "", confirm: str = "", **kwargs) -> str:
        if not app or not version:
            return "Error: 'app' and 'version' are required."
        try:
            c = AscClient()
            app_id = await c.resolve_app(app)
            st = await c.get_submission_state(app_id)
            if st["version"] != version:
                return (f"Refusing: the editable version is {st['version']} (state {st['state']}), "
                        f"not {version}. Re-check with appstore_status.")
            checks = await readiness(c, app_id)
            blk = blockers(checks)
            if blk:
                return ("NOT ready to submit — fix these first:\n"
                        + "\n".join(f"  ✗ {b['check']} — {b['hint']}" for b in blk))
            summary = (f"About to submit for App Review:\n"
                       f"  App: {app} (id {app_id})\n"
                       f"  Version: {st['version']}  |  Build: {st['build_number']}  |  Platform: iOS")
            if confirm != "submit":
                if not self._interactive:
                    return summary + "\n\nHeadless: refusing. Re-invoke with confirm=\"submit\" to proceed."
                return summary + "\n\nThis is a real submission to Apple. Re-invoke with confirm=\"submit\" to proceed."
            sub = (await c.post("/v1/reviewSubmissions", {"data": {"type": "reviewSubmissions",
                "attributes": {"platform": "IOS"},
                "relationships": {"app": {"data": {"type": "apps", "id": app_id}}}}})).get("data", {})
            sid = sub["id"]
            await c.post("/v1/reviewSubmissionItems", {"data": {"type": "reviewSubmissionItems",
                "relationships": {
                    "reviewSubmission": {"data": {"type": "reviewSubmissions", "id": sid}},
                    "appStoreVersion": {"data": {"type": "appStoreVersions", "id": st["version_id"]}}}}})
            res = (await c.patch(f"/v1/reviewSubmissions/{sid}", {"data": {"type": "reviewSubmissions",
                "id": sid, "attributes": {"submitted": True}}})).get("data", {})
            state = res.get("attributes", {}).get("state", "SUBMITTED")
            return (f"✅ Submitted {app} {st['version']} (build {st['build_number']}) for review. "
                    f"reviewSubmission {sid} → state {state}. "
                    f"(A wrong submission can be canceled via PATCH reviewSubmissions/{sid} {{canceled:true}}.)")
        except AscError as e:
            return f"App Store Connect error: {e}"
```
Register in `cli.py` `build_tools`, passing the interactive flag:
```python
    from .tools.appstore_submit import AppStoreSubmitTool
    registry.register(AppStoreSubmitTool(interactive=interactive))
```
Add `interactive: bool = True` param to `build_tools`; pass `interactive=True` at the `run_interactive` call site (~2866) and `interactive=False` at the `_one_shot` call site (~4395).

- [ ] **Step 4: Run tests**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_appstore_tools.py -k submit -v`
Expected: PASS (4 tests). Then full guard: `.venv/bin/python -m pytest tests/ -q | tail -2` and `.venv/bin/python -m ruff check spark_code/`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/tools/appstore_submit.py spark_code/cli.py tests/test_appstore_tools.py
git commit -m "feat: appstore_submit tool — readiness gate + typed confirm + headless refusal"
```

## Self-Review

- **Spec coverage:** appstore.py client+JWT (T1), readiness (T2), read-only status tool + live acceptance (T3), gated submit tool with typed-confirm + headless-refusal + version validation + submit sequence (T4). No build upload / metadata writes anywhere. ✓
- **Placeholders:** none — full code per step. The one env-dependent note (respx vs monkeypatch, async-test convention) points the implementer to the repo's existing convention rather than guessing. ✓
- **Type consistency:** `get_submission_state` dict keys (`version_id`,`version`,`state`,`build_number`,`build_processing_state`) are produced in T1 and consumed identically in T2/T3/T4; `readiness`→`blockers` shape consistent; tool signatures match their tests. ✓
