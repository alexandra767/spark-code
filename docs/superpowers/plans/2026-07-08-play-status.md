# Android/Play Status — Implementation Plan (read-only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only `play_status` tool that lists an Android app's Google Play tracks (production/beta/alpha/internal + wear:*), their versionCodes, release names, and status.

**Architecture:** `spark_code/play.py` — a Play Developer API client: service-account → RS256 jwt-bearer OAuth (PyJWT + httpx, async), and a `track_status(package)` that opens an ephemeral edit, reads tracks, and deletes the edit in a `finally` (nothing committed → read-only in effect). The tool renders it.

**Tech Stack:** Python 3.12 (async httpx, PyJWT RS256), pytest. No new deps.

## Global Constraints

- Changes land in the live `~/spark-code`, branch `feat/play-status`.
- Repo uses `asyncio_mode = "auto"` (async tests need NO decorator); use `monkeypatch`, not respx.
- OAuth: RS256 jwt-bearer assertion `{iss: client_email, scope: "https://www.googleapis.com/auth/androidpublisher", aud: token_uri, iat, exp ≤ iat+3600}` → POST `token_uri` with `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer`.
- SA resolution order: env `PLAY_SA_JSON` → `~/.playconsole/service-account.json` → `~/CodingProjects/revdash/secrets/play-rtdn-sa.json`.
- API base `https://androidpublisher.googleapis.com/androidpublisher/v3`.
- `track_status` MUST delete the ephemeral edit in a `finally` (even if the tracks read raises). Never commit an edit. Never log the token.
- `play_status` tool: `is_read_only=True`, `requires_permission=False`.

---

### Task 1: `play.py` — Play Developer API client

**Files:**
- Create: `spark_code/play.py`
- Test: `tests/test_play.py`

**Interfaces produced (Task 2 consumes):**
- `PlayError(Exception)`
- `PlayClient(sa_path=None)` with `_build_assertion() -> str`, `async _token() -> str`, `async get(path)`, `async post(path, body=None)`, `async delete(path)`, `async track_status(package) -> list[dict]` (items `{track, status, version_name, version_codes}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_play.py
import json, time, jwt as pyjwt, pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from spark_code.play import PlayClient, PlayError

def _sa_file(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    p = tmp_path / "sa.json"
    p.write_text(json.dumps({"client_email": "bot@proj.iam.gserviceaccount.com",
        "private_key": pem, "token_uri": "https://oauth2.googleapis.com/token"}))
    return str(p)

def test_build_assertion_claims(tmp_path):
    c = PlayClient(sa_path=_sa_file(tmp_path))
    tok = c._build_assertion()
    hdr = pyjwt.get_unverified_header(tok)
    assert hdr["alg"] == "RS256"
    payload = pyjwt.decode(tok, options={"verify_signature": False})
    assert payload["iss"] == "bot@proj.iam.gserviceaccount.com"
    assert payload["scope"] == "https://www.googleapis.com/auth/androidpublisher"
    assert payload["aud"] == "https://oauth2.googleapis.com/token"
    assert 0 < payload["exp"] - payload["iat"] <= 3600

def test_missing_sa_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("PLAY_SA_JSON", raising=False)
    monkeypatch.setattr("spark_code.play._DEFAULT_SA_CANDIDATES", [str(tmp_path / "nope.json")])
    with pytest.raises(PlayError):
        PlayClient()

async def test_track_status_parses_and_deletes_edit(tmp_path, monkeypatch):
    c = PlayClient(sa_path=_sa_file(tmp_path))
    calls = []
    async def fake_req(method, path, body=None):
        calls.append((method, path))
        if method == "POST" and path.endswith("/edits"):
            return {"id": "EDIT1"}
        if method == "GET" and path.endswith("/tracks"):
            return {"tracks": [
                {"track": "production", "releases": [{"status": "completed", "name": "1.0.34", "versionCodes": ["44"]}]},
                {"track": "beta", "releases": []}]}
        return {}
    monkeypatch.setattr(c, "_req", fake_req)
    out = await c.track_status("com.x.app")
    assert {"track": "production", "status": "completed", "version_name": "1.0.34", "version_codes": ["44"]} in out
    assert {"track": "beta", "status": None, "version_name": None, "version_codes": []} in out
    assert ("DELETE", "/applications/com.x.app/edits/EDIT1") in calls  # ephemeral edit cleaned up

async def test_track_status_deletes_edit_even_on_read_error(tmp_path, monkeypatch):
    c = PlayClient(sa_path=_sa_file(tmp_path))
    deleted = []
    async def fake_req(method, path, body=None):
        if method == "POST" and path.endswith("/edits"):
            return {"id": "EDIT2"}
        if method == "GET":
            raise PlayError("boom")
        if method == "DELETE":
            deleted.append(path)
            return {}
        return {}
    monkeypatch.setattr(c, "_req", fake_req)
    with pytest.raises(PlayError):
        await c.track_status("com.x.app")
    assert deleted == ["/applications/com.x.app/edits/EDIT2"]  # deleted despite the read error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_play.py -v`
Expected: FAIL — `ModuleNotFoundError: spark_code.play`.

- [ ] **Step 3: Implement `play.py`**

```python
# spark_code/play.py
"""Google Play Developer API client for Spark Code (read-only status slice)."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import httpx
import jwt as pyjwt

API = "https://androidpublisher.googleapis.com/androidpublisher/v3"
SCOPE = "https://www.googleapis.com/auth/androidpublisher"
_DEFAULT_SA_CANDIDATES = [
    str(Path.home() / ".playconsole" / "service-account.json"),
    str(Path.home() / "CodingProjects" / "revdash" / "secrets" / "play-rtdn-sa.json"),
]


class PlayError(Exception):
    pass


def _resolve_sa_path(sa_path=None) -> str:
    candidates = ([sa_path] if sa_path else []) + [os.environ.get("PLAY_SA_JSON")] + _DEFAULT_SA_CANDIDATES
    for cand in candidates:
        if cand and Path(cand).expanduser().exists():
            return str(Path(cand).expanduser())
    raise PlayError("No Play service-account JSON found "
                    "(set PLAY_SA_JSON or ~/.playconsole/service-account.json)")


class PlayClient:
    def __init__(self, sa_path=None):
        self._sa = json.loads(Path(_resolve_sa_path(sa_path)).read_text())
        self._tok = None
        self._tok_exp = 0.0

    def _build_assertion(self) -> str:
        iat = int(time.time())
        return pyjwt.encode(
            {"iss": self._sa["client_email"], "scope": SCOPE,
             "aud": self._sa["token_uri"], "iat": iat, "exp": iat + 3600},
            self._sa["private_key"], algorithm="RS256")

    async def _token(self) -> str:
        now = time.time()
        if self._tok and now < self._tok_exp - 60:
            return self._tok
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.post(self._sa["token_uri"], data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": self._build_assertion()})
        if r.status_code != 200:
            raise PlayError(f"Play OAuth failed: {r.status_code} {r.text[:200]}")
        self._tok = r.json()["access_token"]
        self._tok_exp = time.time() + 3600
        return self._tok

    async def _req(self, method: str, path: str, body=None) -> dict:
        tok = await self._token()
        url = path if path.startswith("http") else f"{API}{path}"
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.request(method, url, headers={"Authorization": f"Bearer {tok}"}, json=body)
        if r.status_code >= 400:
            raise PlayError(f"Play {method} {path} → {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else {}

    async def get(self, path: str) -> dict:
        return await self._req("GET", path)

    async def post(self, path: str, body=None) -> dict:
        return await self._req("POST", path, body)

    async def delete(self, path: str) -> dict:
        return await self._req("DELETE", path)

    async def track_status(self, package: str) -> list[dict]:
        edit = await self.post(f"/applications/{package}/edits", {})
        eid = edit["id"]
        try:
            data = await self.get(f"/applications/{package}/edits/{eid}/tracks")
        finally:
            try:
                await self.delete(f"/applications/{package}/edits/{eid}")
            except PlayError:
                pass
        out = []
        for t in data.get("tracks", []):
            rels = t.get("releases") or []
            rel = rels[0] if rels else {}
            out.append({"track": t.get("track"), "status": rel.get("status"),
                        "version_name": rel.get("name"),
                        "version_codes": rel.get("versionCodes") or []})
        return out
```

- [ ] **Step 4: Run tests + ruff**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_play.py -v` (Expected: PASS, 4 tests) then `.venv/bin/python -m ruff check spark_code/play.py tests/test_play.py`.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/play.py tests/test_play.py
git commit -m "feat: Play Developer API client — SA jwt-bearer OAuth + ephemeral-edit track_status"
```

---

### Task 2: `play_status` tool + registration

**Files:**
- Create: `spark_code/tools/play_status.py`
- Modify: `spark_code/cli.py` (register in `build_tools`, near the appstore tools)
- Test: `tests/test_play_status_tool.py`

**Interfaces:**
- Consumes: `spark_code.play.PlayClient` (`track_status`), `PlayError`; `Tool` base.
- Produces: `PlayStatusTool` — `name="play_status"`, `is_read_only=True`, `requires_permission=False`, param `package`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_play_status_tool.py
import pytest
from spark_code.tools.play_status import PlayStatusTool

async def test_status_renders_tracks(monkeypatch):
    async def fake_track_status(self, package):
        return [{"track": "production", "status": "completed", "version_name": "1.0.34", "version_codes": ["44"]},
                {"track": "alpha", "status": "completed", "version_name": "1.0.35 (45)", "version_codes": ["45"]},
                {"track": "beta", "status": None, "version_name": None, "version_codes": []}]
    monkeypatch.setattr("spark_code.play.PlayClient.track_status", fake_track_status)
    t = PlayStatusTool()
    out = await t.execute(package="com.gigledger.android")
    assert "production" in out and "1.0.34" in out and "44" in out
    assert "alpha" in out and "1.0.35 (45)" in out
    assert "beta" in out  # empty track still listed
    assert t.is_read_only is True and t.requires_permission is False

async def test_status_missing_package(monkeypatch):
    out = await PlayStatusTool().execute(package="")
    assert "required" in out.lower()

async def test_status_surfaces_play_error(monkeypatch):
    from spark_code.play import PlayError
    async def boom(self, package):
        raise PlayError("no access")
    monkeypatch.setattr("spark_code.play.PlayClient.track_status", boom)
    out = await PlayStatusTool().execute(package="com.x")
    assert "no access" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_play_status_tool.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the tool**

```python
# spark_code/tools/play_status.py
from __future__ import annotations
from .base import Tool
from ..play import PlayClient, PlayError


class PlayStatusTool(Tool):
    @property
    def name(self) -> str: return "play_status"

    @property
    def description(self) -> str:
        return ("Read-only: show a Google Play app's tracks (production/beta/alpha/internal, "
                "plus wear:*), each with its current version name, versionCode(s), and release "
                "status. Input: package (e.g. com.gigledger.android).")

    @property
    def parameters(self) -> dict:
        return {"type": "object",
                "properties": {"package": {"type": "string",
                    "description": "Android package name, e.g. com.gigledger.android"}},
                "required": ["package"]}

    @property
    def is_read_only(self) -> bool: return True

    @property
    def requires_permission(self) -> bool: return False

    async def execute(self, package: str = "", **kwargs) -> str:
        if not package:
            return "Error: 'package' (e.g. com.gigledger.android) is required."
        try:
            tracks = await PlayClient().track_status(package)
        except PlayError as e:
            return f"Google Play error: {e}"
        if not tracks:
            return f"{package}: no tracks found."
        lines = [f"Google Play — {package}:"]
        for t in tracks:
            vc = ", ".join(t["version_codes"]) if t["version_codes"] else "-"
            if t["version_name"] or t["version_codes"]:
                lines.append(f"  {t['track']}: {t['version_name'] or '?'} (vc {vc}) — {t['status'] or '?'}")
            else:
                lines.append(f"  {t['track']}: (no release)")
        return "\n".join(lines)
```
Register in `cli.py` `build_tools` (near `AppStoreStatusTool`):
```python
    from .tools.play_status import PlayStatusTool
    registry.register(PlayStatusTool())
```

- [ ] **Step 4: Run test + a live acceptance**

Run: `cd ~/spark-code && .venv/bin/python -m pytest tests/test_play_status_tool.py -v` (Expected: PASS) then `.venv/bin/python -m pytest tests/ -q | tail -2` and `.venv/bin/python -m ruff check spark_code/tools/play_status.py spark_code/cli.py tests/test_play_status_tool.py`.
Live acceptance (safe, read-only) — run against real GigLedger and eyeball:
```bash
cd ~/spark-code && .venv/bin/python -c "
import asyncio; from spark_code.tools.play_status import PlayStatusTool
print(asyncio.run(PlayStatusTool().execute(package='com.gigledger.android')))"
```
Expected: prints the real tracks (production 1.0.34 vc 44, alpha 1.0.35 (45) vc 45, internal 44, beta no release, plus wear:*). No error.

- [ ] **Step 5: Commit**

```bash
cd ~/spark-code && git add spark_code/tools/play_status.py spark_code/cli.py tests/test_play_status_tool.py
git commit -m "feat: play_status tool — read-only Play track/version status"
```

## Self-Review

- **Spec coverage:** OAuth jwt-bearer + SA resolution + ephemeral-edit `track_status` with delete-in-`finally` (T1); read-only tool + registration + live acceptance (T2). ✓
- **Placeholders:** none — full code per step. ✓
- **Type consistency:** `track_status` item keys (`track/status/version_name/version_codes`) produced in T1, consumed identically in T2 render + tests. ✓
