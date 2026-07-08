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
TESTING_TRACKS = frozenset({"internal", "alpha", "beta"})
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
            except Exception:
                # Best-effort cleanup: a transient network error here must never
                # mask a read error, nor turn a good read into a raised exception
                # (the uncommitted edit auto-expires on Play's side).
                pass
        out = []
        for t in data.get("tracks", []):
            rels = t.get("releases") or []
            rel = rels[0] if rels else {}
            out.append({"track": t.get("track"), "status": rel.get("status"),
                        "version_name": rel.get("name"),
                        "version_codes": rel.get("versionCodes") or []})
        return out

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
