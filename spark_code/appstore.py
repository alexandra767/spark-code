"""Minimal App Store Connect REST client for Spark Code (Layer 2, safe slice)."""
from __future__ import annotations

import json
import time
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
