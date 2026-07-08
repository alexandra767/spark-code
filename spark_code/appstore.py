"""Minimal App Store Connect REST client for Spark Code (Layer 2, safe slice)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote

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

    @staticmethod
    def _cfg_val(cfg: dict, key: str) -> str:
        try:
            return cfg[key]
        except KeyError:
            raise AscError(f"ASC config missing '{key}'") from None

    def _private_key(self) -> bytes:
        cfg = self._config()
        key_file = self._cfg_val(cfg, "key_file")
        kf = Path(key_file).expanduser()
        for cand in (kf, self._dir / key_file, self._dir / "private_keys" / key_file):
            if cand.exists():
                return cand.read_bytes()
        raise AscError(f"ASC private key not found: {key_file}")

    def _token(self) -> str:
        now = time.time()
        if self._tok and now < self._tok_exp - 60:
            return self._tok
        cfg = self._config()
        issuer_id = self._cfg_val(cfg, "issuer_id")
        key_id = self._cfg_val(cfg, "key_id")
        iat = int(now)
        self._tok = pyjwt.encode(
            {"iss": issuer_id, "iat": iat, "exp": iat + 1200, "aud": "appstoreconnect-v1"},
            self._private_key(), algorithm="ES256",
            headers={"kid": key_id, "typ": "JWT"})
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
        enc = quote(identifier, safe="")
        data = (await self.get(f"/v1/apps?filter[bundleId]={enc}&limit=2")).get("data", [])
        if not data:
            data = (await self.get(f"/v1/apps?filter[name]={enc}&limit=2")).get("data", [])
        if not data:
            raise AscError(f"No app found for '{identifier}'")
        if len(data) > 1:
            raise AscError(f"'{identifier}' matched {len(data)} apps — use the exact bundle id")
        return data[0]["id"]

    async def get_submission_state(self, app_id: str) -> dict:
        # ASC rejects `sort` on the app→appStoreVersions relationship
        # (PARAMETER_ERROR.ILLEGAL); the endpoint already returns versions
        # newest-first, and Apple keeps at most one editable version (always the
        # newest), so data[0] is the current/editable version — same selection
        # the appstore-connect-submission playbook uses.
        versions = (await self.get(
            f"/v1/apps/{app_id}/appStoreVersions?limit=1")).get("data", [])
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


async def _primary_locale_screenshot_check(client: "AscClient", app_id: str, version_id: str) -> tuple[bool, str]:
    """Verify at least one actual screenshot image exists for the app's primary locale.

    Fails safe: any missing/unexpected data resolves to ok=False with a hint,
    never a silent ok=True. Falls back to the first localization only if none
    matches the primary locale, noting that explicitly in the hint.
    """
    app_data = (await client.get(f"/v1/apps/{app_id}")).get("data") or {}
    primary_locale = (app_data.get("attributes") or {}).get("primaryLocale")

    locs = (await client.get(
        f"/v1/appStoreVersions/{version_id}/appStoreVersionLocalizations")).get("data") or []
    if not locs:
        return False, "No App Store version localizations found — add a localization and upload screenshots."

    loc = None
    if primary_locale:
        loc = next((L for L in locs if (L.get("attributes") or {}).get("locale") == primary_locale), None)

    fallback_note = ""
    if loc is None:
        loc = locs[0]
        fallback_locale = (loc.get("attributes") or {}).get("locale", loc.get("id"))
        if primary_locale:
            fallback_note = (f" Note: no localization matched primary locale '{primary_locale}'; "
                              f"checked '{fallback_locale}' instead.")
        else:
            fallback_note = (" Note: could not determine primary locale; "
                              f"checked '{fallback_locale}' instead.")

    loc_id = loc["id"]
    locale_label = (loc.get("attributes") or {}).get("locale", loc_id)

    sets = (await client.get(
        f"/v1/appStoreVersionLocalizations/{loc_id}/appScreenshotSets")).get("data") or []
    if not sets:
        return False, (f"No screenshot sets found for locale {locale_label} — "
                        f"upload screenshots in App Store Connect.{fallback_note}")

    for s in sets:
        shots = (await client.get(
            f"/v1/appScreenshotSets/{s['id']}/appScreenshots")).get("data") or []
        if len(shots) > 0:
            return True, fallback_note.strip()

    return False, (f"No uploaded screenshots found for primary locale {locale_label} — "
                    f"upload them in App Store Connect.{fallback_note}")


async def readiness(client: "AscClient", app_id: str) -> list[dict]:
    checks: list[dict] = []

    try:
        st = await client.get_submission_state(app_id)
    except Exception as e:
        hint = f"Could not verify submission state: {e}"
        return [
            {"check": "Build attached & processed", "ok": False, "hint": hint},
            {"check": "Export-compliance / encryption flag set", "ok": False, "hint": hint},
            {"check": "Screenshots present", "ok": False, "hint": hint},
        ]

    try:
        build_ok = bool(st["build_id"]) and st["build_processing_state"] == "VALID"
        checks.append({"check": "Build attached & processed",
                       "ok": build_ok,
                       "hint": "" if build_ok else "Attach a build with processingState=VALID"})
    except Exception as e:
        checks.append({"check": "Build attached & processed", "ok": False,
                       "hint": f"Could not verify build: {e}"})

    try:
        enc_ok = False
        if st["build_id"]:
            bd = (await client.get(f"/v1/builds/{st['build_id']}")).get("data", {})
            enc_ok = bd.get("attributes", {}).get("usesNonExemptEncryption") is not None
        checks.append({"check": "Export-compliance / encryption flag set",
                       "ok": enc_ok, "hint": "" if enc_ok else "Set usesNonExemptEncryption on the build"})
    except Exception as e:
        checks.append({"check": "Export-compliance / encryption flag set", "ok": False,
                       "hint": f"Could not verify encryption flag: {e}"})

    try:
        shot_ok, shot_hint = await _primary_locale_screenshot_check(client, app_id, st["version_id"])
        checks.append({"check": "Screenshots present", "ok": shot_ok, "hint": shot_hint})
    except Exception as e:
        checks.append({"check": "Screenshots present", "ok": False,
                       "hint": f"Could not verify screenshots: {e}"})

    return checks


def blockers(checks: list[dict]) -> list[dict]:
    return [c for c in checks if not c["ok"]]
