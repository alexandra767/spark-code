import json

import httpx as _httpx
import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization

# minimal ES256 key for signing tests
from cryptography.hazmat.primitives.asymmetric import ec

from spark_code.appstore import AscClient, AscError


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


# --- FIX 1: percent-encode identifier in resolve_app ---

async def test_resolve_app_encodes_special_characters(tmp_path, monkeypatch):
    _write_config(tmp_path)
    c = AscClient(config_dir=str(tmp_path))
    captured = []
    async def fake(method, path, body=None):
        captured.append(path)
        return {"data": [{"id": "APP1"}]}
    monkeypatch.setattr(c, "_req", fake)
    assert await c.resolve_app("App & Co #1") == "APP1"

    path = captured[0]
    assert "limit=2" in path
    # the only literal "&" left must be the "&limit=2" separator — the
    # identifier's own "&" (and "#", which would otherwise truncate the
    # query as a URL fragment) must be percent-encoded away.
    assert path.count("&") == 1
    assert "#" not in path
    assert "%26" in path  # encoded "&"
    assert "%23" in path  # encoded "#"


# --- FIX 3: clean AscError (not KeyError) on malformed config ---

def test_missing_key_file_raises_ascerror_not_keyerror(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "key_id": "KID123", "issuer_id": "ISS-abc"}))
    c = AscClient(config_dir=str(tmp_path))
    with pytest.raises(AscError):
        c._token()


def test_missing_issuer_id_raises_ascerror_not_keyerror(tmp_path):
    (tmp_path / "AuthKey.p8").write_bytes(b"not-a-real-key")
    (tmp_path / "config.json").write_text(json.dumps({
        "key_id": "KID123", "key_file": "AuthKey.p8"}))
    c = AscClient(config_dir=str(tmp_path))
    with pytest.raises(AscError):
        c._token()


# --- ADD COVERAGE: _req error handling ---

def _fake_async_client_cls(response):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False
        async def request(self, method, url, headers=None, content=None):
            return response
    return FakeAsyncClient


class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text
    def json(self):
        return json.loads(self.text) if self.text else {}


async def test_req_raises_ascerror_on_4xx(monkeypatch):
    c = AscClient(config_dir="/nonexistent")
    monkeypatch.setattr(c, "_token", lambda: "tok")
    monkeypatch.setattr(_httpx, "AsyncClient", _fake_async_client_cls(_FakeResponse(400, "bad request")))
    with pytest.raises(AscError):
        await c._req("GET", "/v1/apps")


async def test_req_raises_ascerror_on_429_mentions_rate_limit(monkeypatch):
    c = AscClient(config_dir="/nonexistent")
    monkeypatch.setattr(c, "_token", lambda: "tok")
    monkeypatch.setattr(_httpx, "AsyncClient", _fake_async_client_cls(_FakeResponse(429, "slow down")))
    with pytest.raises(AscError, match="rate limit"):
        await c._req("GET", "/v1/apps")


# --- ADD COVERAGE: get_submission_state with no attached build ---

async def test_get_submission_state_no_build(tmp_path, monkeypatch):
    _write_config(tmp_path)
    c = AscClient(config_dir=str(tmp_path))

    async def fake_get(path):
        if "appStoreVersions?limit=1" in path:
            return {"data": [{
                "id": "V1",
                "attributes": {"appStoreState": "READY_FOR_SALE", "versionString": "1.0"},
            }]}
        if path.endswith("/build"):
            return {"data": None}
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(c, "get", fake_get)
    result = await c.get_submission_state("APP1")
    assert result == {
        "version_id": "V1", "version": "1.0", "state": "READY_FOR_SALE",
        "build_id": None, "build_number": None, "build_processing_state": None,
    }


async def test_get_submission_state_build_fetch_raises_ascerror_is_swallowed(tmp_path, monkeypatch):
    _write_config(tmp_path)
    c = AscClient(config_dir=str(tmp_path))

    async def fake_get(path):
        if "appStoreVersions?limit=1" in path:
            return {"data": [{
                "id": "V1",
                "attributes": {"appStoreState": "PREPARE_FOR_SUBMISSION", "versionString": "2.0"},
            }]}
        if path.endswith("/build"):
            raise AscError("ASC GET /v1/appStoreVersions/V1/build → 404: not found")
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(c, "get", fake_get)
    result = await c.get_submission_state("APP1")
    assert result == {
        "version_id": "V1", "version": "2.0", "state": "PREPARE_FOR_SUBMISSION",
        "build_id": None, "build_number": None, "build_processing_state": None,
    }
