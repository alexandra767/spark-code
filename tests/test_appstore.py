import json

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
