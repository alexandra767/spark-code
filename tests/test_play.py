import json

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

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
