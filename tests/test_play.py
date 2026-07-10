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


def test_malformed_sa_json_raises_playerror_not_traceback(tmp_path):
    # Fix 10: a corrupt service-account.json must raise PlayError (which
    # callers already catch), not leak a raw JSONDecodeError traceback.
    bad = tmp_path / "sa.json"
    bad.write_text("{ this is : not valid json ,,,")
    with pytest.raises(PlayError) as ei:
        PlayClient(sa_path=str(bad))
    assert "malformed" in str(ei.value).lower() or "unreadable" in str(ei.value).lower()


async def test_upload_bundle_offloads_read_to_thread(tmp_path, monkeypatch):
    # Fix 9: the (20-150MB) AAB read must be offloaded to a worker thread so it
    # doesn't block the event loop; the bytes still reach the upload.
    import spark_code.play as play_mod

    c = PlayClient(sa_path=_sa_file(tmp_path))

    async def fake_token():
        return "tok123"
    monkeypatch.setattr(c, "_token", fake_token)

    aab = tmp_path / "app.aab"
    aab.write_bytes(b"AAB-BYTES")

    seen = {}
    real_to_thread = play_mod.asyncio.to_thread

    async def spy_to_thread(fn, *a, **kw):
        seen["offloaded"] = True
        return await real_to_thread(fn, *a, **kw)
    monkeypatch.setattr(play_mod.asyncio, "to_thread", spy_to_thread)

    captured = {}

    class FakeResp:
        status_code = 200
        text = '{"versionCode": 77}'

        def json(self):
            return {"versionCode": 77}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, content=None):
            captured["content"] = content
            return FakeResp()
    monkeypatch.setattr(play_mod.httpx, "AsyncClient", FakeClient)

    vc = await c.upload_bundle("com.x.app", "EDIT1", str(aab))
    assert vc == 77
    assert seen.get("offloaded") is True
    assert captured["content"] == b"AAB-BYTES"
