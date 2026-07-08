import json

import pytest

from spark_code.play import TESTING_TRACKS, PlayClient, PlayError


def _sa_file(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
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
        if path.endswith("/edits"):
            return {"id": "E9"}
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
    aab = tmp_path / "app.aab"
    aab.write_bytes(b"AABDATA")

    async def fake_token():
        return "TOK"
    monkeypatch.setattr(c, "_token", fake_token)
    seen = {}

    class FakeResp:
        status_code = 200
        text = '{"versionCode": 46}'
        def json(self):
            return {"versionCode": 46}

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, headers=None, content=None):
            seen["url"] = url
            seen["headers"] = headers
            seen["content"] = content
            return FakeResp()
    monkeypatch.setattr("spark_code.play.httpx.AsyncClient", FakeClient)
    vc = await c.upload_bundle("com.x", "E9", str(aab))
    assert vc == 46
    assert "uploadType=media" in seen["url"] and "/upload/androidpublisher/" in seen["url"]
    assert seen["headers"]["Content-Type"] == "application/octet-stream"
    assert seen["content"] == b"AABDATA"

async def test_upload_bundle_raises_on_error(tmp_path, monkeypatch):
    c = PlayClient(sa_path=_sa_file(tmp_path))
    aab = tmp_path / "app.aab"
    aab.write_bytes(b"X")

    async def fake_token():
        return "TOK"
    monkeypatch.setattr(c, "_token", fake_token)

    class FakeResp:
        status_code = 403
        text = '{"error":"duplicate versionCode"}'
        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, url, headers=None, content=None):
            return FakeResp()
    monkeypatch.setattr("spark_code.play.httpx.AsyncClient", FakeClient)
    with pytest.raises(PlayError):
        await c.upload_bundle("com.x", "E9", str(aab))
