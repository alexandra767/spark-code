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
