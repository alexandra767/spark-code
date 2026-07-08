from spark_code.tools.appstore_status import AppStoreStatusTool


async def test_status_reports_state_and_readiness(monkeypatch):
    t = AppStoreStatusTool()
    async def fake_resolve(self, ident): return "APP1"
    async def fake_state(self, app_id):
        return {"version_id":"V","version":"1.3.22","state":"PREPARE_FOR_SUBMISSION",
                "build_id":"B","build_number":"55","build_processing_state":"VALID"}
    async def fake_ready(client, app_id):
        return [{"check":"Build attached & processed","ok":True,"hint":""},
                {"check":"Screenshots present","ok":False,"hint":"Upload required screenshots"}]
    monkeypatch.setattr("spark_code.appstore.AscClient.resolve_app", fake_resolve)
    monkeypatch.setattr("spark_code.appstore.AscClient.get_submission_state", fake_state)
    monkeypatch.setattr("spark_code.tools.appstore_status.readiness", fake_ready)
    out = await t.execute(app="com.gigledger")
    assert "1.3.22" in out and "PREPARE_FOR_SUBMISSION" in out
    assert "55" in out and "Screenshots present" in out and "Upload required screenshots" in out
    assert t.is_read_only is True and t.requires_permission is False
