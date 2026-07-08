from spark_code.tools.appstore_status import AppStoreStatusTool
from spark_code.tools.appstore_submit import AppStoreSubmitTool


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


def _client_monkeypatch(monkeypatch, ready=True, calls=None):
    async def fake_resolve(self, ident): return "APP1"
    async def fake_state(self, app_id):
        return {"version_id":"V1","version":"1.3.22","state":"PREPARE_FOR_SUBMISSION",
                "build_id":"B","build_number":"55","build_processing_state":"VALID"}
    async def fake_ready(client, app_id):
        return [{"check":"Build attached & processed","ok":ready,"hint":"" if ready else "attach build"}]
    async def fake_post(self, path, body):
        if calls is not None:
            calls.append(("POST", path))
        return {"data":{"id":"SID1"}}
    async def fake_patch(self, path, body):
        if calls is not None:
            calls.append(("PATCH", path))
        return {"data":{"id":"SID1","attributes":{"state":"WAITING_FOR_REVIEW"}}}
    monkeypatch.setattr("spark_code.appstore.AscClient.resolve_app", fake_resolve)
    monkeypatch.setattr("spark_code.appstore.AscClient.get_submission_state", fake_state)
    monkeypatch.setattr("spark_code.tools.appstore_submit.readiness", fake_ready)
    monkeypatch.setattr("spark_code.appstore.AscClient.post", fake_post)
    monkeypatch.setattr("spark_code.appstore.AscClient.patch", fake_patch)


async def test_submit_refuses_when_not_ready(monkeypatch):
    calls = []
    _client_monkeypatch(monkeypatch, ready=False, calls=calls)
    out = await AppStoreSubmitTool().execute(app="com.x", version="1.3.22", confirm="submit")
    assert "not ready" in out.lower() and calls == []  # never hit the API


async def test_submit_without_confirm_shows_summary_only(monkeypatch):
    calls = []
    _client_monkeypatch(monkeypatch, ready=True, calls=calls)
    out = await AppStoreSubmitTool().execute(app="com.x", version="1.3.22")
    assert "1.3.22" in out and "confirm" in out.lower() and calls == []


async def test_submit_confirmed_runs_sequence_in_order(monkeypatch):
    calls = []
    _client_monkeypatch(monkeypatch, ready=True, calls=calls)
    out = await AppStoreSubmitTool().execute(app="com.x", version="1.3.22", confirm="submit")
    assert [m for m,_ in calls] == ["POST","POST","PATCH"]
    assert "reviewSubmissions" in calls[0][1] and "reviewSubmissionItems" in calls[1][1]
    assert "submitted" in out.lower() or "waiting_for_review" in out.lower()


async def test_headless_refuses_without_confirm(monkeypatch):
    calls = []
    _client_monkeypatch(monkeypatch, ready=True, calls=calls)
    out = await AppStoreSubmitTool(interactive=False).execute(app="com.x", version="1.3.22")
    assert "confirm=" in out and calls == []
