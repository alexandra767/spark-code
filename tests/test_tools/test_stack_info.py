"""Unit tests for stack_info.py: reads Orrery's peer API (Task 9). All HTTP
is mocked — no real Orrery service is required to run this suite."""

import httpx

from spark_code.tools.stack_info import StackInfoTool

MODEL_NODES = [
    {"id": "app:claude-ui", "label": "claude-ui", "ring": "applications", "kind": "assistant"},
    {"id": "routine:second-brain-catchup", "label": "second-brain-catchup", "ring": "routines",
     "kind": "systemd", "schedule": "every 10 min", "last_run": "2026-07-10T09:55:30-04:00",
     "last_result": "success", "next_run": "2026-07-10T10:05:30-04:00"},
    {"id": "routine:revdash-funnel", "label": "revdash-funnel", "ring": "routines",
     "kind": "systemd", "schedule": "every 5 min", "last_result": "success"},
    {"id": "mem:rag-second_brain", "label": "second_brain", "ring": "memory", "kind": "rag", "count": 617},
    {"id": "skill:execute-code", "label": "execute_code", "ring": "skills", "family": "core"},
]


class _FakeResp:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


async def test_default_summary(monkeypatch):
    async def fake_get(self, url, params=None):
        assert url.endswith("/api/stack/summary")
        return _FakeResp({
            "counts": {"applications": 16, "routines": 28, "memory": 72, "skills": 78},
            "generated": {"spark": "2026-07-10T14:00:00+00:00"},
            "warnings": [],
        })

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await StackInfoTool().execute()

    assert "applications: 16" in result
    assert "routines: 28" in result
    assert "warnings: none" in result


async def test_section_routines_filters_by_ring(monkeypatch):
    async def fake_get(self, url, params=None):
        assert url.endswith("/api/stack/model")
        return _FakeResp({"nodes": MODEL_NODES})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await StackInfoTool().execute(section="routines")

    assert "routines (2)" in result
    assert "second-brain-catchup" in result
    assert "revdash-funnel" in result
    # Cross-ring leakage would be a real bug — a skills/memory/app name here
    # means the ring filter isn't actually filtering.
    assert "execute_code" not in result
    assert "second_brain" not in result
    assert "claude-ui" not in result
    # Freshness stamps present for a routine.
    assert "last_run=" in result
    assert "schedule=every 10 min" in result


async def test_section_and_query_narrows_further(monkeypatch):
    async def fake_get(self, url, params=None):
        return _FakeResp({"nodes": MODEL_NODES})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await StackInfoTool().execute(section="routines", query="revdash")

    assert "revdash-funnel" in result
    assert "second-brain-catchup" not in result


async def test_query_general_search(monkeypatch):
    async def fake_get(self, url, params=None):
        assert url.endswith("/api/stack/search")
        assert params == {"q": "vault"}
        return _FakeResp({"nodes": [
            {"id": "mem:vault", "label": "SecondBrain vault", "ring": "memory", "kind": "vault", "count": 617},
        ]})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await StackInfoTool().execute(query="vault")

    assert "Stack matches for 'vault' (1)" in result
    assert "SecondBrain vault" in result
    assert "File matches" not in result  # "vault" alone isn't file-ish


async def test_file_ish_query_fans_out_to_file_search(monkeypatch):
    calls = []

    async def fake_get(self, url, params=None):
        calls.append(url)
        if url.endswith("/api/stack/search"):
            return _FakeResp({"nodes": []})
        if url.endswith("/api/files/search"):
            return _FakeResp({
                "hits": [{"host": "mac", "root": "/Users/x", "path": "/Users/x/budget.xlsx"}],
                "truncated": False,
            })
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await StackInfoTool().execute(query="budget.xlsx")

    assert any(u.endswith("/api/files/search") for u in calls)
    assert "File matches" in result
    assert "budget.xlsx" in result


async def test_unknown_section_is_friendly_not_a_crash():
    result = await StackInfoTool().execute(section="bogus")
    assert "Unknown section" in result


async def test_orrery_unreachable_is_friendly_never_raises(monkeypatch):
    async def fake_get(self, url, params=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    result = await StackInfoTool().execute()

    assert "Orrery is not running" in result
