"""Tests for WebFetchTool — focus on the SSRF guard."""

import pytest

from spark_code.tools.web_fetch import WebFetchTool, _validate_url


def _fake_getaddrinfo(ip: str):
    async def _inner(host, port, **kw):
        return [(2, 1, 6, "", (ip, 0))]
    return _inner


@pytest.fixture
def tool():
    return WebFetchTool()


async def test_validate_blocks_loopback(monkeypatch):
    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(_fake_getaddrinfo("127.0.0.1"))})(),
    )
    err = await _validate_url("http://localhost:8010/stats")
    assert err and "SSRF" in err


async def test_validate_blocks_metadata_endpoint(monkeypatch):
    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(_fake_getaddrinfo("169.254.169.254"))})(),
    )
    err = await _validate_url("http://metadata.example/latest/meta-data/")
    assert err and "SSRF" in err


async def test_validate_blocks_private(monkeypatch):
    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(_fake_getaddrinfo("192.168.1.188"))})(),
    )
    assert await _validate_url("http://router.local/") is not None


async def test_validate_blocks_tailnet_cgnat(monkeypatch):
    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(_fake_getaddrinfo("100.100.1.1"))})(),
    )
    assert await _validate_url("http://spark-4a54.tailcade53.ts.net:4200/") is not None


async def test_validate_allows_public(monkeypatch):
    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(_fake_getaddrinfo("93.184.216.34"))})(),
    )
    assert await _validate_url("https://example.com/docs") is None


async def test_validate_rejects_non_http_scheme():
    assert "scheme" in (await _validate_url("file:///etc/passwd") or "")
    assert "scheme" in (await _validate_url("gopher://x/") or "")


async def test_validate_rejects_unresolvable(monkeypatch):
    async def _boom(host, port, **kw):
        raise OSError("no such host")
    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(_boom)})(),
    )
    assert await _validate_url("http://nonexistent.invalid/") is not None


async def test_execute_refuses_internal(tool, monkeypatch):
    monkeypatch.setattr(
        "asyncio.get_event_loop",
        lambda: type("L", (), {"getaddrinfo": staticmethod(_fake_getaddrinfo("127.0.0.1"))})(),
    )
    result = await tool.execute(url="http://localhost:8010/stats")
    assert "SSRF" in result or "private/internal" in result
