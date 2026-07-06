import httpx
import pytest

from spark_code.preflight import context_window_warning, fetch_server_max_context


def _transport(payload, status=200):
    def handler(request):
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_fetch_returns_max_model_len_for_matching_model():
    payload = {"data": [{"id": "qwen3.5:122b", "max_model_len": 32768}]}
    got = await fetch_server_max_context(
        "http://x:30000", "qwen3.5:122b", transport=_transport(payload))
    assert got == 32768


@pytest.mark.asyncio
async def test_fetch_returns_none_when_field_missing():
    # Ollama's OpenAI-compat /v1/models has no max_model_len
    payload = {"data": [{"id": "qwen3-coder:30b"}]}
    got = await fetch_server_max_context(
        "http://x:11434", "qwen3-coder:30b", transport=_transport(payload))
    assert got is None


@pytest.mark.asyncio
async def test_fetch_returns_none_when_model_absent():
    payload = {"data": [{"id": "other-model", "max_model_len": 4096}]}
    got = await fetch_server_max_context(
        "http://x:30000", "qwen3.5:122b", transport=_transport(payload))
    assert got is None


@pytest.mark.asyncio
async def test_fetch_returns_none_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("boom")
    got = await fetch_server_max_context(
        "http://x:30000", "m", transport=httpx.MockTransport(handler))
    assert got is None


@pytest.mark.asyncio
async def test_fetch_returns_none_on_http_error():
    got = await fetch_server_max_context(
        "http://x:30000", "m", transport=_transport({}, status=500))
    assert got is None


def test_warning_when_configured_below_server():
    msg = context_window_warning(16384, 32768)
    assert msg is not None
    assert "16384" in msg and "32768" in msg


def test_warning_when_configured_above_server():
    msg = context_window_warning(65536, 32768)
    assert msg is not None
    assert "exceed" in msg.lower()


def test_no_warning_on_match():
    assert context_window_warning(32768, 32768) is None


def test_no_warning_when_server_unknown():
    assert context_window_warning(32768, None) is None
