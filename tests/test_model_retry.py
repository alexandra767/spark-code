"""Regression tests for ModelClient._stream_request — the retry/overflow wrapper.

Covers the findings fixed on 2026-07-10:
  #2 precise overflow-marker matching (unrelated 400s echoing schema field
     names like "input_tokens" must NOT be hijacked into a silent halve+retry);
  #3 deterministic max_tokens halving on its OWN bounded budget — every
     computed halving is actually SENT (no off-by-one), floor respected, and
     the overflow budget is independent of the transient 429/5xx budget;
  #4 no exponential-backoff sleep for the deterministic overflow retries;
  #6 the wrapper itself is exercised (previously only _stream_request_inner
     was), including transient-500-then-success and prompt-side overflow
     terminating with a recoverable error instead of looping.
"""

import json

import pytest

from spark_code.model import (
    _MAX_OVERFLOW_HALVINGS,
    _OVERFLOW_MIN_TOKENS,
    ModelClient,
    _looks_like_context_overflow,
)


class _FakeResponse:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(self, lines=None, status_code=200, body=b""):
        self._lines = lines or []
        self.status_code = status_code
        self._body = body

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self):
        return self._body


class _FakeStreamCtx:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *a):
        return False


def _sse(chunks):
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return lines


def _ok_response(text="hi"):
    return _FakeResponse(_sse([
        {"choices": [{"delta": {"content": text}}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]))


def _err_response(status, body_text):
    return _FakeResponse(status_code=status, body=body_text.encode())


def _seq_stream(client, responses):
    """Route successive client._client.stream() calls to `responses` in order."""
    it = iter(responses)

    def _stream(*a, **k):
        return _FakeStreamCtx(next(it))

    client._client.stream = _stream


def _client(max_retries=3):
    return ModelClient(endpoint="http://localhost:11434", model="qwen3.5:122b",
                       provider="ollama", max_retries=max_retries)


def _sleep_recorder():
    """Return (calls_list, async_sleep) — patch spark_code.model.asyncio.sleep
    with the second to record every backoff delay without actually sleeping."""
    calls: list = []

    async def _sleep(d):
        calls.append(d)

    return calls, _sleep


async def _run(client, payload):
    out = []
    async for c in client._stream_request(payload):
        out.append(c)
    return out


# --------------------------------------------------------------------------
# #2 — precise overflow markers
# --------------------------------------------------------------------------

def test_looks_like_overflow_matches_only_precise_phrases():
    # Real vLLM/OpenAI overflow fragments — must match.
    assert _looks_like_context_overflow(
        "This model's maximum context length is 8192 tokens")
    assert _looks_like_context_overflow(
        "Please reduce the length of the messages or completion")
    assert _looks_like_context_overflow(
        "prompt is longer than the maximum model length of 4096")
    assert _looks_like_context_overflow("max_model_len (32768) exceeded")
    # Generic field/limit mentions that used to hijack unrelated 400s — must NOT.
    assert not _looks_like_context_overflow(
        "Invalid value for field input_tokens: expected integer")
    assert not _looks_like_context_overflow("the context length parameter is bad")


@pytest.mark.asyncio
async def test_400_echoing_input_tokens_not_treated_as_overflow(monkeypatch):
    calls, fake_sleep = _sleep_recorder()
    monkeypatch.setattr("spark_code.model.asyncio.sleep", fake_sleep)
    client = _client()
    sent = []

    def _stream(*a, **k):
        sent.append(k["json"]["max_tokens"])
        return _FakeStreamCtx(_err_response(
            400, "Invalid value for field input_tokens: expected integer"))

    client._client.stream = _stream
    payload = {"model": "x", "messages": [], "max_tokens": 4096}
    out = await _run(client, payload)

    # Immediate recoverable error — no halving, no retry, no backoff.
    assert len(sent) == 1
    assert payload["max_tokens"] == 4096  # untouched
    assert calls == []
    errs = [c for c in out if c["type"] == "error"]
    assert errs and errs[0]["recoverable"] is True
    assert "input_tokens" in errs[0]["content"]


# --------------------------------------------------------------------------
# #3 / #4 / #6 — overflow halving budget, off-by-one, no sleep
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_overflow_halves_then_resends_and_succeeds(monkeypatch):
    calls, fake_sleep = _sleep_recorder()
    monkeypatch.setattr("spark_code.model.asyncio.sleep", fake_sleep)
    client = _client()
    _seq_stream(client, [
        _err_response(400, "This model's maximum context length is 32768 tokens."),
        _ok_response("done"),
    ])
    payload = {"model": "x", "messages": [], "max_tokens": 4096}
    out = await _run(client, payload)

    text = "".join(c["content"] for c in out if c["type"] == "text")
    assert text == "done"
    assert payload["max_tokens"] == 2048   # halved exactly once before success
    assert calls == []                     # #4: no backoff for deterministic retry
    assert not any(c["type"] == "error" for c in out)


@pytest.mark.asyncio
async def test_every_computed_halving_is_actually_sent(monkeypatch):
    """#3 off-by-one: 4096→2048→1024→512, where 512 fits. The old code computed
    512, wrote it into the payload, then exited the loop without ever sending
    it (overflow retries shared the 3-attempt transient budget)."""
    calls, fake_sleep = _sleep_recorder()
    monkeypatch.setattr("spark_code.model.asyncio.sleep", fake_sleep)
    client = _client(max_retries=3)
    sent = []

    def _stream(*a, **k):
        mt = k["json"]["max_tokens"]
        sent.append(mt)
        # Server accepts only once the completion budget is halved to <= 512.
        if mt > 512:
            return _FakeStreamCtx(_err_response(400, "maximum context length"))
        return _FakeStreamCtx(_ok_response("fit"))

    client._client.stream = _stream
    payload = {"model": "x", "messages": [], "max_tokens": 4096}
    out = await _run(client, payload)

    text = "".join(c["content"] for c in out if c["type"] == "text")
    assert text == "fit"
    assert sent == [4096, 2048, 1024, 512]  # the 512 attempt is actually sent
    assert payload["max_tokens"] == 512
    assert calls == []                       # still no backoff
    assert not any(c["type"] == "error" for c in out)


@pytest.mark.asyncio
async def test_prompt_side_overflow_terminates_at_floor(monkeypatch):
    """#6: when even the smallest completion still overflows (the PROMPT itself
    doesn't fit), halving can't help — terminate with a recoverable error at the
    floor rather than looping forever."""
    calls, fake_sleep = _sleep_recorder()
    monkeypatch.setattr("spark_code.model.asyncio.sleep", fake_sleep)
    client = _client()
    sent = []

    def _stream(*a, **k):
        sent.append(k["json"]["max_tokens"])
        return _FakeStreamCtx(_err_response(400, "maximum context length"))

    client._client.stream = _stream
    payload = {"model": "x", "messages": [], "max_tokens": 4096}
    out = await _run(client, payload)

    errs = [c for c in out if c["type"] == "error"]
    assert errs and errs[-1]["recoverable"] is True
    assert min(sent) >= _OVERFLOW_MIN_TOKENS           # never sent below the floor
    assert sent[-1] == _OVERFLOW_MIN_TOKENS            # floor was actually tried
    assert len(sent) <= _MAX_OVERFLOW_HALVINGS + 1     # bounded, no infinite loop
    assert calls == []                                 # no backoff on overflow path


# --------------------------------------------------------------------------
# #6 — transient retries still work and are NOT confused with overflow
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transient_500_then_success(monkeypatch):
    calls, fake_sleep = _sleep_recorder()
    monkeypatch.setattr("spark_code.model.asyncio.sleep", fake_sleep)
    client = _client()
    _seq_stream(client, [
        _err_response(500, "internal server error"),
        _ok_response("recovered"),
    ])
    out = await _run(client, {"model": "x", "messages": [], "max_tokens": 4096})

    text = "".join(c["content"] for c in out if c["type"] == "text")
    assert text == "recovered"
    assert calls == [1]   # transient path DID back off (1s), unlike overflow
    assert not any(c["type"] == "error" for c in out)


@pytest.mark.asyncio
async def test_overflow_marker_on_non_400_is_transient_not_halved(monkeypatch):
    """#6: an overflow *phrase* in a non-400 body (e.g. a 503 error page) must
    NOT trigger halving — only a real 400 does. It's a transient error: retried
    with backoff, max_tokens untouched."""
    calls, fake_sleep = _sleep_recorder()
    monkeypatch.setattr("spark_code.model.asyncio.sleep", fake_sleep)
    client = _client(max_retries=2)
    sent = []

    def _stream(*a, **k):
        sent.append(k["json"]["max_tokens"])
        return _FakeStreamCtx(_err_response(503, "maximum context length"))

    client._client.stream = _stream
    payload = {"model": "x", "messages": [], "max_tokens": 4096}
    out = await _run(client, payload)

    assert all(mt == 4096 for mt in sent)   # never halved — status wasn't 400
    assert len(sent) == 2                    # exhausted max_retries=2
    assert calls == [1]                      # one 1s backoff between the 2 tries
    errs = [c for c in out if c["type"] == "error"]
    assert errs and errs[-1]["recoverable"] is True
    assert "after 2 retries" in errs[-1]["content"]
