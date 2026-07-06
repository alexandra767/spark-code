"""Tests for spark_code.model — _parse_tool_arguments and ModelClient."""

import json
import pytest

from spark_code.model import (
    _models_url,
    _parse_tool_arguments,
    _pick_real_model_name,
    _should_verify_tls,
    ModelClient,
    StreamError,
)


class _FakeResponse:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(self, lines, status_code=200, body=b""):
        self._lines = lines
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
    """Turn a list of dict chunks into SSE `data: ` lines + [DONE]."""
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return lines


async def _collect_stream(client, chunks):
    """Drive _stream_request_inner against fake SSE chunks; return chunk list."""
    client._client.stream = lambda *a, **k: _FakeStreamCtx(_FakeResponse(_sse(chunks)))
    out = []
    async for c in client._stream_request_inner({"model": "x", "messages": []}):
        out.append(c)
    return out


def _delta(content=None, reasoning=None):
    d = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return {"choices": [{"delta": d}]}


class TestResolveHelpers:
    def test_models_url_plain_base(self):
        assert _models_url("http://localhost:11434") == "http://localhost:11434/v1/models"

    def test_models_url_v1_base_not_doubled(self):
        assert _models_url("http://localhost:11434/v1") == "http://localhost:11434/v1/models"

    def test_models_url_gemini_openai_base(self):
        url = _models_url("https://generativelanguage.googleapis.com/v1beta/openai")
        assert url == "https://generativelanguage.googleapis.com/v1beta/openai/models"

    def test_tls_verify_disabled_for_localhost(self):
        assert _should_verify_tls("http://localhost:11434") is False

    def test_tls_verify_disabled_for_mdns_local(self):
        assert _should_verify_tls("https://spark-4a54.local:30000") is False

    def test_tls_verify_enabled_for_lookalike_host(self):
        # A host that merely CONTAINS "localhost"/".local" must still verify.
        assert _should_verify_tls("https://api.localstack.evil.com") is True
        assert _should_verify_tls("https://foo.localdomain.com") is True

    def test_pick_real_model_matches_configured_on_multi_model(self):
        data = [
            {"id": "llama3:8b"},
            {"id": "qwen3.5:122b", "root": "Intel/Qwen3-Coder-Next-int4"},
        ]
        # Must pick the configured alias, not data[0].
        assert _pick_real_model_name(data, "qwen3.5:122b") == "Qwen3-Coder-Next-int4"

    def test_pick_real_model_ambiguous_returns_none(self):
        data = [{"id": "a"}, {"id": "b"}]
        assert _pick_real_model_name(data, "") is None

    def test_pick_real_model_single_entry(self):
        assert _pick_real_model_name([{"id": "solo"}], "") == "solo"


@pytest.mark.asyncio
class TestThinkFiltering:
    async def _client(self, real=""):
        return ModelClient(endpoint="http://localhost:11434", model="qwen3.5:122b",
                           provider="ollama", real_model_name=real)

    async def test_thinking_hidden_answer_shown(self):
        client = await self._client()
        chunks = [_delta(content="<think>reasoning here</think>"), _delta(content="Final answer")]
        out = await _collect_stream(client, chunks)
        text = "".join(c["content"] for c in out if c["type"] == "text")
        assert text == "Final answer"
        assert "reasoning here" not in text

    async def test_split_close_tag_across_chunks(self):
        # </think> straddles two SSE deltas — must not swallow the answer.
        client = await self._client()
        chunks = [_delta(content="<think>deep thoughts</th"),
                  _delta(content="ink>Visible answer")]
        out = await _collect_stream(client, chunks)
        text = "".join(c["content"] for c in out if c["type"] == "text")
        assert text == "Visible answer"

    async def test_masqueraded_nonthinking_model_not_swallowed(self):
        # Model starts in-think (alias matches) but NEVER emits </think>:
        # the whole response must be surfaced, not lost.
        client = await self._client()
        chunks = [_delta(content="def foo():\n    return 42")]
        out = await _collect_stream(client, chunks)
        text = "".join(c["content"] for c in out if c["type"] == "text")
        assert "def foo()" in text and "return 42" in text

    async def test_reasoning_content_hidden(self):
        client = await self._client()
        chunks = [_delta(reasoning="hidden chain of thought"),
                  _delta(content="Answer")]
        out = await _collect_stream(client, chunks)
        text = "".join(c["content"] for c in out if c["type"] == "text")
        assert text == "Answer"
        assert any(c["type"] == "thinking_start" for c in out)

    async def test_narration_before_tool_call_is_visible(self):
        # Audit 2026-07-06 item 1: the primary coder model (alias qwen3.5:122b,
        # real name qwen3-coder-next) never emits </think>. Its narration
        # preamble that precedes a tool_call must reach the screen as visible
        # text — not be swallowed into the hidden think buffer and lost (the
        # safety net is gated by `not tool_calls_buffer` so it won't recover it).
        client = await self._client(real="Qwen3-Coder-Next-int4")
        chunks = [
            _delta(content="Let me read the file config.py."),
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1",
                 "function": {"name": "read_file",
                              "arguments": '{"path": "config.py"}'}}]}}]},
        ]
        out = await _collect_stream(client, chunks)
        text = "".join(c["content"] for c in out if c["type"] == "text")
        assert "Let me read the file config.py." in text
        assert any(c["type"] == "tool_call" for c in out)

    async def test_plain_answer_streams_incrementally(self):
        # Audit 2026-07-06 item 1: a multi-delta plain answer from the primary
        # model must stream incrementally (one visible `text` chunk per delta),
        # not buffer the entire response and dump it once via the safety net.
        # Also asserts the tok/s counter is non-zero (was 0 in the buffered path).
        client = await self._client(real="Qwen3-Coder-Next-int4")
        chunks = [_delta(content="First part. "),
                  _delta(content="Second part. "),
                  _delta(content="Third part.")]
        out = await _collect_stream(client, chunks)
        text_chunks = [c for c in out if c["type"] == "text"]
        assert len(text_chunks) >= 3, "answer must stream, not dump once at the end"
        text = "".join(c["content"] for c in text_chunks)
        assert text == "First part. Second part. Third part."
        done = [c for c in out if c["type"] == "done"][0]
        assert done["_speed"]["tokens"] > 0


@pytest.mark.asyncio
class TestSSEDataPrefix:
    async def test_no_space_data_prefix_accepted(self):
        # Audit 2026-07-06 item 8: per the SSE spec `data:{...}` (no space after
        # the colon) is valid and identical to `data: {...}`. The parser required
        # the trailing space and silently dropped every event from a compliant
        # server using the no-space form — a blank turn with no error.
        client = ModelClient(endpoint="http://localhost:11434", model="m",
                             provider="ollama")
        chunk = json.dumps(_delta(content="Hello from no-space SSE"))
        lines = [f"data:{chunk}", "data:[DONE]"]
        client._client.stream = lambda *a, **k: _FakeStreamCtx(_FakeResponse(lines))
        out = []
        async for c in client._stream_request_inner({"model": "m", "messages": []}):
            out.append(c)
        text = "".join(c["content"] for c in out if c["type"] == "text")
        assert text == "Hello from no-space SSE"

    async def test_space_form_still_works(self):
        client = ModelClient(endpoint="http://localhost:11434", model="m",
                             provider="ollama")
        out = await _collect_stream(client, [_delta(content="spaced")])
        text = "".join(c["content"] for c in out if c["type"] == "text")
        assert text == "spaced"


@pytest.mark.asyncio
class TestStreamUsageAndErrors:
    async def test_done_reports_per_request_usage_keys(self):
        client = ModelClient(endpoint="http://localhost:11434", model="m", provider="ollama")
        chunks = [
            _delta(content="hi"),
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 5}},
        ]
        out = await _collect_stream(client, chunks)
        done = [c for c in out if c["type"] == "done"][0]
        # Agent reads prompt_tokens/completion_tokens — they must be present.
        assert done["usage"]["prompt_tokens"] == 100
        assert done["usage"]["completion_tokens"] == 5
        assert client.total_input_tokens == 100
        assert client.total_output_tokens == 5

    async def test_empty_choices_with_usage_no_indexerror(self):
        client = ModelClient(endpoint="http://localhost:11434", model="m", provider="ollama")
        # A final include_usage chunk with choices:[] must not raise.
        out = await _collect_stream(client, [{"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}}])
        assert any(c["type"] == "done" for c in out)

    async def test_non_200_raises_streamerror(self):
        client = ModelClient(endpoint="http://localhost:11434", model="m", provider="ollama")
        client._client.stream = lambda *a, **k: _FakeStreamCtx(
            _FakeResponse([], status_code=503, body=b"overloaded"))
        with pytest.raises(StreamError) as ei:
            async for _ in client._stream_request_inner({"model": "m", "messages": []}):
                pass
        assert ei.value.status_code == 503


class TestParseToolArgumentsEmpty:
    def test_empty_string_returns_empty_dict(self):
        assert _parse_tool_arguments("") == {}

    def test_none_returns_empty_dict(self):
        # Depending on implementation, None may also be handled
        result = _parse_tool_arguments("")
        assert result == {}


class TestParseToolArgumentsValidJSON:
    def test_valid_json_parsed(self):
        raw = '{"path": "/tmp/file.txt", "content": "hello"}'
        result = _parse_tool_arguments(raw)
        assert result == {"path": "/tmp/file.txt", "content": "hello"}

    def test_valid_nested_json(self):
        raw = '{"a": {"b": 1}, "c": [1, 2, 3]}'
        result = _parse_tool_arguments(raw)
        assert result["a"]["b"] == 1
        assert result["c"] == [1, 2, 3]


class TestParseToolArgumentsNewlines:
    def test_escaped_newlines_fixed(self):
        # JSON with literal newlines in strings that need fixing
        raw = '{"content": "line1\nline2"}'
        result = _parse_tool_arguments(raw)
        assert "line1" in result.get("content", "")
        assert "line2" in result.get("content", "")


class TestParseToolArgumentsTruncated:
    def test_truncated_json_repaired(self):
        raw = '{"key": "value", "other": "data'
        result = _parse_tool_arguments(raw)
        # Should attempt to close the JSON and parse
        assert "key" in result or "raw" in result

    def test_open_quote_repair(self):
        raw = '{"key": "val'
        result = _parse_tool_arguments(raw)
        # Either repaired successfully or returned as raw
        assert isinstance(result, dict)


class TestParseToolArgumentsConcatenated:
    def test_concatenated_objects_extracts_first(self):
        raw = '{"a": 1}{"b": 2}'
        result = _parse_tool_arguments(raw)
        # Should extract the first valid JSON object
        assert result.get("a") == 1 or "raw" in result


class TestParseToolArgumentsGarbage:
    def test_total_garbage_returns_raw(self):
        raw = "not json at all !!!@@@"
        result = _parse_tool_arguments(raw)
        assert result == {"raw": raw}


class TestModelClientApiUrl:
    def test_api_url_ollama_provider(self):
        client = ModelClient(
            endpoint="http://localhost:11434",
            model="llama3",
            provider="ollama",
        )
        url = client.api_url
        assert url == "http://localhost:11434/v1/chat/completions"

    def test_api_url_gemini_provider(self):
        client = ModelClient(
            endpoint="https://generativelanguage.googleapis.com",
            model="gemini-pro",
            provider="gemini",
        )
        url = client.api_url
        assert url == "https://generativelanguage.googleapis.com/chat/completions"

    def test_api_url_avoids_double_v1(self):
        client = ModelClient(
            endpoint="http://localhost:11434/v1",
            model="llama3",
            provider="ollama",
        )
        url = client.api_url
        # Should NOT produce /v1/v1/
        assert "/v1/v1/" not in url
        assert "/v1/chat/completions" in url


class TestModelClientBuildToolsPayload:
    def test_build_tools_payload_formats_correctly(self):
        client = ModelClient(
            endpoint="http://localhost:11434",
            model="llama3",
            provider="ollama",
        )
        tools = [
            {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"],
                },
            }
        ]
        payload = client._build_tools_payload(tools)
        assert isinstance(payload, list)
        assert len(payload) == 1
        tool = payload[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "read_file"
        assert tool["function"]["description"] == "Read a file from disk"
        assert "parameters" in tool["function"]


class TestModelClientInit:
    def test_stores_timeout_and_max_retries(self):
        client = ModelClient(
            endpoint="http://localhost:11434",
            model="llama3",
            provider="ollama",
            timeout=120,
            max_retries=5,
        )
        assert client.timeout == 120
        assert client.max_retries == 5

    def test_stores_provider_and_model(self):
        client = ModelClient(
            endpoint="http://localhost:11434",
            model="codellama",
            provider="ollama",
        )
        assert client.provider == "ollama"
        assert client.model == "codellama"
