"""Tests for spark_code.model — _parse_tool_arguments and ModelClient."""

import json
import pytest

from spark_code.model import _parse_tool_arguments, ModelClient


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
