"""OpenAI-compatible API client with streaming and tool calling.

Supports multiple providers:
- Ollama (local, DGX Spark)
- Google Gemini (via OpenAI-compatible API)
- OpenAI
- Any OpenAI-compatible endpoint
"""

import json
import httpx
from typing import AsyncIterator


def _parse_tool_arguments(raw: str) -> dict:
    """Parse tool call arguments with fallback repairs for malformed JSON."""
    if not raw:
        return {}

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try escaping literal newlines (common with Gemini streaming)
    try:
        fixed = raw.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object (Gemini sometimes concatenates multiple)
    if raw.startswith("{"):
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(raw):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[:i + 1])
                    except json.JSONDecodeError:
                        break

    # Try fixing truncated JSON — add missing closing braces
    attempt = raw.rstrip()
    for _ in range(3):
        attempt += "}"
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

    # Try fixing truncated string values — close open quotes then braces
    attempt = raw.rstrip()
    if attempt.count('"') % 2 == 1:
        attempt += '"}'
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    # Last resort — return raw string in a dict
    return {"raw": raw}


# Known provider configurations
PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "api_path": "/v1/chat/completions",
        "needs_key": False,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_path": "/chat/completions",
        "needs_key": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_path": "/chat/completions",
        "needs_key": True,
    },
}


class ModelClient:
    """Client for OpenAI-compatible chat APIs (Ollama, Gemini, OpenAI, etc.)."""

    def __init__(self, endpoint: str, model: str, temperature: float = 0.7,
                 max_tokens: int = 4096, api_key: str = "",
                 provider: str = "ollama"):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Set up endpoint
        if provider in PROVIDERS and endpoint == PROVIDERS[provider]["base_url"]:
            self.endpoint = PROVIDERS[provider]["base_url"]
        else:
            self.endpoint = endpoint.rstrip("/")

        # Build headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Allow self-signed certs for local endpoints
        verify = True
        if "localhost" in self.endpoint or ".local" in self.endpoint:
            verify = False

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            verify=verify,
            headers=headers,
        )

    @property
    def api_url(self) -> str:
        provider_conf = PROVIDERS.get(self.provider, {})
        api_path = provider_conf.get("api_path", "/v1/chat/completions")
        # Don't double up /v1 if endpoint already has it
        if self.endpoint.endswith("/v1") and api_path.startswith("/v1"):
            api_path = api_path[3:]
        return f"{self.endpoint}{api_path}"

    def _build_tools_payload(self, tools: list[dict]) -> list[dict]:
        """Convert tool definitions to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   stream: bool = True) -> AsyncIterator[dict]:
        """Send chat request, yield streaming chunks.

        Each chunk is one of:
          {"type": "text", "content": "..."}
          {"type": "tool_call", "id": "...", "name": "...", "arguments": {...}}
          {"type": "done", "usage": {...}}
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        if tools:
            payload["tools"] = self._build_tools_payload(tools)

        if stream:
            async for chunk in self._stream_request(payload):
                yield chunk
        else:
            async for chunk in self._blocking_request(payload):
                yield chunk

    async def _stream_request(self, payload: dict) -> AsyncIterator[dict]:
        """Handle streaming response."""
        tool_calls_buffer: dict[int, dict] = {}

        async with self._client.stream("POST", self.api_url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break

                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Text content
                if "content" in delta and delta["content"]:
                    yield {"type": "text", "content": delta["content"]}

                # Tool calls (streamed)
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.get("id", f"call_{idx}"),
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                        if "function" in tc:
                            if "name" in tc["function"] and tc["function"]["name"]:
                                tool_calls_buffer[idx]["name"] = tc["function"]["name"]
                            if "arguments" in tc["function"]:
                                tool_calls_buffer[idx]["arguments"] += tc["function"]["arguments"]

                # Usage info
                usage = chunk.get("usage")
                if usage:
                    self.total_input_tokens += usage.get("prompt_tokens", 0)
                    self.total_output_tokens += usage.get("completion_tokens", 0)

        # Emit buffered tool calls
        for idx in sorted(tool_calls_buffer):
            tc = tool_calls_buffer[idx]
            args = _parse_tool_arguments(tc["arguments"])
            yield {"type": "tool_call", "id": tc["id"], "name": tc["name"], "arguments": args}

        yield {"type": "done", "usage": {
            "total_input": self.total_input_tokens,
            "total_output": self.total_output_tokens,
        }}

    async def _blocking_request(self, payload: dict) -> AsyncIterator[dict]:
        """Handle non-streaming response."""
        payload["stream"] = False
        response = await self._client.post(self.api_url, json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Text
        if message.get("content"):
            yield {"type": "text", "content": message["content"]}

        # Tool calls
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            yield {
                "type": "tool_call",
                "id": tc.get("id", "call_0"),
                "name": func.get("name", ""),
                "arguments": args,
            }

        # Usage
        usage = data.get("usage", {})
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)
        yield {"type": "done", "usage": {
            "total_input": self.total_input_tokens,
            "total_output": self.total_output_tokens,
        }}

    async def close(self):
        await self._client.aclose()
