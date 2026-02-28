"""Ollama / OpenAI-compatible API client with streaming and tool calling."""

import json
import httpx
from typing import AsyncIterator


class ModelClient:
    """Client for Ollama's OpenAI-compatible chat API."""

    def __init__(self, endpoint: str, model: str, temperature: float = 0.7,
                 max_tokens: int = 4096):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Allow self-signed certs (DGX Spark uses self-signed SSL)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            verify=False,
        )

    @property
    def api_url(self) -> str:
        return f"{self.endpoint}/v1/chat/completions"

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
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"raw": tc["arguments"]}
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
