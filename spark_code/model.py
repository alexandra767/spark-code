"""OpenAI-compatible API client with streaming and tool calling.

Supports multiple providers:
- Ollama (local, DGX Spark)
- Google Gemini (via OpenAI-compatible API)
- OpenAI
- Any OpenAI-compatible endpoint
"""

import asyncio
import json
import logging
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Retryable HTTP status codes
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class StreamError(Exception):
    """A non-200 HTTP status raised from inside the streaming request so the
    retry wrapper can decide whether to retry or surface it."""

    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        super().__init__(f"API error ({status_code}): {text[:200]}")


def _should_verify_tls(base: str) -> bool:
    """Disable TLS verification only for genuinely local hosts.

    Uses a real hostname parse rather than a substring check, so hosts like
    ``api.localstack.evil.com`` or ``foo.localdomain.com`` are NOT treated as
    local (which would silently disable certificate verification).
    """
    host = (urlparse(base).hostname or "").lower()
    if not host:
        return True
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False
    # mDNS / LAN hostnames end in .local (e.g. spark-4a54.local)
    if host.endswith(".local"):
        return False
    return True


def _models_url(base: str) -> str:
    """Build the `/models` URL for an OpenAI-compatible endpoint.

    Handles bases that already include a version segment (``/v1``) or an
    OpenAI-compat suffix (Gemini's ``/v1beta/openai``) so we don't produce a
    doubled/wrong path that 404s.
    """
    base = base.rstrip("/")
    if base.endswith("/models"):
        return base
    if base.endswith("/v1") or base.endswith("/openai") or "/v1beta" in base:
        return base + "/models"
    return base + "/v1/models"


def _pick_real_model_name(data: list, configured_model: str = "") -> str | None:
    """Given a `/v1/models` ``data`` list, return the underlying model name.

    Handles vLLM's `--served-model-name` masquerade: when a model's ``root``
    field differs from its ``id``, ``root`` is the real model and ``id`` is the
    alias. ``root`` is often a HuggingFace path (e.g.
    ``Intel/Qwen3-Coder-Next-int4-AutoRound``); we strip to the last path
    component for display.

    Selects the entry whose ``id`` matches ``configured_model`` when possible —
    critical for multi-model servers (Ollama lists every pulled model), where
    blindly taking ``data[0]`` reports an arbitrary, wrong model.
    """
    if not data:
        return None
    entry = None
    if configured_model:
        for e in data:
            if e.get("id") == configured_model:
                entry = e
                break
    if entry is None:
        # No exact match: only fall back to the sole entry when unambiguous.
        if len(data) == 1:
            entry = data[0]
        else:
            return None
    served_id = entry.get("id") or ""
    root = entry.get("root") or ""
    if root and root != served_id:
        return root.split("/")[-1]
    return served_id or None


def resolve_real_model_name_sync(endpoint: str, api_key: str = "",
                                 timeout: float = 2.0,
                                 configured_model: str = "") -> str | None:
    """Synchronous variant of `resolve_real_model_name` for non-async callers."""
    if not endpoint:
        return None
    base = endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=timeout, verify=_should_verify_tls(base),
                          headers=headers) as client:
            r = client.get(_models_url(base))
            r.raise_for_status()
            return _pick_real_model_name(r.json().get("data") or [], configured_model)
    except Exception:
        return None


async def resolve_real_model_name(endpoint: str, api_key: str = "",
                                  timeout: float = 2.0,
                                  configured_model: str = "") -> str | None:
    """Query `/v1/models` on an OpenAI-compatible endpoint and return the
    underlying model name (unmasking vLLM's `--served-model-name` alias).

    Pass ``configured_model`` so the correct entry is chosen on multi-model
    servers (e.g. Ollama). Returns None on any failure — caller should fall
    back to the configured name.
    """
    if not endpoint:
        return None
    base = endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout,
                                     verify=_should_verify_tls(base),
                                     headers=headers) as client:
            r = await client.get(_models_url(base))
            r.raise_for_status()
            return _pick_real_model_name(r.json().get("data") or [], configured_model)
    except Exception:
        return None


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


# Approximate cost per 1K tokens by provider (USD)
_COST_TABLE = {
    "gemini": {"input": 0.00001875, "output": 0.000075},  # Gemini 2.0 Flash
    "openai": {"input": 0.0005, "output": 0.0015},  # GPT-4o-mini
    "ollama": {"input": 0.0, "output": 0.0},  # Local = free
    "sglang": {"input": 0.0, "output": 0.0},  # Local = free
}

# Known provider configurations
PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "api_path": "/v1/chat/completions",
        "needs_key": False,
    },
    "sglang": {
        "base_url": "http://spark-4a54.local:30000",
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
                 provider: str = "ollama", timeout: float = 300.0,
                 max_retries: int = 3, real_model_name: str = "",
                 supports_vision: bool = False):
        self.provider = provider
        self.model = model
        # Phase 4 Task 4: gates spark_code.tools.simulator.SimulatorScreenshotTool
        # (and anything else multimodal) — set from config
        # `providers.<name>.vision` / `model.vision` (see config.resolve_provider).
        # False by default: most providers/models this client talks to have no
        # vision, and a wrongly-True default would silently feed an image the
        # model can't read into every request.
        self.supports_vision = supports_vision
        # Underlying model name (unmasked from a vLLM --served-model-name alias);
        # display-only (shown in the startup banner). Falls back to `model`.
        self.real_model_name = real_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._cost_per_1k_input = _COST_TABLE.get(provider, {}).get("input", 0.0)
        self._cost_per_1k_output = _COST_TABLE.get(provider, {}).get("output", 0.0)

        self.endpoint = endpoint.rstrip("/")

        # Build headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            verify=_should_verify_tls(self.endpoint),
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

        if stream:
            # Ask the server to emit a final usage chunk (OpenAI/vLLM/recent
            # Ollama). Servers that don't support it ignore the field.
            payload["stream_options"] = {"include_usage": True}

        if tools:
            payload["tools"] = self._build_tools_payload(tools)

        if stream:
            async for chunk in self._stream_request(payload):
                yield chunk
        else:
            async for chunk in self._blocking_request(payload):
                yield chunk

    async def _stream_request(self, payload: dict) -> AsyncIterator[dict]:
        """Handle streaming response with retry on transient errors.

        Retries only happen *before the first chunk is yielded*: once partial
        output has streamed, replaying the request would duplicate text in the
        conversation, so an interruption is surfaced as an error instead.
        """
        last_error = None
        for attempt in range(self.max_retries):
            if attempt > 0:
                delay = 2 ** (attempt - 1)  # 1s, 2s
                logger.info("Retry %d/%d after %ds", attempt + 1, self.max_retries, delay)
                await asyncio.sleep(delay)
            yielded = False
            try:
                async for chunk in self._stream_request_inner(payload):
                    yielded = True
                    yield chunk
                return
            except StreamError as e:
                if yielded:
                    logger.warning("Stream error after partial output: %s", e)
                    yield {"type": "error", "recoverable": False,
                           "content": f"[stream error after partial output: {e}]"}
                    yield {"type": "done", "usage": {}}
                    return
                if e.status_code not in _RETRYABLE_STATUSES:
                    yield {"type": "error", "recoverable": True,
                           "content": f"API error ({e.status_code}): {e.text[:500]}"}
                    yield {"type": "done", "usage": {}}
                    return
                last_error = e
                logger.warning("Retryable error %d: %s", e.status_code, str(e)[:200])
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                    httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                if yielded:
                    logger.warning("Stream interrupted after partial output: %s", e)
                    yield {"type": "error", "recoverable": False,
                           "content": f"[stream interrupted: {type(e).__name__}]"}
                    yield {"type": "done", "usage": {}}
                    return
                last_error = e
                logger.warning("Connection error (attempt %d): %s", attempt + 1, str(e)[:200])

        # Exhausted retries with no content yielded — a recoverable failure a
        # fallback chain can retry on another provider.
        yield {"type": "error", "recoverable": True,
               "content": f"API error after {self.max_retries} retries: {last_error}"}
        yield {"type": "done", "usage": {}}

    async def _stream_request_inner(self, payload: dict) -> AsyncIterator[dict]:
        """Inner streaming request (single attempt).

        Raises ``StreamError`` on a non-200 response so the outer retry wrapper
        can decide whether to retry. Filters ``<think>...</think>`` blocks and
        ``reasoning_content`` deltas out of the visible stream, holding back
        partial tags that straddle SSE chunks. As a safety net, if the *entire*
        response gets classified as thinking (e.g. a non-reasoning model served
        under a reasoning alias that never emits ``</think>``), the buffered
        thinking is surfaced as text rather than silently swallowed.
        """
        import time as _time
        tool_calls_buffer: dict[int, dict] = {}
        _first_token_time = None
        _token_count = 0
        _req_input = 0
        _req_output = 0

        # Track <think>...</think> blocks so thinking tokens are hidden.
        # Detection is CONTENT-based only: we enter think-hiding when an actual
        # <think> opener arrives in the stream (or a dedicated reasoning_content
        # field appears). We do NOT seed _in_think from the model name/alias —
        # doing so misclassified non-reasoning coder models (e.g. Qwen3-Coder-
        # Next, served as alias qwen3.5:122b) that never emit </think>, routing
        # every content token into the hidden buffer: plain answers never
        # streamed and narration before a tool_call was silently lost.
        _in_think = False
        _think_buf = ""
        _think_accumulated = ""   # raw hidden thinking, for the safety-net flush
        _think_notified = False   # whether we've ever emitted a thinking indicator
        _thinking_open = False    # whether a thinking indicator is currently open
        _visible_emitted = False

        _CLOSE = "</think>"
        _OPEN = "<think>"

        async with self._client.stream("POST", self.api_url, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                error_text = body.decode("utf-8", errors="replace")[:500]
                raise StreamError(response.status_code, error_text)
            async for line in response.aiter_lines():
                # SSE allows both `data: {...}` and the no-space `data:{...}`
                # form; accept either (one optional leading space is stripped
                # per the spec).
                if not line.startswith("data:"):
                    continue
                data = line[5:]
                if data.startswith(" "):
                    data = data[1:]
                if data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed SSE chunk: %s", data[:200])
                    continue

                # Usage can arrive on a chunk with an empty `choices` list
                # (OpenAI/vLLM include_usage) — read it before touching choices.
                usage = chunk.get("usage")
                if usage:
                    _req_input = usage.get("prompt_tokens", _req_input)
                    _req_output = usage.get("completion_tokens", _req_output)

                choices = chunk.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}

                # Dedicated reasoning field (vLLM/DeepSeek reasoning parser) —
                # always hidden thinking. Its presence means the server splits
                # reasoning out of `content`, so `content` is the clean answer:
                # cancel any assumed mid-<think> start.
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    _in_think = False
                    if not _thinking_open:
                        _think_notified = _thinking_open = True
                        yield {"type": "thinking_start"}
                    _think_accumulated += reasoning

                # Text content — filter <think>...</think> blocks
                content = delta.get("content")
                if content:
                    _think_buf += content
                    visible = ""
                    while _think_buf:
                        if _in_think:
                            if not _thinking_open:
                                _think_notified = _thinking_open = True
                                yield {"type": "thinking_start"}
                            end = _think_buf.find(_CLOSE)
                            if end != -1:
                                _think_accumulated += _think_buf[:end]
                                _think_buf = _think_buf[end + len(_CLOSE):]
                                _in_think = False
                                _thinking_open = False
                                yield {"type": "thinking_end"}
                                continue
                            # No closer yet — hold back a possible partial one.
                            keep = len(_CLOSE) - 1
                            if len(_think_buf) > keep:
                                _think_accumulated += _think_buf[:-keep]
                                _think_buf = _think_buf[-keep:]
                            break
                        else:
                            start = _think_buf.find(_OPEN)
                            if start != -1:
                                visible += _think_buf[:start]
                                _think_buf = _think_buf[start + len(_OPEN):]
                                _in_think = True
                                continue
                            # No opener — emit all but a possible partial one.
                            keep = len(_OPEN) - 1
                            if len(_think_buf) > keep:
                                visible += _think_buf[:-keep]
                                _think_buf = _think_buf[-keep:]
                            break
                    if visible:
                        if _thinking_open:
                            _thinking_open = False
                            yield {"type": "thinking_end"}
                        _visible_emitted = True
                        if _first_token_time is None:
                            _first_token_time = _time.monotonic()
                        _token_count += max(1, len(visible.split()))
                        yield {"type": "text", "content": visible}

                # Tool calls (streamed)
                if delta.get("tool_calls"):
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

        # Flush any content still held in the tag buffer.
        if _think_buf:
            if _in_think:
                _think_accumulated += _think_buf
            else:
                if _thinking_open:
                    _thinking_open = False
                    yield {"type": "thinking_end"}
                _visible_emitted = True
                yield {"type": "text", "content": _think_buf}
            _think_buf = ""

        # Safety net: the whole response was classified as thinking and nothing
        # visible (or tool calls) came out — surface it instead of losing it.
        if (not _visible_emitted and not tool_calls_buffer
                and _think_accumulated.strip()):
            if _thinking_open:
                _thinking_open = False
                yield {"type": "thinking_end"}
            recovered = _think_accumulated.replace(_OPEN, "").replace(_CLOSE, "")
            yield {"type": "text", "content": recovered}

        # Emit buffered tool calls
        for idx in sorted(tool_calls_buffer):
            tc = tool_calls_buffer[idx]
            args = _parse_tool_arguments(tc["arguments"])
            yield {"type": "tool_call", "id": tc["id"], "name": tc["name"], "arguments": args}

        # Fold this request's usage into the session totals once.
        self.total_input_tokens += _req_input
        self.total_output_tokens += _req_output

        _elapsed = (_time.monotonic() - _first_token_time) if _first_token_time else 0
        yield {"type": "done", "usage": {
            "prompt_tokens": _req_input,
            "completion_tokens": _req_output,
            "total_input": self.total_input_tokens,
            "total_output": self.total_output_tokens,
        }, "_speed": {"tokens": _token_count, "elapsed": _elapsed}}

    async def _blocking_request(self, payload: dict) -> AsyncIterator[dict]:
        """Handle non-streaming response."""
        payload["stream"] = False
        try:
            response = await self._client.post(self.api_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:500] if e.response else str(e)
            recoverable = e.response.status_code in _RETRYABLE_STATUSES if e.response else True
            yield {"type": "error", "recoverable": recoverable,
                   "content": f"API error ({e.response.status_code}): {error_text}"}
            yield {"type": "done", "usage": {}}
            return
        except httpx.RequestError as e:
            yield {"type": "error", "recoverable": True, "content": f"Request error: {e}"}
            yield {"type": "done", "usage": {}}
            return
        data = response.json()

        choices = data.get("choices") or [{}]
        message = choices[0].get("message", {})

        # Text
        if message.get("content"):
            yield {"type": "text", "content": message["content"]}

        # Tool calls (field may be present but null)
        for tc in (message.get("tool_calls") or []):
            func = tc.get("function", {})
            args = _parse_tool_arguments(func.get("arguments", "{}"))
            yield {
                "type": "tool_call",
                "id": tc.get("id", "call_0"),
                "name": func.get("name", ""),
                "arguments": args,
            }

        # Usage
        usage = data.get("usage", {}) or {}
        req_input = usage.get("prompt_tokens", 0)
        req_output = usage.get("completion_tokens", 0)
        self.total_input_tokens += req_input
        self.total_output_tokens += req_output
        yield {"type": "done", "usage": {
            "prompt_tokens": req_input,
            "completion_tokens": req_output,
            "total_input": self.total_input_tokens,
            "total_output": self.total_output_tokens,
        }}

    @property
    def estimated_cost(self) -> float:
        """Estimated session cost in USD."""
        return (
            (self.total_input_tokens / 1000) * self._cost_per_1k_input
            + (self.total_output_tokens / 1000) * self._cost_per_1k_output
        )

    async def ping(self) -> tuple[bool, str]:
        """Check connectivity to the model endpoint.

        Returns (success, message) tuple.
        """
        try:
            # Use /models endpoint (standard OpenAI-compatible)
            response = await self._client.get(_models_url(self.endpoint), timeout=5.0)
            if response.status_code == 200:
                return True, f"Connected to {self.provider} ({self.model})"
            if response.status_code in (401, 403):
                return False, f"Authentication failed for {self.provider} ({response.status_code}) — check your API key"
            return True, f"Reached {self.provider} at {self.endpoint} [status {response.status_code}]"
        except httpx.ConnectError:
            return False, f"Cannot connect to {self.endpoint} — is the server running?"
        except httpx.TimeoutException:
            return False, f"Connection to {self.endpoint} timed out (5s)"
        except Exception as e:
            return False, f"Connection check failed: {e}"

    async def close(self):
        await self._client.aclose()
