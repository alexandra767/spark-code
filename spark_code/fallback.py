"""Model fallback chain — auto-switch providers on failure.

Configuration:
  fallback:
    chain: [ollama, gemini]
    timeout_threshold: 30  # seconds before trying next
    max_retries_per_provider: 2
"""

import logging
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class FallbackChain:
    """Wraps multiple ModelClients with automatic failover."""

    def __init__(self, providers: dict, config: dict, model_factory):
        """
        providers: config provider dict (name -> settings)
        config: fallback config section
        model_factory: callable(provider_name, provider_config) -> ModelClient
        """
        self.chain = config.get("chain", [])
        self.timeout_threshold = config.get("timeout_threshold", 30)
        self.max_retries = config.get("max_retries_per_provider", 2)
        self._providers = providers
        self._model_factory = model_factory
        self._clients: dict = {}  # name -> ModelClient
        self._current_idx = 0
        self._failure_counts: dict[str, int] = {}

    @property
    def current_provider(self) -> str:
        if not self.chain:
            return ""
        return self.chain[min(self._current_idx, len(self.chain) - 1)]

    def _client_for(self, name: str):
        """Return (creating if needed) the client for a provider name."""
        if name not in self._clients and name in self._providers:
            self._clients[name] = self._model_factory(name, self._providers[name])
        return self._clients.get(name)

    def get_client(self):
        """Get the current active client, creating if needed."""
        return self._client_for(self.current_provider)

    async def chat_with_fallback(self, messages, tools=None,
                                 stream=True) -> AsyncIterator[dict]:
        """Try the current provider; fall over to the next on a RECOVERABLE
        failure that happens BEFORE any real content is streamed.

        Failure is detected via the client's ``{"type": "error"}`` chunk (with
        ``recoverable``). Once real content (text/tool_call) has been yielded to
        the consumer we can't un-yield it, so a later error is surfaced rather
        than retried on another provider.
        """
        last_error = "no providers configured"
        n = len(self.chain)
        for attempt_idx in range(n):
            provider_name = self.chain[(self._current_idx + attempt_idx) % n]
            if provider_name not in self._providers:
                continue
            client = self._client_for(provider_name)
            if client is None:
                continue

            start_time = time.monotonic()
            yielded_real = False
            failed_recoverably = False
            try:
                async for chunk in client.chat(messages, tools, stream):
                    ctype = chunk.get("type")
                    if ctype == "error":
                        last_error = chunk.get("content", "model error")
                        if yielded_real or not chunk.get("recoverable", True):
                            # Can't fail over cleanly — surface it.
                            yield chunk
                            yield {"type": "done", "usage": {}}
                            return
                        failed_recoverably = True
                        break  # try the next provider
                    yield chunk
                    if ctype in ("text", "tool_call"):
                        yielded_real = True
                    if ctype == "done":
                        # Success — remember this provider as the new default.
                        self._failure_counts[provider_name] = 0
                        self._current_idx = (self._current_idx + attempt_idx) % n
                        return
            except Exception as e:  # a client that raises instead of yielding
                last_error = str(e)
                if yielded_real:
                    yield {"type": "error", "recoverable": False,
                           "content": f"[stream interrupted: {type(e).__name__}]"}
                    yield {"type": "done", "usage": {}}
                    return
                failed_recoverably = True

            if failed_recoverably:
                self._failure_counts[provider_name] = (
                    self._failure_counts.get(provider_name, 0) + 1)
                logger.warning(
                    "Provider %s failed after %.1fs — falling back. %s",
                    provider_name, time.monotonic() - start_time, str(last_error)[:200])

        # Every provider failed before producing content.
        yield {"type": "error", "recoverable": False,
               "content": f"All providers in the fallback chain failed. Last: {last_error}"}
        yield {"type": "done", "usage": {}}

    # ----- ModelClient-compatible surface (so it can be dropped in as
    # ----- `agent.model`). Attributes delegate to the current client.

    async def chat(self, messages, tools=None, stream=True) -> AsyncIterator[dict]:
        async for chunk in self.chat_with_fallback(messages, tools, stream):
            yield chunk

    @property
    def model(self) -> str:
        c = self.get_client()
        return getattr(c, "model", "") if c else ""

    @property
    def provider(self) -> str:
        return self.current_provider

    @property
    def real_model_name(self) -> str:
        c = self.get_client()
        return getattr(c, "real_model_name", "") if c else ""

    @property
    def total_input_tokens(self) -> int:
        return sum(getattr(c, "total_input_tokens", 0) for c in self._clients.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(getattr(c, "total_output_tokens", 0) for c in self._clients.values())

    @property
    def estimated_cost(self) -> float:
        return sum(getattr(c, "estimated_cost", 0.0) for c in self._clients.values())

    async def ping(self):
        c = self.get_client()
        if c is None:
            return False, "No providers configured in fallback chain"
        return await c.ping()

    async def close(self):
        await self.close_all()

    async def close_all(self):
        """Close all clients."""
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
