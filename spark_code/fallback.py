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

    def get_client(self):
        """Get the current active client, creating if needed."""
        name = self.current_provider
        if name not in self._clients:
            if name in self._providers:
                self._clients[name] = self._model_factory(
                    name, self._providers[name])
        return self._clients.get(name)

    async def chat_with_fallback(self, messages, tools=None,
                                 stream=True) -> AsyncIterator[dict]:
        """Try current provider, fallback on failure."""
        for attempt_idx in range(len(self.chain)):
            provider_name = self.chain[
                (self._current_idx + attempt_idx) % len(self.chain)]

            if provider_name not in self._providers:
                continue

            client = self._clients.get(provider_name)
            if not client:
                client = self._model_factory(
                    provider_name, self._providers[provider_name])
                self._clients[provider_name] = client

            start_time = time.monotonic()
            try:
                got_content = False
                async for chunk in client.chat(messages, tools, stream):
                    got_content = True
                    yield chunk
                    if chunk.get("type") == "done":
                        # Success — reset failure count
                        self._failure_counts[provider_name] = 0
                        self._current_idx = (
                            self._current_idx + attempt_idx) % len(self.chain)
                        return

                if got_content:
                    return  # Got something, even if no explicit done

            except Exception as e:
                elapsed = time.monotonic() - start_time
                self._failure_counts[provider_name] = (
                    self._failure_counts.get(provider_name, 0) + 1)

                logger.warning(
                    "Provider %s failed after %.1fs (%d failures): %s",
                    provider_name, elapsed,
                    self._failure_counts[provider_name], str(e)[:200],
                )

                if self._failure_counts[provider_name] >= self.max_retries:
                    logger.info("Falling back from %s", provider_name)
                    continue

        # All providers failed
        yield {"type": "text", "content": "All providers in the fallback chain failed."}
        yield {"type": "done", "usage": {}}

    async def close_all(self):
        """Close all clients."""
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
