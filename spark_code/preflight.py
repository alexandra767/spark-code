"""Startup preflight checks against the live inference server."""

from __future__ import annotations

import httpx


async def fetch_server_max_context(
    endpoint: str,
    model: str,
    api_key: str = "",
    timeout: float = 5.0,
    transport=None,
) -> int | None:
    """Return the server-reported max context length for ``model``, or None.

    vLLM reports ``max_model_len`` per model on /v1/models; Ollama's
    OpenAI-compat endpoint omits the field. Any error means "unknown" —
    preflight must never block startup.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"{endpoint.rstrip('/')}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            for item in resp.json().get("data", []):
                if item.get("id") == model:
                    value = item.get("max_model_len")
                    return value if isinstance(value, int) else None
    except Exception:
        return None
    return None


def context_window_warning(configured: int, server_max: int | None) -> str | None:
    """Human-readable drift warning, or None when config matches reality."""
    if server_max is None or configured == server_max:
        return None
    if configured > server_max:
        return (
            f"context_window {configured} exceeds the server's max "
            f"{server_max} — long sessions will 400; lower context_window "
            f"in ~/.spark/config.yaml"
        )
    return (
        f"server supports {server_max} tokens but context_window is "
        f"{configured} — raise it in ~/.spark/config.yaml to use the full window"
    )
