"""Startup preflight checks against the live inference server."""

from __future__ import annotations

import httpx

from .model import _models_url, _should_verify_tls


async def fetch_server_models(
    endpoint: str,
    api_key: str = "",
    timeout: float = 5.0,
    transport=None,
) -> list[dict] | None:
    """Fetch the ``data`` list from an OpenAI-compatible ``/v1/models``.

    Returns None on any error (empty endpoint, non-200 status, connection
    failure, bad JSON) — callers must never block startup on this. Shared by
    the real-model-name banner lookup and the context-window preflight check
    so startup issues a single ``/v1/models`` request instead of one per
    consumer.
    """
    if not endpoint:
        return None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = _models_url(endpoint)
    try:
        # Same local-host TLS bypass resolve_real_model_name used: self-signed
        # HTTPS on localhost/.local endpoints must not fail verification.
        async with httpx.AsyncClient(timeout=timeout, transport=transport,
                                     verify=_should_verify_tls(endpoint)) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            return resp.json().get("data", [])
    except Exception:
        return None


def _extract_max_context(data: list[dict] | None, model: str) -> int | None:
    """Pull ``max_model_len`` for ``model`` out of an already-fetched
    ``/v1/models`` ``data`` list. Pure/sync so callers that already hold the
    fetched payload (e.g. cli.py's startup path) can reuse it without a second
    request."""
    for item in data or []:
        if item.get("id") == model:
            value = item.get("max_model_len")
            return value if isinstance(value, int) else None
    return None


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
    preflight must never block startup. Thin filter over
    ``fetch_server_models``.
    """
    data = await fetch_server_models(endpoint, api_key, timeout, transport)
    return _extract_max_context(data, model)


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
