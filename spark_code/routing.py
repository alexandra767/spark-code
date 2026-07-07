"""Dual-model routing (Phase 4 Task 2).

A second, cheaper "utility" model (default: the DGX's Ollama-served 30B,
``qwen3-coder:30b`` at ``:11434``) can absorb the janitorial requests that
would otherwise burn the primary 80B's context/GPU time: auto-compaction
summaries, the ``/review`` swarm's lens+skeptic dispatches, and (once
model-driven) session labeling. This module is the ONE place callers decide
which client handles a given call — nothing else in the codebase should read
``config["utility_model"]`` directly.

Three small pieces:
  * :func:`get_utility_client` builds a :class:`~spark_code.model.ModelClient`
    for ``config["utility_model"]`` — a provider-shaped dict (same shape as a
    ``providers.<name>`` block). ``None`` when the key is absent from the
    config dict handed in, or explicitly disabled (``utility_model: null``).
  * :func:`enabled_for` reads ``routing.use_for`` (default: every consumer)
    so a user can scope which call sites actually use the utility model.
  * :func:`route_or_fallback` / :func:`call_with_fallback` decide, per call,
    which client handles it — the latter additionally retries against the
    primary model if the utility call raises or otherwise fails, so an
    operation never fails just because the utility model was unreachable.

Session lifecycle (enforced by callers, not this module): the utility
client is built ONCE per session and reused — never rebuilt or closed by a
primary ``/model`` switch, which only touches the primary client. See
``run_interactive``/``_one_shot`` in cli.py.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from .config import get
from .model import ModelClient

logger = logging.getLogger(__name__)

# The real 30B utility model, on the DGX's Ollama instance — NOT vLLM's
# :30000 (that's the primary 80B alias, qwen3.5:122b). Used both as the
# shipped default (config.DEFAULT_CONFIG["utility_model"]) and to fill any
# fields a partial user override omits.
DEFAULT_UTILITY_MODEL: dict[str, Any] = {
    "endpoint": "http://spark-4a54.local:11434",
    "model": "qwen3-coder:30b",
    "provider": "ollama",
}

# Consumers `routing.use_for` can scope.
USE_FOR_CHOICES = ("compaction", "review", "labels")


def get_utility_client(config: dict,
                       client_factory: Callable[..., Any] = ModelClient
                       ) -> Any | None:
    """Build a client for ``config["utility_model"]``, or ``None``.

    * ``"utility_model"`` absent from ``config`` → ``None`` (nothing
      configured — callers that hand in an ad-hoc/minimal config dict, e.g.
      most of this codebase's tests, never unintentionally spin up a
      network client).
    * ``config["utility_model"] is None`` (``utility_model: null`` in YAML)
      → ``None`` (explicitly disabled).
    * A dict → merged over :data:`DEFAULT_UTILITY_MODEL` (so overriding
      just ``model:`` keeps the default endpoint/provider) and built via
      ``client_factory`` (defaults to the real ``ModelClient``; tests inject
      a fake factory so no network is touched).

    Production sessions go through ``config.load_config()``, whose
    ``DEFAULT_CONFIG`` sets ``utility_model`` to :data:`DEFAULT_UTILITY_MODEL`
    — so out of the box this returns a client pointed at the real 30B, and a
    user opts out with an explicit ``utility_model: null``.
    """
    if "utility_model" not in config:
        return None
    block = config["utility_model"]
    if block is None:
        return None
    if not isinstance(block, dict):
        logger.warning(
            "utility_model config is neither a dict nor null (got %s) — "
            "treating as disabled", type(block).__name__)
        return None
    merged = {**DEFAULT_UTILITY_MODEL, **block}
    return client_factory(
        endpoint=merged.get("endpoint", DEFAULT_UTILITY_MODEL["endpoint"]),
        model=merged.get("model", DEFAULT_UTILITY_MODEL["model"]),
        provider=merged.get("provider", DEFAULT_UTILITY_MODEL["provider"]),
        temperature=merged.get("temperature", 0.3),
        max_tokens=merged.get("max_tokens", 4096),
        api_key=merged.get("api_key", ""),
    )


def enabled_for(config: dict, consumer: str) -> bool:
    """True iff ``consumer`` should route through the utility model.

    Reads ``routing.use_for`` — a list of consumer names. Absent (no
    ``routing`` block, or no ``use_for`` key) defaults to EVERY consumer
    (opt-out, not opt-in: once ``utility_model`` is configured it covers
    everything unless scoped down). An explicit list — including an empty
    one — is honored exactly, so ``routing: {use_for: []}`` disables all
    three without having to also null out ``utility_model``.
    """
    use_for = get(config, "routing", "use_for", default=None)
    if use_for is None:
        return True
    if not isinstance(use_for, (list, tuple, set)):
        return True
    return consumer in use_for


def resolve_utility_for(config: dict, consumer: str,
                        utility: Any | None) -> Any | None:
    """Gate an already-built utility client for one consumer.

    Returns ``utility`` unchanged when it is non-``None`` AND ``consumer``
    is in ``routing.use_for``; otherwise ``None`` — meaning "nothing to
    route to", which every downstream ``route_or_fallback``/
    ``call_with_fallback`` call already treats as "use the primary". This is
    the single place the use_for gate is applied, so callers (Agent
    construction, ``/compact``, the ``/review`` swarm) don't each re-derive
    it and risk drifting out of sync.
    """
    if utility is None:
        return None
    return utility if enabled_for(config, consumer) else None


def route_or_fallback(utility: Any | None, primary: Any) -> Any:
    """The one place a caller decides which client to use for a call: the
    utility client when configured (non-``None``), else the primary.

    Pure selection — it does not know about or handle per-call failure; see
    :func:`call_with_fallback` for the version that also retries on error.
    """
    return utility if utility is not None else primary


async def call_with_fallback(
    utility: Any | None,
    primary: Any,
    call: Callable[[Any], Awaitable[Any]],
    is_failure: Callable[[Any], bool] = lambda r: r is None,
) -> Any:
    """Invoke ``await call(model)``, preferring ``utility`` when given.

    * ``utility`` is ``None`` → calls ``primary`` directly. Nothing to fall
      back from.
    * ``utility`` raises (connection error, timeout, ...), or its result
      satisfies ``is_failure`` (default: a ``None`` result counts as a
      failure — matching ``generate_compaction_summary``'s "returns None on
      any failure" contract) → silently retried against ``primary``. The
      operation itself must never fail just because the utility model was
      unreachable for this one call.
    * Otherwise the utility's result is returned as-is.

    A failure from the ``primary`` call is NOT caught here — that's a real
    operation failure, not a routing concern, and propagates to the caller
    exactly as it would without any utility model configured.
    """
    if utility is None:
        return await call(primary)
    try:
        result = await call(utility)
    except Exception:
        logger.warning(
            "utility model call raised — falling back to primary", exc_info=True)
        return await call(primary)
    if is_failure(result):
        logger.info("utility model call failed — falling back to primary")
        return await call(primary)
    return result
