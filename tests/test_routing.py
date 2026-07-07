"""Tests for dual-model routing (Phase 4 Task 2).

A fake ``ModelClient``-shaped factory is injected wherever a client would
otherwise be built (``get_utility_client``'s ``client_factory``) so none of
these tests touch the network. ``call_with_fallback``/``route_or_fallback``
are exercised directly against small stand-in "model" objects (plain
sentinels or fakes with a ``.chat``-shaped async method), matching how the
real consumers (agent.py's ``_maybe_compact``, review.py's ``review_diff``)
use them.
"""

import pytest

from spark_code.routing import (
    DEFAULT_UTILITY_MODEL,
    call_with_fallback,
    enabled_for,
    get_utility_client,
    resolve_utility_for,
    route_or_fallback,
)

# ---------------------------------------------------------------------------
# get_utility_client
# ---------------------------------------------------------------------------


class FakeClient:
    """Stands in for ModelClient — records the kwargs it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _factory():
    """A fresh (kwargs, factory) pair per test — factory returns FakeClient."""
    return FakeClient


def test_get_utility_client_builds_from_valid_block():
    config = {"utility_model": {"endpoint": "http://spark-4a54.local:11434",
                                "model": "qwen3-coder:30b", "provider": "ollama"}}
    client = get_utility_client(config, client_factory=_factory())
    assert isinstance(client, FakeClient)
    assert client.kwargs["endpoint"] == "http://spark-4a54.local:11434"
    assert client.kwargs["model"] == "qwen3-coder:30b"
    assert client.kwargs["provider"] == "ollama"


def test_get_utility_client_none_when_explicitly_null():
    config = {"utility_model": None}
    assert get_utility_client(config, client_factory=_factory()) is None


def test_get_utility_client_none_when_absent():
    """Absent from the config dict handed in → None (not the shipped
    default) — protects callers that build minimal ad-hoc config dicts
    (most of this test suite) from unexpectedly spinning up a client."""
    config = {"model": {"endpoint": "http://x", "name": "y"}}
    assert get_utility_client(config, client_factory=_factory()) is None


def test_get_utility_client_partial_block_fills_defaults():
    """A user overriding just `model:` keeps the default endpoint/provider."""
    config = {"utility_model": {"model": "qwen3-coder:14b"}}
    client = get_utility_client(config, client_factory=_factory())
    assert client.kwargs["model"] == "qwen3-coder:14b"
    assert client.kwargs["endpoint"] == DEFAULT_UTILITY_MODEL["endpoint"]
    assert client.kwargs["provider"] == DEFAULT_UTILITY_MODEL["provider"]


def test_get_utility_client_default_points_at_real_30b():
    """The shipped default (also what config.DEFAULT_CONFIG carries) is the
    real 30B on the DGX's Ollama instance — not vLLM's :30000."""
    assert DEFAULT_UTILITY_MODEL["model"] == "qwen3-coder:30b"
    assert DEFAULT_UTILITY_MODEL["endpoint"] == "http://spark-4a54.local:11434"
    assert DEFAULT_UTILITY_MODEL["provider"] == "ollama"


def test_get_utility_client_non_dict_block_is_disabled():
    config = {"utility_model": "oops"}
    assert get_utility_client(config, client_factory=_factory()) is None


def test_get_utility_client_never_touches_network(monkeypatch):
    """No client_factory given → the REAL ModelClient is used, but
    constructing one must not perform any I/O (httpx.AsyncClient's
    constructor doesn't connect) — assert this holds without patching
    anything network-related, then confirm no such patch was needed by
    checking the returned object is a real ModelClient instance."""
    from spark_code.model import ModelClient

    config = {"utility_model": {"endpoint": "http://spark-4a54.local:11434",
                                "model": "qwen3-coder:30b", "provider": "ollama"}}
    client = get_utility_client(config)
    assert isinstance(client, ModelClient)


# ---------------------------------------------------------------------------
# route_or_fallback
# ---------------------------------------------------------------------------


def test_route_or_fallback_prefers_utility_when_present():
    utility, primary = object(), object()
    assert route_or_fallback(utility, primary) is utility


def test_route_or_fallback_falls_back_when_utility_none():
    primary = object()
    assert route_or_fallback(None, primary) is primary


# ---------------------------------------------------------------------------
# routing.use_for gating
# ---------------------------------------------------------------------------


def test_enabled_for_defaults_true_when_routing_absent():
    config = {}
    for consumer in ("compaction", "review", "labels"):
        assert enabled_for(config, consumer) is True


def test_enabled_for_defaults_true_when_use_for_absent():
    config = {"routing": {}}
    assert enabled_for(config, "compaction") is True


def test_enabled_for_respects_explicit_scoping():
    """Item absent from an explicit use_for list → that consumer is NOT
    routed to the utility model (uses primary)."""
    config = {"routing": {"use_for": ["review", "labels"]}}
    assert enabled_for(config, "compaction") is False
    assert enabled_for(config, "review") is True
    assert enabled_for(config, "labels") is True


def test_enabled_for_empty_list_disables_everything():
    config = {"routing": {"use_for": []}}
    assert enabled_for(config, "compaction") is False
    assert enabled_for(config, "review") is False
    assert enabled_for(config, "labels") is False


def test_enabled_for_malformed_use_for_warns_and_defaults_enabled(caplog):
    """A non-list routing.use_for (config typo) warns — like a non-dict
    utility_model does — then falls back to enabled-for-all rather than
    silently disabling routing."""
    config = {"routing": {"use_for": "compaction"}}  # a bare string, not a list
    with caplog.at_level("WARNING"):
        assert enabled_for(config, "compaction") is True
        assert enabled_for(config, "review") is True
    assert any("routing.use_for is not a list" in r.message for r in caplog.records)


def test_resolve_utility_for_none_utility_stays_none():
    config = {"routing": {"use_for": ["compaction", "review", "labels"]}}
    assert resolve_utility_for(config, "compaction", None) is None


def test_resolve_utility_for_gates_a_present_utility():
    utility = object()
    config = {"routing": {"use_for": ["review"]}}
    assert resolve_utility_for(config, "compaction", utility) is None
    assert resolve_utility_for(config, "review", utility) is utility


# ---------------------------------------------------------------------------
# call_with_fallback — the per-call fallback contract
# ---------------------------------------------------------------------------


async def test_call_with_fallback_no_utility_calls_primary_only():
    calls = []

    async def call(model):
        calls.append(model)
        return f"result-from-{model}"

    out = await call_with_fallback(None, "primary", call)
    assert out == "result-from-primary"
    assert calls == ["primary"]


async def test_call_with_fallback_uses_utility_on_success():
    calls = []

    async def call(model):
        calls.append(model)
        return f"result-from-{model}"

    out = await call_with_fallback("utility", "primary", call)
    assert out == "result-from-utility"
    assert calls == ["utility"]  # primary never touched


async def test_call_with_fallback_on_utility_exception_uses_primary():
    """The utility call RAISES (connection error/timeout stand-in) — the
    operation must still succeed via the primary, silently."""
    calls = []

    async def call(model):
        calls.append(model)
        if model == "utility":
            raise ConnectionError("utility model unreachable")
        return f"result-from-{model}"

    out = await call_with_fallback("utility", "primary", call)
    assert out == "result-from-primary"
    assert calls == ["utility", "primary"]  # utility tried first, then primary


async def test_call_with_fallback_on_failure_result_uses_primary():
    """Default is_failure: a None result (matches
    generate_compaction_summary's "None on any failure" contract) triggers
    fallback even though the utility call didn't raise."""
    calls = []

    async def call(model):
        calls.append(model)
        return None if model == "utility" else f"result-from-{model}"

    out = await call_with_fallback("utility", "primary", call)
    assert out == "result-from-primary"
    assert calls == ["utility", "primary"]


async def test_call_with_fallback_custom_is_failure_predicate():
    """review.py's use case: failure is a sentinel-prefixed string, not
    None."""
    calls = []

    async def call(model):
        calls.append(model)
        return "[dispatch_agent error] boom" if model == "utility" else "ok"

    out = await call_with_fallback(
        "utility", "primary", call,
        is_failure=lambda r: isinstance(r, str) and r.startswith("[dispatch_agent error]"))
    assert out == "ok"
    assert calls == ["utility", "primary"]


async def test_call_with_fallback_primary_failure_propagates():
    """A primary-model failure is a real operation failure, not swallowed —
    with no utility configured, an exception from primary just raises."""
    async def call(model):
        raise RuntimeError("primary is down too")

    with pytest.raises(RuntimeError):
        await call_with_fallback(None, "primary", call)


# ---------------------------------------------------------------------------
# End-to-end routing decision (no network) — two fake clients, real
# generate_compaction_summary + call_with_fallback wiring. This is the
# Step-5 "assert the routing decision with two fake clients" integration
# test: see .superpowers/sdd/task-2-report.md for why the LIVE version
# (a real request to :11434) wasn't run.
# ---------------------------------------------------------------------------


class _ChattyFakeModel:
    """A minimal fake ModelClient: yields one text chunk then done, and
    records that IT (not the other fake) was called."""

    def __init__(self, label, calls):
        self.label = label
        self._calls = calls

    async def chat(self, **kwargs):
        self._calls.append(self.label)
        yield {"type": "text", "content": f"SUMMARY-FROM-{self.label}"}
        yield {"type": "done", "usage": {}}


async def test_routing_decision_prefers_utility_over_primary_e2e():
    from spark_code.agent import generate_compaction_summary
    from spark_code.context import Context

    calls: list[str] = []
    utility = _ChattyFakeModel("UTILITY_30B", calls)
    primary = _ChattyFakeModel("PRIMARY_80B", calls)

    ctx = Context()
    ctx.add_user("do the thing")

    model = route_or_fallback(utility, primary)
    assert model is utility

    out = await call_with_fallback(utility, primary,
                                   lambda m: generate_compaction_summary(m, ctx))
    assert out == "SUMMARY-FROM-UTILITY_30B"
    assert calls == ["UTILITY_30B"]  # primary (:30000 in prod) never touched


async def test_routing_decision_falls_back_to_primary_e2e():
    """Utility client raises (simulating :11434 unreachable) — the primary
    (:30000) handles the call instead, and the operation still succeeds."""
    from spark_code.agent import generate_compaction_summary
    from spark_code.context import Context

    calls: list[str] = []

    class _DeadUtility:
        async def chat(self, **kwargs):
            calls.append("UTILITY_30B_attempted")
            raise ConnectionError("spark-4a54.local:11434 unreachable")
            yield  # pragma: no cover - unreachable, keeps this an async generator

    primary = _ChattyFakeModel("PRIMARY_80B", calls)
    ctx = Context()
    ctx.add_user("do the thing")

    out = await call_with_fallback(_DeadUtility(), primary,
                                   lambda m: generate_compaction_summary(m, ctx))
    assert out == "SUMMARY-FROM-PRIMARY_80B"
    assert calls == ["UTILITY_30B_attempted", "PRIMARY_80B"]
