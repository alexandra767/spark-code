import asyncio

import pytest

from spark_code import dispatch
from spark_code.agents_registry import AgentDef
from spark_code.dispatch import _resolve_subagent_model, run_subagent

CFG = {"providers": {"gemini-pro": {
    "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
    "model": "gemini-2.5-pro", "vision": True, "api_key": "k"}}}

def test_named_provider_builds_owned_client():
    d = AgentDef(name="web-driver", description="x", system_prompt="",
                 base_type="implementer", provider="gemini-pro")
    chosen, owns = _resolve_subagent_model(d, model="LEAD", utility_model="UTIL", config=CFG)
    assert owns is True
    assert chosen.provider == "gemini-pro"
    assert chosen.supports_vision is True

def test_unknown_provider_falls_back_to_primary():
    d = AgentDef(name="x", description="x", system_prompt="", provider="nope")
    chosen, owns = _resolve_subagent_model(d, model="LEAD", utility_model="UTIL", config=CFG)
    assert chosen == "LEAD" and owns is False

def test_no_provider_preserves_utility_hint():
    d = AgentDef(name="x", description="x", system_prompt="", model_hint="utility")
    chosen, owns = _resolve_subagent_model(d, model="LEAD", utility_model="UTIL", config=CFG)
    assert chosen == "UTIL" and owns is False


# ---------------------------------------------------------------------------
# Resource lifecycle: the owned client built by _resolve_subagent_model must
# be closed on every exit path of run_subagent — normal completion, and
# CancelledError while still parked on the concurrency semaphore (regression
# guard for the bug introduced alongside this feature in 6da5de6: the close
# used to live nested inside `async with semaphore:`, so a cancellation
# while still awaiting a slot — or a `_get_semaphore` failure — skipped it
# and leaked the owned client's eagerly-opened httpx.AsyncClient).
# ---------------------------------------------------------------------------


class _TrackingModel:
    """Minimal fake ModelClient: scripted chat() + a close() call counter."""

    def __init__(self, text="done"):
        self._text = text
        self.closed = 0

    async def chat(self, **kwargs):
        yield {"type": "text", "content": self._text}
        yield {"type": "done", "usage": {}}

    async def close(self):
        self.closed += 1


def _cfg(max_concurrent=3, providers=None):
    return {
        "model": {"context_window": 32768},
        "agents": {"max_concurrent": max_concurrent, "max_rounds": 15},
        "providers": providers or {},
    }


def _fake_owned_def(name="web-driver", provider="gemini-pro"):
    return AgentDef(name=name, description="x", system_prompt="Do the task.",
                     base_type="implementer", provider=provider)


async def test_owned_client_closed_after_normal_completion(monkeypatch):
    created = {}

    def fake_model_client(**kwargs):
        client = _TrackingModel()
        created["client"] = client
        return client

    monkeypatch.setattr(dispatch, "ModelClient", fake_model_client)
    dispatch._reset_semaphore()

    lead_model = _TrackingModel("SHOULD NOT BE USED")
    custom = _fake_owned_def()
    config = _cfg(providers={"gemini-pro": {"endpoint": "http://x", "model": "y"}})

    result = await run_subagent(
        lead_model, "do the task", "web-driver", config, "auto",
        agent_defs={"web-driver": custom})

    assert not result.startswith("[dispatch_agent error]")
    assert created["client"].closed == 1
    # The lead's shared model is caller-managed — dispatch must never close it.
    assert lead_model.closed == 0

    dispatch._reset_semaphore()


async def test_no_provider_never_closes_lead_model(monkeypatch):
    lead_model = _TrackingModel("from lead")
    custom = AgentDef(name="plain", description="x", system_prompt="Do the task.",
                       base_type="explore")
    dispatch._reset_semaphore()

    result = await run_subagent(
        lead_model, "do the task", "plain", _cfg(), "auto",
        agent_defs={"plain": custom})

    assert result == "from lead"
    assert lead_model.closed == 0

    dispatch._reset_semaphore()


async def test_owned_client_closed_when_cancelled_awaiting_semaphore(monkeypatch):
    """Regression test for the exact bug: fill the (size-1) shared semaphore
    so a second dispatch is forced to park on `semaphore.acquire()`, cancel
    it while parked there (before it ever enters `async with semaphore:`),
    and assert the owned client built just before that point was still
    closed."""
    created = {}

    def fake_model_client(**kwargs):
        client = _TrackingModel()
        created["client"] = client
        return client

    monkeypatch.setattr(dispatch, "ModelClient", fake_model_client)

    config = _cfg(max_concurrent=1,
                  providers={"gemini-pro": {"endpoint": "http://x", "model": "y"}})
    dispatch._reset_semaphore(limit=1)
    sem = dispatch._get_semaphore(config)
    # Occupy the single slot ourselves, standing in for another in-flight
    # sub-agent, so the task below is guaranteed to block on acquire().
    await sem.acquire()
    try:
        lead_model = _TrackingModel("unused")
        custom = _fake_owned_def()
        task = asyncio.create_task(run_subagent(
            lead_model, "do the task", "web-driver", config, "auto",
            agent_defs={"web-driver": custom}))

        # Give the task enough loop turns to build the owned client and
        # reach (and block on) `await semaphore.acquire()`.
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
        assert created["client"].closed == 0  # still open — parked on the semaphore

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert created["client"].closed == 1  # closed despite never entering the block
        assert lead_model.closed == 0
    finally:
        sem.release()
        dispatch._reset_semaphore()
