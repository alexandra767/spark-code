from spark_code.agents_registry import AgentDef
from spark_code.dispatch import _resolve_subagent_model

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
