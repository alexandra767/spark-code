"""Tests for Task 1: AgentDef.provider field.

Tests for adding an optional provider field to agent definitions, allowing
a sub-agent to run on a configured provider (e.g. "gemini-pro") instead of
the lead's model.
"""

from spark_code.agents_registry import _parse_agent_def


def test_provider_field_parsed(tmp_path):
    p = tmp_path / "web-driver.md"
    p.write_text(
        "---\nname: web-driver\ndescription: drives the browser\n"
        "base_type: implementer\nprovider: gemini-pro\n---\nBody.\n")
    d = _parse_agent_def(p, source="project")
    assert d is not None
    assert d.provider == "gemini-pro"


def test_provider_defaults_none(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("---\nname: plain\ndescription: x\n---\nBody.\n")
    d = _parse_agent_def(p, source="project")
    assert d is not None and d.provider is None
