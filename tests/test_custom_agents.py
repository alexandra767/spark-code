"""Tests for custom subagent definitions (Phase 5 Task 4).

``.spark/agents/*.md`` (project) + ``~/.spark/agents/*.md`` (global) let a
user define new ``dispatch_agent`` agent_type values: a markdown file with
YAML frontmatter (reusing skills/base.py's ``_FRONTMATTER_RE`` parser) whose
body becomes the sub-agent's system prompt.

The security-critical property under test: a custom def's ``tools``
allowlist can only NARROW its ``base_type``'s permissions, never widen them
— a read-only-class def (the default) can never end up with a write tool no
matter what it claims in ``tools:``.
"""

from pathlib import Path

import pytest

from spark_code import dispatch
from spark_code.agent import Agent
from spark_code.agents_registry import AgentDef, build_agent_index, load_agent_defs
from spark_code.context import Context
from spark_code.permissions import PermissionManager
from spark_code.plan_mode import PlanState
from spark_code.tools.base import Tool, ToolRegistry

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own fake $HOME so load_agent_defs's global-dir
    scan (``~/.spark/agents``) can never see this dev machine's real
    ``~/.spark/agents`` (or lack thereof) — deterministic across machines/CI.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    return tmp_path


def _write_def(directory: Path, filename: str, frontmatter: str, body: str = "Do the task.") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    md = directory / filename
    md.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return md


class MockModel:
    """Yields a scripted stream per chat() call (spark's standard fake model)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self._call = 0
        self.captured_messages: list = []

    async def chat(self, **kwargs):
        self.captured_messages.append(kwargs.get("messages"))
        if self._call < len(self.responses):
            chunks = self.responses[self._call]
            self._call += 1
            for chunk in chunks:
                yield chunk

    async def close(self):
        pass


def _final_text(text):
    return [[{"type": "text", "content": text}, {"type": "done", "usage": {}}]]


def _cfg(**overrides):
    cfg = {"model": {"context_window": 32768}, "agents": {"max_concurrent": 3, "max_rounds": 15}}
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def _reset_dispatch_semaphore():
    dispatch._reset_semaphore()
    yield
    dispatch._reset_semaphore()


# ---------------------------------------------------------------------------
# load_agent_defs — parsing, project/global merge, malformed-file tolerance
# ---------------------------------------------------------------------------


def test_load_agent_defs_reads_project_def_with_frontmatter(tmp_path):
    project = tmp_path / "proj"
    _write_def(
        project / ".spark" / "agents", "reviewer2.md",
        "name: reviewer2\ndescription: A second read-only reviewer\ntools: [read_file, grep]",
        "You review code for style issues only.",
    )

    defs = load_agent_defs(str(project))

    assert "reviewer2" in defs
    d = defs["reviewer2"]
    assert isinstance(d, AgentDef)
    assert d.description == "A second read-only reviewer"
    assert d.tools == ["read_file", "grep"]
    assert d.system_prompt == "You review code for style issues only."
    assert d.source == "project"
    assert d.base_type == "explore"  # default — read-only
    assert d.is_read_only is True


def test_load_agent_defs_requires_explicit_name_field(tmp_path):
    """Unlike some loaders, a missing `name:` is NOT inferred from the file
    stem (matches skills/base.py's _parse_claude_skill, which has the same
    requirement) — it's treated as malformed and skipped."""
    project = tmp_path / "proj"
    _write_def(project / ".spark" / "agents", "finder.md", "description: finds things", "Find X.")
    defs = load_agent_defs(str(project))
    assert defs == {}


def test_load_agent_defs_no_directories_returns_empty(tmp_path):
    assert load_agent_defs(str(tmp_path / "nowhere")) == {}


def test_load_agent_defs_project_overrides_global_on_name_clash(tmp_path):
    home = Path.home()  # the autouse _isolated_home fixture points this at a temp dir
    _write_def(
        home / ".spark" / "agents", "scout.md",
        "name: scout\ndescription: GLOBAL scout",
        "Global body.",
    )
    project = tmp_path / "proj"
    _write_def(
        project / ".spark" / "agents", "scout.md",
        "name: scout\ndescription: PROJECT scout",
        "Project body.",
    )

    defs = load_agent_defs(str(project))

    assert defs["scout"].description == "PROJECT scout"
    assert defs["scout"].system_prompt == "Project body."
    assert defs["scout"].source == "project"


def test_load_agent_defs_global_only_def_still_loads(tmp_path):
    home = Path.home()
    _write_def(
        home / ".spark" / "agents", "global_only.md",
        "name: global_only\ndescription: only in global",
        "Body.",
    )
    project = tmp_path / "proj"
    project.mkdir()

    defs = load_agent_defs(str(project))

    assert "global_only" in defs
    assert defs["global_only"].source == "global"


@pytest.mark.parametrize("filename,frontmatter,body", [
    ("no_frontmatter.md", None, None),          # handled specially below
    ("bad_yaml.md", "this: [is: not: yaml", "body"),
    ("no_name.md", "description: missing name", "body"),
    ("empty_body.md", "name: empty_body", ""),
    ("bad_tools_type.md", "name: bad_tools\ntools: 5", "body"),
    ("bad_frontmatter_type.md", "- just\n- a\n- list", "body"),
])
def test_malformed_agent_defs_skipped_not_raised(tmp_path, filename, frontmatter, body):
    project = tmp_path / "proj"
    directory = project / ".spark" / "agents"
    directory.mkdir(parents=True)
    path = directory / filename
    if frontmatter is None:
        path.write_text("Just a markdown body, no frontmatter at all.\n", encoding="utf-8")
    else:
        path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")

    # Must not raise.
    defs = load_agent_defs(str(project))
    # None of these malformed files produced a registered def.
    assert defs == {}


def test_one_malformed_file_does_not_block_the_others(tmp_path):
    project = tmp_path / "proj"
    directory = project / ".spark" / "agents"
    _write_def(directory, "good.md", "name: good\ndescription: fine", "Body.")
    (directory / "bad.md").write_text("---\n[not: yaml: at: all\n---\nbody", encoding="utf-8")

    defs = load_agent_defs(str(project))

    assert "good" in defs
    assert len(defs) == 1


def test_custom_def_cannot_redefine_builtin_agent_type_name(tmp_path):
    project = tmp_path / "proj"
    _write_def(
        project / ".spark" / "agents", "explore.md",
        "name: explore\ndescription: shadow attempt",
        "Body.",
    )
    defs = load_agent_defs(str(project))
    assert "explore" not in defs  # reserved — the built-in wins, def is skipped


def test_custom_def_type_field_sets_base_type(tmp_path):
    project = tmp_path / "proj"
    _write_def(
        project / ".spark" / "agents", "fixer.md",
        "name: fixer\ndescription: can write\ntype: implementer",
        "Fix the bug and save the file.",
    )
    defs = load_agent_defs(str(project))
    assert defs["fixer"].base_type == "implementer"
    assert defs["fixer"].is_read_only is False


def test_custom_def_invalid_type_defaults_to_explore(tmp_path):
    project = tmp_path / "proj"
    _write_def(
        project / ".spark" / "agents", "weird.md",
        "name: weird\ntype: wizard",
        "Body.",
    )
    defs = load_agent_defs(str(project))
    assert defs["weird"].base_type == "explore"


def test_custom_def_model_hint_parsed_and_validated(tmp_path):
    project = tmp_path / "proj"
    _write_def(project / ".spark" / "agents", "cheap.md",
              "name: cheap\nmodel: utility", "Body.")
    _write_def(project / ".spark" / "agents", "bogus.md",
              "name: bogus\nmodel: nonsense", "Body.")
    defs = load_agent_defs(str(project))
    assert defs["cheap"].model_hint == "utility"
    assert defs["bogus"].model_hint is None  # invalid hint ignored, not raised


def test_build_agent_index_lean_one_liner_per_def(tmp_path):
    defs = {
        "finder": AgentDef(name="finder", description="finds things", system_prompt="x"),
        "alpha": AgentDef(name="alpha", description="", system_prompt="x"),
    }
    index = build_agent_index(defs)
    lines = index.splitlines()
    assert lines == ["- alpha", "- finder: finds things"]  # sorted, full bodies absent
    assert "x" not in index  # system_prompt body never appears in the index


# ---------------------------------------------------------------------------
# Permission intersection — the security-critical property
# ---------------------------------------------------------------------------


def test_readonly_class_def_with_write_tool_claim_gets_no_write_access():
    """A def whose base_type is read-only (the default) claims write_file in
    its tools allowlist — it must NOT get write_file, because write_file was
    never in the base explore-filtered registry to intersect with."""
    sneaky = AgentDef(
        name="sneaky", description="", system_prompt="x",
        tools=["write_file", "read_file", "bash"],
        base_type="explore",
    )
    registry = dispatch._build_custom_registry(sneaky)
    names = registry.names()
    assert "write_file" not in names
    assert "bash" not in names
    assert "read_file" in names  # the one legitimately-read-only tool survives
    for tool in registry.all():
        assert tool.is_read_only


def test_reviewer_class_def_is_also_read_only_by_construction():
    d = AgentDef(name="r2", description="", system_prompt="x", base_type="reviewer")
    registry = dispatch._build_custom_registry(d)
    assert "write_file" not in registry.names()
    assert "read_file" in registry.names()


def test_implementer_class_def_can_get_write_tools_when_declared():
    d = AgentDef(
        name="fixer", description="", system_prompt="x",
        tools=["write_file", "read_file"], base_type="implementer",
    )
    registry = dispatch._build_custom_registry(d)
    names = registry.names()
    assert "write_file" in names
    assert "read_file" in names
    assert "dispatch_agent" not in names  # depth guard still applies


def test_implementer_class_def_tools_none_gets_full_implementer_default():
    d = AgentDef(name="fixer2", description="", system_prompt="x",
                tools=None, base_type="implementer")
    registry = dispatch._build_custom_registry(d)
    assert "write_file" in registry.names()
    assert "edit_file" in registry.names()


def test_explore_class_def_tools_none_gets_full_readonly_default():
    d = AgentDef(name="explorer2", description="", system_prompt="x", tools=None)
    registry = dispatch._build_custom_registry(d)
    names = registry.names()
    assert "read_file" in names
    assert "grep" in names
    assert "write_file" not in names


def test_custom_def_tools_allowlist_narrows_within_base_type():
    d = AgentDef(name="narrow", description="", system_prompt="x",
                tools=["read_file"], base_type="explore")
    registry = dispatch._build_custom_registry(d)
    assert registry.names() == ["read_file"]


def test_custom_def_unknown_tool_name_silently_ignored():
    d = AgentDef(name="typo", description="", system_prompt="x",
                tools=["read_fiel"], base_type="explore")  # typo
    registry = dispatch._build_custom_registry(d)
    assert registry.names() == []


# ---------------------------------------------------------------------------
# run_subagent — dispatch of a custom type
# ---------------------------------------------------------------------------


async def test_dispatch_of_custom_type_uses_its_system_prompt():
    model = MockModel(_final_text("finder result"))
    custom = AgentDef(
        name="finder", description="finds definitions", system_prompt="You find where things are defined.",
        base_type="explore",
    )
    result = await dispatch.run_subagent(
        model, "find GREETING", "finder", _cfg(), "auto",
        agent_defs={"finder": custom})

    assert result == "finder result"
    # The def's system_prompt body must be present in the request sent to
    # the model (it becomes the sub-agent's Context.system_prompt).
    sent_messages = model.captured_messages[0]
    system_msg = next(m for m in sent_messages if m["role"] == "system")
    assert "You find where things are defined." in system_msg["content"]


async def test_unknown_custom_type_still_errors_like_before():
    model = MockModel(_final_text("unused"))
    result = await dispatch.run_subagent(
        model, "do it", "not-a-real-type", _cfg(), "auto", agent_defs={"finder": AgentDef(
            name="finder", description="", system_prompt="x")})
    assert result.startswith("[dispatch_agent error]")
    assert "finder" in result  # helpful: lists the custom name that WAS available


async def test_custom_implementer_def_can_actually_write(tmp_path):
    """End-to-end: an implementer-class custom def gets a registry containing
    write_file and the sub-agent can use it (proves the registry returned by
    _build_custom_registry is actually wired into the running sub-agent, not
    just unit-tested in isolation)."""
    target = tmp_path / "out.txt"
    model = MockModel([
        [
            {"type": "tool_call", "id": "c1", "name": "write_file",
             "arguments": {"file_path": str(target), "content": "hello"}},
            {"type": "done", "usage": {}},
        ],
        [{"type": "text", "content": "wrote it"}, {"type": "done", "usage": {}}],
    ])
    custom = AgentDef(name="writer", description="", system_prompt="Write files.",
                      base_type="implementer")
    result = await dispatch.run_subagent(
        model, "write the file", "writer", _cfg(), "trust",
        agent_defs={"writer": custom})
    assert result == "wrote it"
    assert target.read_text() == "hello"


async def test_custom_model_hint_utility_routes_to_utility_model():
    primary = MockModel(_final_text("SHOULD NOT BE USED"))
    utility = MockModel(_final_text("from utility"))
    custom = AgentDef(name="cheap", description="", system_prompt="x", model_hint="utility")

    result = await dispatch.run_subagent(
        primary, "task", "cheap", _cfg(), "auto",
        agent_defs={"cheap": custom}, utility_model=utility)

    assert result == "from utility"
    assert primary.captured_messages == []  # primary never called


async def test_custom_no_model_hint_uses_primary_even_with_utility_available():
    primary = MockModel(_final_text("from primary"))
    utility = MockModel(_final_text("SHOULD NOT BE USED"))
    custom = AgentDef(name="normal", description="", system_prompt="x", model_hint=None)

    result = await dispatch.run_subagent(
        primary, "task", "normal", _cfg(), "auto",
        agent_defs={"normal": custom}, utility_model=utility)

    assert result == "from primary"
    assert utility.captured_messages == []


# ---------------------------------------------------------------------------
# DispatchAgentTool — schema gains custom names + lean index
# ---------------------------------------------------------------------------


def test_tool_schema_enum_gains_custom_names():
    defs = {
        "finder": AgentDef(name="finder", description="finds stuff", system_prompt="x"),
    }
    tool = dispatch.DispatchAgentTool(model=None, config=_cfg(), agent_defs=defs)
    schema = tool.to_schema()
    assert schema["parameters"]["properties"]["agent_type"]["enum"] == [
        "explore", "reviewer", "implementer", "finder"]
    assert "finder" in schema["description"]
    assert "finds stuff" in schema["description"]


def test_tool_schema_unaffected_when_no_custom_defs():
    tool = dispatch.DispatchAgentTool(model=None, config=_cfg())
    schema = tool.to_schema()
    assert schema["parameters"]["properties"]["agent_type"]["enum"] == [
        "explore", "reviewer", "implementer"]


async def test_tool_execute_dispatches_custom_type():
    model = MockModel(_final_text("custom done"))
    defs = {"finder": AgentDef(name="finder", description="", system_prompt="Find things.")}
    tool = dispatch.DispatchAgentTool(model=model, config=_cfg(), lead_mode="auto", agent_defs=defs)
    result = await tool.execute(prompt="find X", agent_type="finder")
    assert result == "custom done"


# ---------------------------------------------------------------------------
# build_tools registration threading
# ---------------------------------------------------------------------------


def test_build_tools_threads_agent_defs_into_dispatch_tool():
    from spark_code.cli import build_tools
    defs = {"finder": AgentDef(name="finder", description="d", system_prompt="x")}
    registry = build_tools(model=MockModel([]), config=_cfg(), agent_defs=defs)
    dispatch_tool = registry.get("dispatch_agent")
    assert dispatch_tool is not None
    schema = dispatch_tool.to_schema()
    assert "finder" in schema["parameters"]["properties"]["agent_type"]["enum"]


# ---------------------------------------------------------------------------
# Plan mode: custom read-only agents treated like explore/reviewer
# ---------------------------------------------------------------------------


class _FakeDispatch(Tool):
    name = "dispatch_agent"
    description = "fake"
    is_read_only = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "ok"


def _agent_with_defs(agent_defs, plan_active=True):
    return Agent(
        model=MockModel([]),
        context=Context(),
        tools=ToolRegistry(),
        permissions=PermissionManager(mode="auto"),
        plan_state=PlanState(active=plan_active),
        agent_defs=agent_defs,
    )


def test_plan_denies_allows_custom_readonly_class_dispatch():
    defs = {"finder": AgentDef(name="finder", description="", system_prompt="x", base_type="explore")}
    agent = _agent_with_defs(defs)
    dispatch_tool = _FakeDispatch()
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": {"agent_type": "finder"}},
        dispatch_tool) is False


def test_plan_denies_blocks_custom_implementer_class_dispatch():
    defs = {"fixer": AgentDef(name="fixer", description="", system_prompt="x", base_type="implementer")}
    agent = _agent_with_defs(defs)
    dispatch_tool = _FakeDispatch()
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": {"agent_type": "fixer"}},
        dispatch_tool) is True


def test_plan_denies_blocks_unknown_custom_type_fail_closed():
    agent = _agent_with_defs({})
    dispatch_tool = _FakeDispatch()
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": {"agent_type": "does-not-exist"}},
        dispatch_tool) is True


def test_plan_denies_still_allows_builtin_explore_with_agent_defs_present():
    """Adding custom defs must not disturb the pre-existing built-in behavior."""
    defs = {"finder": AgentDef(name="finder", description="", system_prompt="x")}
    agent = _agent_with_defs(defs)
    dispatch_tool = _FakeDispatch()
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": {"agent_type": "explore"}},
        dispatch_tool) is False
    assert agent._plan_denies(
        {"name": "dispatch_agent", "arguments": {"agent_type": "implementer"}},
        dispatch_tool) is True
