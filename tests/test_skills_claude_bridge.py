"""Tests for the Claude Code skill loader bridge in skills/base.py.

Also covers Phase 2 Task 8 (80B-friendly skills): slash-command fallback to
skills in cli.py's handle_slash_command, the lean skill INDEX that's all the
system prompt ever sees (never full bodies), and agent.py's model-reply
auto-expand for a bare "/<skill-name> [args]" final answer.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from rich.console import Console

from spark_code.agent import Agent
from spark_code.cli import handle_slash_command
from spark_code.context import Context, build_skill_prompt_section
from spark_code.model import ModelClient
from spark_code.permissions import PermissionManager
from spark_code.skills.base import (
    CLAUDE_TOOL_ALIASES,
    Skill,
    SkillRegistry,
    _parse_claude_skill,
    check_skill_compatibility,
)
from spark_code.tools.base import ToolRegistry


def _write_claude_skill(root: Path, name: str, frontmatter: str, body: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    md = skill_dir / "SKILL.md"
    md.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return md


def test_parse_claude_skill_basic(tmp_path):
    md = _write_claude_skill(
        tmp_path,
        "hello-skill",
        "name: hello-skill\ndescription: Say hi\nallowed-tools: Bash, Read",
        "# Hello\nDo a thing.",
    )

    skill = _parse_claude_skill(md)

    assert skill is not None
    assert skill.name == "hello-skill"
    assert skill.description == "Say hi"
    assert skill.required_tools == ["bash", "read_file"]
    assert skill.source == "claude"
    assert "Do a thing." in skill.prompt
    assert "Skill: hello-skill" in skill.prompt


def test_parse_claude_skill_with_compatibility(tmp_path):
    md = _write_claude_skill(
        tmp_path,
        "kuiper-skill",
        "name: kuiper-skill\ndescription: x\ncompatibility: Requires kuiper daemon\nallowed-tools: Bash",
        "Body.",
    )
    skill = _parse_claude_skill(md)
    assert skill.compatibility == "Requires kuiper daemon"
    assert "Requires kuiper daemon" in skill.prompt


def test_parse_claude_skill_missing_frontmatter_returns_none(tmp_path):
    md = tmp_path / "no-fm" / "SKILL.md"
    md.parent.mkdir()
    md.write_text("Just a markdown body with no frontmatter.\n", encoding="utf-8")
    assert _parse_claude_skill(md) is None


def test_parse_claude_skill_missing_name_returns_none(tmp_path):
    md = _write_claude_skill(
        tmp_path, "no-name", "description: missing name", "body"
    )
    assert _parse_claude_skill(md) is None


def test_load_claude_skills_from_dir_registers_all(tmp_path):
    _write_claude_skill(
        tmp_path, "alpha", "name: alpha\ndescription: A\nallowed-tools: Read", "Body A"
    )
    _write_claude_skill(
        tmp_path, "beta", "name: beta\ndescription: B\nallowed-tools: Write", "Body B"
    )

    reg = SkillRegistry()
    reg.load_claude_skills_from_dir(str(tmp_path))

    names = {s.name for s in reg.all()}
    assert {"alpha", "beta"} <= names
    assert reg.get("alpha").required_tools == ["read_file"]
    assert reg.get("beta").required_tools == ["write_file"]


def test_load_claude_skills_from_dir_skips_missing_dir(tmp_path):
    reg = SkillRegistry()
    # Should not raise
    reg.load_claude_skills_from_dir(str(tmp_path / "does-not-exist"))
    assert reg.all() == []


def test_load_claude_skills_from_dir_tolerates_bad_file(tmp_path):
    _write_claude_skill(
        tmp_path, "good", "name: good\ndescription: G", "Body"
    )
    bad = tmp_path / "bad" / "SKILL.md"
    bad.parent.mkdir()
    bad.write_text("---\nthis: [is: not: yaml\n---\nbody", encoding="utf-8")

    reg = SkillRegistry()
    reg.load_claude_skills_from_dir(str(tmp_path))

    assert reg.get("good") is not None  # malformed file didn't kill the loader


def test_allowed_tools_accepts_list_form(tmp_path):
    md = _write_claude_skill(
        tmp_path,
        "list-tools",
        "name: list-tools\ndescription: x\nallowed-tools:\n  - Bash\n  - Grep",
        "body",
    )
    skill = _parse_claude_skill(md)
    assert skill.required_tools == ["bash", "grep"]


def test_unknown_tool_passes_through_unchanged(tmp_path):
    md = _write_claude_skill(
        tmp_path,
        "exotic",
        "name: exotic\ndescription: x\nallowed-tools: Bash, MysteryTool",
        "body",
    )
    skill = _parse_claude_skill(md)
    assert skill.required_tools == ["bash", "MysteryTool"]


def test_builtin_skills_have_builtin_source():
    reg = SkillRegistry()
    reg.load_builtin()
    for s in reg.all():
        assert s.source == "builtin"


def test_alias_map_covers_core_tools():
    # Smoke check: the aliases users care about exist.
    for claude_name in ("Read", "Write", "Edit", "Bash", "Glob", "Grep"):
        assert claude_name in CLAUDE_TOOL_ALIASES


# ---------------------------------------------------------------------------
# Plugin skill loader
# ---------------------------------------------------------------------------


def _write_plugin_skill(cache: Path, marketplace: str, plugin: str, version: str,
                        skill_name: str, frontmatter: str, body: str) -> Path:
    skill_dir = cache / marketplace / plugin / version / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    md = skill_dir / "SKILL.md"
    md.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return md


def test_load_plugin_skills_namespaces_by_plugin(tmp_path):
    _write_plugin_skill(
        tmp_path, "marketplace-a", "superpowers", "5.1.0", "test-driven-development",
        "name: test-driven-development\ndescription: TDD\nallowed-tools: Bash", "TDD body",
    )
    _write_plugin_skill(
        tmp_path, "marketplace-a", "code-review", "1.0.0", "code-review",
        "name: code-review\ndescription: Review code", "Review body",
    )

    reg = SkillRegistry()
    reg.load_claude_plugin_skills_from_dir(str(tmp_path))

    names = {s.name for s in reg.all()}
    assert "superpowers:test-driven-development" in names
    assert "code-review:code-review" in names
    assert reg.get("superpowers:test-driven-development").source == "claude-plugin"


def test_load_plugin_skills_missing_dir_is_noop(tmp_path):
    reg = SkillRegistry()
    reg.load_claude_plugin_skills_from_dir(str(tmp_path / "nope"))
    assert reg.all() == []


def test_plugin_skill_does_not_collide_with_user_skill(tmp_path):
    # User-level skill `/frontend-design`
    user_dir = tmp_path / "user-skills" / "frontend-design"
    user_dir.mkdir(parents=True)
    (user_dir / "SKILL.md").write_text(
        "---\nname: frontend-design\ndescription: user version\n---\n\nuser body\n",
        encoding="utf-8",
    )
    # Plugin-level skill — same skill name, different plugin
    _write_plugin_skill(
        tmp_path / "plugin-cache", "marketplace", "frontend-design", "1.0.0",
        "frontend-design",
        "name: frontend-design\ndescription: plugin version", "plugin body",
    )

    reg = SkillRegistry()
    reg.load_claude_skills_from_dir(str(tmp_path / "user-skills"))
    reg.load_claude_plugin_skills_from_dir(str(tmp_path / "plugin-cache"))

    assert reg.get("frontend-design").description == "user version"
    assert reg.get("frontend-design:frontend-design").description == "plugin version"


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------


def test_check_compatibility_no_field_is_ok():
    s = Skill(name="x", description="", prompt="p")
    ok, reason = check_skill_compatibility(s)
    assert ok and reason is None


def test_check_compatibility_no_url_is_ok():
    s = Skill(name="x", description="", prompt="p",
              compatibility="Requires manual setup of X.")
    ok, reason = check_skill_compatibility(s)
    assert ok and reason is None


def test_check_compatibility_url_reachable(tmp_path):
    s = Skill(name="x", description="", prompt="p",
              compatibility="Requires service at http://127.0.0.1:9/mcp")

    class FakeResp:
        status_code = 200

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url): return FakeResp()

    with patch("httpx.Client", FakeClient):
        ok, reason = check_skill_compatibility(s, timeout=0.1)

    assert ok and reason is None


def test_check_compatibility_url_unreachable():
    s = Skill(name="x", description="", prompt="p",
              compatibility="Requires service at http://127.0.0.1:9/mcp")

    class FailingClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url): raise ConnectionError("nope")

    with patch("httpx.Client", FailingClient):
        ok, reason = check_skill_compatibility(s, timeout=0.1)

    assert ok is False
    assert "127.0.0.1:9" in reason


# ---------------------------------------------------------------------------
# Slash-command fallback: unknown /name → skills.get(name).get_prompt(args)
# (cli.py's handle_slash_command). Real commands always win.
# ---------------------------------------------------------------------------

def _slash_deps():
    config = {
        "model": {
            "endpoint": "http://localhost:11434", "name": "test-model",
            "temperature": 0.7, "max_tokens": 4096, "api_key": "", "provider": "ollama",
        },
        "permissions": {"mode": "auto", "always_allow": []},
    }
    reg = SkillRegistry()
    reg.load_builtin()
    return {
        "context": Context(),
        "console": Console(record=True, width=120),
        "config": config,
        "skills": reg,
        "model": MagicMock(spec=ModelClient),
        "permissions": PermissionManager(mode="auto"),
    }


class TestSlashFallbackToSkills:
    def test_commit_skill_expands_to_its_prompt(self):
        deps = _slash_deps()

        result = handle_slash_command(
            "/commit", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )

        assert result is not None
        assert result == deps["skills"].get("commit").get_prompt("")
        assert "create a commit" in result

    def test_unknown_skill_name_still_errors(self):
        deps = _slash_deps()

        result = handle_slash_command(
            "/nope", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )

        assert result is None
        assert "Unknown command: /nope" in deps["console"].export_text()

    def test_review_command_precedence_over_review_skill(self):
        # The /review COMMAND (multi-agent swarm, cli.py) must win over the
        # builtin `review` SKILL of the same name — real commands are checked
        # first in handle_slash_command, the skill fallback only runs for
        # names that matched none of them.
        deps = _slash_deps()
        assert deps["skills"].get("review") is not None  # the skill DOES exist

        result = handle_slash_command(
            "/review", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )

        assert result == "__REVIEW__"  # the command's sentinel, not the skill prompt
        assert "Rate severity" not in (result or "")


# ---------------------------------------------------------------------------
# Lean index: the system prompt carries ONLY name + one-line description per
# skill (capped total), never full skill bodies.
# ---------------------------------------------------------------------------

class TestSkillIndex:
    def test_bridged_skill_body_absent_index_present(self, tmp_path):
        distinctive = "ZZZ_TOTALLY_DISTINCTIVE_BODY_PHRASE_1234"
        long_body = "\n".join([f"Step {i}: do a thing." for i in range(200)])
        long_body += f"\n{distinctive}\n"
        _write_claude_skill(
            tmp_path, "big-bodied-skill",
            "name: big-bodied-skill\ndescription: A skill with a huge body",
            long_body,
        )
        reg = SkillRegistry()
        reg.load_claude_skills_from_dir(str(tmp_path))

        section = build_skill_prompt_section(reg.build_index())

        assert "big-bodied-skill" in section
        assert distinctive not in section
        assert "Step 199: do a thing." not in section

    def test_no_skills_yields_empty_section(self):
        assert build_skill_prompt_section(SkillRegistry().build_index()) == ""

    def test_index_cap_enforced(self):
        reg = SkillRegistry()
        for i in range(200):
            reg.register(Skill(
                name=f"skill-{i:03d}",
                description="A moderately long one-line description " * 3,
                prompt="body " * 500,
            ))

        index = reg.build_index(cap=200)

        assert len(index) <= 200

    def test_default_cap_is_1500(self):
        reg = SkillRegistry()
        for i in range(200):
            reg.register(Skill(
                name=f"skill-{i:03d}",
                description="A moderately long one-line description " * 3,
                prompt="body " * 500,
            ))

        index = reg.build_index()

        assert len(index) <= 1500

    def test_builtins_listed_before_bridged_skills(self):
        reg = SkillRegistry()
        # Alphabetically this would sort BEFORE any builtin ("aaa" < "commit"),
        # but builtins must still come first.
        reg.register(Skill(name="aaa-bridged", description="d", prompt="p", source="claude"))
        reg.register(Skill(name="zzz-builtin", description="d", prompt="p", source="builtin"))

        index = reg.build_index()

        assert index.index("zzz-builtin") < index.index("aaa-bridged")

    def test_index_order_independent_of_registration_order(self):
        reg1 = SkillRegistry()
        reg1.register(Skill(name="beta", description="B thing", prompt="body"))
        reg1.register(Skill(name="alpha", description="A thing", prompt="body"))

        reg2 = SkillRegistry()
        reg2.register(Skill(name="alpha", description="A thing", prompt="body"))
        reg2.register(Skill(name="beta", description="B thing", prompt="body"))

        assert reg1.build_index() == reg2.build_index()

    def test_names_capped_and_deterministic(self):
        reg = SkillRegistry()
        for i in range(80):
            reg.register(Skill(name=f"s{i:03d}", description="d", prompt="p"))

        names = reg.names(limit=10)

        assert len(names) == 10
        assert names == sorted(names)  # name-sorted (all same source here)

    def test_whitespace_only_description_does_not_crash_index(self):
        # Review fix 1: a whitespace-only description ("   ") is truthy, but
        # strip().splitlines() on it is [] — indexing [0] raised IndexError,
        # and build_skill_prompt_section is unguarded at cli.py startup, so
        # ONE bad bridged SKILL.md killed run_interactive.
        reg = SkillRegistry()
        reg.register(Skill(name="blank-desc", description="   ", prompt="p"))
        reg.register(Skill(name="newline-desc", description=" \n ", prompt="p"))

        index = reg.build_index()  # must not raise

        assert "/blank-desc" in index
        assert "/newline-desc" in index
        # Empty one-liner → bare name, no dangling " — " separator.
        assert "/blank-desc —" not in index

    def test_index_excludes_command_colliding_names(self):
        # Review fix 2: the builtin `review` SKILL must not be advertised to
        # the model — the /review COMMAND (multi-agent swarm) supersedes it at
        # the CLI, so an index entry would teach the model a name that
        # resolves to the lesser skill through auto-expand.
        from spark_code.ui.input import RESERVED_COMMAND_NAMES
        reg = SkillRegistry()
        reg.load_builtin()

        index = reg.build_index(exclude=RESERVED_COMMAND_NAMES)

        assert "/review" not in index
        assert "/commit" in index  # non-colliding builtins still listed

    def test_reserved_command_names_derived_from_builtin_commands(self):
        from spark_code.ui.input import _BUILTIN_COMMANDS, RESERVED_COMMAND_NAMES
        assert "review" in RESERVED_COMMAND_NAMES
        assert "plan" in RESERVED_COMMAND_NAMES
        # Bare first tokens only — subcommand entries ("/plan show") collapse
        # into their command, they don't reserve their argument words.
        assert "show" not in RESERVED_COMMAND_NAMES
        assert len(RESERVED_COMMAND_NAMES) <= len(_BUILTIN_COMMANDS)


# ---------------------------------------------------------------------------
# Trigger hint + model-reply auto-expand (agent.py) — a FINAL answer that is
# SOLELY a "/<skill-name> [args]" line expands into the skill prompt and
# continues the turn ONCE; unknown skill or non-solo slash line → left as
# normal text; a guard flag prevents a second expansion in the same turn.
# ---------------------------------------------------------------------------

class _ScriptedModel:
    """Yields one pre-scripted list of chunks per call to .chat()."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def chat(self, **kwargs):
        chunks = self.responses[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk

    async def close(self):
        pass


def _make_agent(model, skills=None):
    return Agent(
        model=model,
        context=Context(),
        tools=ToolRegistry(),
        permissions=PermissionManager(mode="trust"),
        console=Console(file=io.StringIO(), force_terminal=True),
        skills=skills,
    )


class TestSkillAutoExpand:
    async def test_solo_slash_line_expands_once(self):
        reg = SkillRegistry()
        reg.load_builtin()
        model = _ScriptedModel([
            [{"type": "text", "content": "/codesearch foo"}, {"type": "done", "usage": {}}],
            [{"type": "text", "content": "Found it in bar.py"}, {"type": "done", "usage": {}}],
        ])
        agent = _make_agent(model, skills=reg)

        result = await agent.run("please look for foo")

        assert "Found it in bar.py" in result
        assert model.calls == 2
        assert agent._skill_auto_expanded is True
        user_msgs = [m["content"] for m in agent.context.messages if m["role"] == "user"]
        assert any("Search the codebase thoroughly" in c for c in user_msgs)
        assert any("foo" in c for c in user_msgs)  # args carried through

    async def test_guard_prevents_second_expansion_same_turn(self):
        reg = SkillRegistry()
        reg.load_builtin()
        # Both rounds reply with a bare slash line — only the FIRST may expand.
        model = _ScriptedModel([
            [{"type": "text", "content": "/codesearch foo"}, {"type": "done", "usage": {}}],
            [{"type": "text", "content": "/codesearch bar"}, {"type": "done", "usage": {}}],
        ])
        agent = _make_agent(model, skills=reg)

        result = await agent.run("please look for foo")

        assert result.endswith("/codesearch bar")
        assert model.calls == 2  # not a third round — the guard stopped it
        user_msgs = [m for m in agent.context.messages if m["role"] == "user"]
        # Original prompt + exactly ONE expansion — the second slash line did
        # NOT synthesize another user message.
        assert len(user_msgs) == 2

    async def test_unknown_skill_name_treated_as_normal_text(self):
        reg = SkillRegistry()
        reg.load_builtin()
        model = _ScriptedModel([
            [{"type": "text", "content": "/nope-not-a-skill do something"},
             {"type": "done", "usage": {}}],
        ])
        agent = _make_agent(model, skills=reg)

        result = await agent.run("hello")

        assert result == "/nope-not-a-skill do something"
        assert model.calls == 1
        assert agent._skill_auto_expanded is False

    async def test_command_colliding_name_not_auto_expanded(self):
        # Review fix 2 (model surface): a bare "/review" reply must NOT expand
        # into the builtin `review` SKILL — at the CLI that name is the
        # multi-agent swarm COMMAND, and the model surface must not resolve
        # the same name to something lesser. Treated as normal text.
        reg = SkillRegistry()
        reg.load_builtin()
        assert reg.get("review") is not None  # the colliding skill DOES exist
        model = _ScriptedModel([
            [{"type": "text", "content": "/review"}, {"type": "done", "usage": {}}],
        ])
        agent = _make_agent(model, skills=reg)

        result = await agent.run("review my changes")

        assert result == "/review"
        assert model.calls == 1
        assert agent._skill_auto_expanded is False

    async def test_slash_line_among_other_text_does_not_expand(self):
        reg = SkillRegistry()
        reg.load_builtin()
        model = _ScriptedModel([
            [{"type": "text", "content": "Sure, I'll do that.\n/commit"},
             {"type": "done", "usage": {}}],
        ])
        agent = _make_agent(model, skills=reg)

        result = await agent.run("commit please")

        assert result == "Sure, I'll do that.\n/commit"
        assert model.calls == 1
        assert agent._skill_auto_expanded is False

    async def test_no_skills_configured_leaves_slash_reply_as_text(self):
        model = _ScriptedModel([
            [{"type": "text", "content": "/commit"}, {"type": "done", "usage": {}}],
        ])
        agent = _make_agent(model, skills=None)

        result = await agent.run("commit please")

        assert result == "/commit"
        assert model.calls == 1
