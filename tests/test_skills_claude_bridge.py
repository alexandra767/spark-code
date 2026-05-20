"""Tests for the Claude Code skill loader bridge in skills/base.py."""

from pathlib import Path
from unittest.mock import patch

from spark_code.skills.base import (
    CLAUDE_TOOL_ALIASES,
    Skill,
    SkillRegistry,
    _parse_claude_skill,
    check_skill_compatibility,
)


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
