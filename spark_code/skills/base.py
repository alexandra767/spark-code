"""Skill system — slash commands that inject specialized prompts."""

import os
import re
from pathlib import Path

import yaml

# Claude Code -> Spark Code tool name aliases. Claude skills reference tools
# like `Bash`/`Read`/`Write`; Spark's registry uses snake_case equivalents.
CLAUDE_TOOL_ALIASES = {
    "Read": "read_file",
    "Write": "write_file",
    "Edit": "edit_file",
    "Bash": "bash",
    "Glob": "glob",
    "Grep": "grep",
    "LS": "list_dir",
    "WebSearch": "web_search",
    "WebFetch": "web_fetch",
    "TodoWrite": "todo_write",
}


class Skill:
    """A skill is a pre-written prompt triggered by a slash command."""

    def __init__(self, name: str, description: str, prompt: str,
                 required_tools: list[str] | None = None,
                 requires_args: bool = False,
                 compatibility: str | None = None,
                 source: str = "builtin"):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.required_tools = required_tools or []
        self.requires_args = requires_args
        self.compatibility = compatibility
        self.source = source

    def get_prompt(self, args: str = "") -> str:
        """Get the full prompt, optionally with user arguments."""
        if args:
            return f"{self.prompt}\n\nUser context: {args}"
        return self.prompt


class SkillRegistry:
    """Loads and manages skills."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Get skill by name (with or without / prefix)."""
        name = name.lstrip("/")
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def _ordered(self) -> list[Skill]:
        """Skills in a STABLE, deterministic order: builtins first (most
        reliable / always available), then everything else — both groups
        sorted by name.

        This does NOT preserve registration order. Registration order for
        bridged skills comes from ``Path.glob()`` (load_from_dir /
        load_claude_skills_from_dir / load_claude_plugin_skills_from_dir),
        which is filesystem-iteration order, not sorted — using it directly
        would make the skill index (and therefore the STATIC system prompt
        prefix a server prefix-caches across turns) vary run to run even when
        the skill set on disk hasn't changed. Sorting here is cheap and makes
        the index cache-friendly and reproducible.
        """
        builtins = sorted(
            (s for s in self._skills.values() if s.source == "builtin"),
            key=lambda s: s.name,
        )
        others = sorted(
            (s for s in self._skills.values() if s.source != "builtin"),
            key=lambda s: s.name,
        )
        return builtins + others

    def names(self, limit: int = 50) -> list[str]:
        """Skill names for the input-autocomplete completer.

        Capped to a reasonable count — a large bridge from
        ``~/.claude/skills`` + installed plugins could otherwise dump dozens
        of entries into the completion menu. Uses the same builtins-first,
        name-sorted ordering as ``build_index()`` so the subset that survives
        the cap is deterministic rather than whatever order the filesystem
        happened to glob() them in.
        """
        return [f"/{s.name}" for s in self._ordered()[:limit]]

    def build_index(self, cap: int = 1500,
                    exclude: frozenset[str] | set[str] | None = None) -> str:
        """Compact skill index for the system prompt: one line per skill,
        ``/<name> — <one-line description>``, builtins first then the rest,
        both sorted by name (see ``_ordered``) for a stable, cache-friendly
        result.

        Capped to ``cap`` total characters — lines are added in order only
        while they still fit; once the next line would overflow, the list is
        truncated (with a trailing "more skills" marker, itself subject to
        the same cap) rather than exceeding the budget. This index carries
        ONLY names + one-liners; full skill bodies (``Skill.prompt``) never
        appear here — they enter context only when a skill is actually
        invoked, via ``Skill.get_prompt()``.

        ``exclude``: skill names to omit entirely — cli.py passes the real
        commands' RESERVED_COMMAND_NAMES (ui/input.py) so a skill sharing a
        name with a command (the builtin `review` skill vs. the /review
        swarm command) is never advertised to the model: at the CLI that
        name resolves to the COMMAND, so an index entry would teach the
        model a name that expands to something else.
        """
        lines: list[str] = []
        truncated = False
        for skill in self._ordered():
            if exclude and skill.name in exclude:
                continue
            # Test the STRIPPED text, not raw description truthiness — a
            # whitespace-only description ("   ") is truthy but its
            # strip().splitlines() is [], so [0] raised IndexError (and one
            # bad bridged SKILL.md killed startup via the unguarded
            # build_skill_prompt_section call in run_interactive).
            stripped_desc = (skill.description or "").strip()
            desc = stripped_desc.splitlines()[0] if stripped_desc else ""
            line = f"/{skill.name} — {desc}" if desc else f"/{skill.name}"
            candidate = lines + [line]
            if len("\n".join(candidate)) > cap:
                truncated = True
                break
            lines = candidate

        if truncated:
            candidate = lines + ["… more skills available (see /help)"]
            if len("\n".join(candidate)) <= cap:
                lines = candidate

        return "\n".join(lines)

    def load_builtin(self):
        """Load built-in skills."""
        for skill in BUILTIN_SKILLS:
            self.register(skill)

    def load_from_dir(self, directory: str):
        """Load custom skills from a directory of YAML files."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return

        for f in dir_path.glob("*.yaml"):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh)
                if data and "name" in data and "prompt" in data:
                    skill = Skill(
                        name=data["name"],
                        description=data.get("description", ""),
                        prompt=data["prompt"],
                        required_tools=data.get("required_tools", []),
                    )
                    self.register(skill)
            except Exception:
                pass  # Skip invalid skill files

    def load_claude_skills_from_dir(self, directory: str):
        """Load Claude Code skills (each in `<dir>/<name>/SKILL.md` with YAML frontmatter)."""
        dir_path = Path(directory)
        if not dir_path.exists():
            return

        for skill_md in dir_path.glob("*/SKILL.md"):
            try:
                skill = _parse_claude_skill(skill_md)
                if skill is not None:
                    self.register(skill)
            except Exception:
                pass  # Skip malformed Claude skills rather than crashing startup

    def load_claude_plugin_skills_from_dir(self, plugin_cache_dir: str):
        """Load installed Claude Code plugin skills.

        Layout: <cache>/<marketplace>/<plugin>/<version>/skills/<skill>/SKILL.md.
        Skills get namespaced as `<plugin>:<skill>` so they don't collide with
        user-level skills of the same name (matches Claude's own convention).
        """
        cache = Path(plugin_cache_dir)
        if not cache.exists():
            return

        for skill_md in cache.glob("*/*/*/skills/*/SKILL.md"):
            try:
                # parts: [..., marketplace, plugin, version, "skills", skillname, "SKILL.md"]
                plugin_name = skill_md.parts[-5]
                skill = _parse_claude_skill(skill_md, name_prefix=f"{plugin_name}:")
                if skill is not None:
                    skill.source = "claude-plugin"
                    self.register(skill)
            except Exception:
                pass

    def load_all(self):
        """Load built-in + global + project skills + Claude Code skills + plugin skills."""
        self.load_builtin()
        self.load_from_dir(os.path.expanduser("~/.spark/skills"))
        self.load_from_dir(".spark/skills")
        # Bridge: surface Claude Code's user-level skills (~/.claude/skills/<name>/SKILL.md)
        # so commands like /app-gap-research work from Spark too.
        self.load_claude_skills_from_dir(os.path.expanduser("~/.claude/skills"))
        # Bridge: surface installed Claude Code plugin skills as /plugin:skill.
        self.load_claude_plugin_skills_from_dir(os.path.expanduser("~/.claude/plugins/cache"))


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_URL_RE = re.compile(r"https?://[^\s)>'\"]+")


def _parse_claude_skill(path: Path, name_prefix: str = "") -> Skill | None:
    """Parse a Claude Code SKILL.md (YAML frontmatter + markdown body) into a Skill.

    `name_prefix` is prepended to the skill name (used for plugin namespacing
    e.g. `superpowers:test-driven-development`).
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None

    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    name = meta.get("name")
    if not name or not body:
        return None
    full_name = f"{name_prefix}{name}"

    raw_tools = meta.get("allowed-tools") or meta.get("required_tools") or []
    if isinstance(raw_tools, str):
        raw_tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
    required_tools = [CLAUDE_TOOL_ALIASES.get(t, t) for t in raw_tools]

    compatibility = meta.get("compatibility")
    header_parts = [f"# Skill: {full_name} (Claude Code)"]
    if compatibility:
        header_parts.append(f"Compatibility: {compatibility}")
    if raw_tools:
        mapped = ", ".join(
            f"{t}->{CLAUDE_TOOL_ALIASES[t]}" for t in raw_tools if t in CLAUDE_TOOL_ALIASES
        )
        if mapped:
            header_parts.append(f"Tool name mapping in Spark Code: {mapped}")
    prompt = "\n".join(header_parts) + "\n\n" + body

    return Skill(
        name=full_name,
        description=meta.get("description", ""),
        prompt=prompt,
        required_tools=required_tools,
        compatibility=compatibility,
        source="claude",
    )


def check_skill_compatibility(skill: Skill, timeout: float = 2.0) -> tuple[bool, str | None]:
    """Pre-flight a skill's compatibility requirement.

    If the `compatibility` text mentions an http(s) URL, GET it with a short
    timeout. Return (ok, reason_if_not_ok). Skills without a URL we can probe
    are always reported OK — they'll explain themselves to the model in the prompt.
    """
    if not skill.compatibility:
        return True, None

    urls = _URL_RE.findall(skill.compatibility)
    if not urls:
        return True, None

    # Skills often point to an MCP/health endpoint. Try the literal URL first;
    # if it ends in /mcp, also try /health on the same host (kuiper convention).
    candidates = []
    for u in urls:
        u = u.rstrip("/").rstrip(",;.)")
        candidates.append(u)
        if u.endswith("/mcp"):
            candidates.append(u[: -len("/mcp")] + "/health")

    try:
        import httpx
    except ImportError:
        return True, None  # don't block if httpx missing — skill can self-report

    for url in candidates:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url)
            if resp.status_code < 500:
                return True, None
        except Exception:
            continue

    return False, (
        f"Compatibility check failed: could not reach {urls[0]} "
        f"({skill.compatibility})"
    )


# Built-in skills
BUILTIN_SKILLS = [
    Skill(
        name="commit",
        description="Generate a git commit message from current changes",
        prompt="""Look at the current git changes and create a commit.

Steps:
1. Run `git status` to see what files changed
2. Run `git diff` to see staged and unstaged changes
3. Run `git log --oneline -5` to see recent commit style
4. Analyze the changes and write a concise, meaningful commit message
5. Show me the proposed commit message
6. If I approve, stage the relevant files and commit

Rules:
- Focus on WHY, not WHAT
- Keep the first line under 72 characters
- Use conventional commit format if the project uses it
- Don't commit .env files or secrets""",
        required_tools=["bash"],
    ),
    Skill(
        name="review",
        description="Review code changes for bugs, security issues, and improvements",
        prompt="""Review the recent code changes for quality issues.

Steps:
1. Run `git diff` to see all changes
2. If there are staged changes, also run `git diff --cached`
3. Analyze each changed file for:
   - Bugs and logic errors
   - Security vulnerabilities (injection, XSS, etc.)
   - Performance issues
   - Code style and readability
   - Missing error handling
   - Missing tests
4. Provide specific, actionable feedback with line references
5. Rate severity: critical / warning / suggestion

Be thorough but practical. Focus on issues that matter.""",
        required_tools=["bash", "read_file"],
    ),
    Skill(
        name="test",
        description="Find and run the project's test suite",
        prompt="""Find and run the project's tests.

Steps:
1. Look for test configuration: package.json (jest/vitest), pytest.ini, Cargo.toml, etc.
2. Identify the test command
3. Run the tests
4. If tests fail:
   - Read the failing test file
   - Read the source file being tested
   - Diagnose the issue
   - Suggest or apply a fix
   - Re-run tests to verify
5. Report: total passed, failed, skipped""",
        required_tools=["bash", "glob", "read_file"],
    ),
    Skill(
        name="explain",
        description="Explain the current project structure and architecture",
        prompt="""Explain this project's structure and architecture.

Steps:
1. List the top-level directory
2. Read key files: README, package.json/pyproject.toml/Cargo.toml, main entry point
3. Identify the framework/stack
4. Map out the directory structure
5. Explain:
   - What the project does
   - Tech stack and key dependencies
   - Directory layout and what each folder contains
   - Entry points and main flow
   - Key patterns used (MVC, etc.)

Keep it concise and practical.""",
        required_tools=["list_dir", "read_file", "glob"],
    ),
    Skill(
        name="fix",
        description="Diagnose and fix an error",
        prompt="""Diagnose and fix the error the user is experiencing.

Steps:
1. Ask the user to describe the error (or read from their message)
2. Search for relevant files using grep/glob
3. Read the relevant source files
4. Identify the root cause
5. Propose a fix with explanation
6. Apply the fix (with permission)
7. Run tests or verify the fix works

Think systematically: reproduce → diagnose → fix → verify.""",
        required_tools=["bash", "read_file", "edit_file", "grep"],
    ),
    Skill(
        name="refactor",
        description="Suggest and apply code refactoring",
        prompt="""Refactor the code the user points to.

Steps:
1. Read the file(s) to refactor
2. Identify issues:
   - Code duplication
   - Long functions that should be split
   - Poor naming
   - Missing abstractions
   - Complex conditionals
3. Propose specific refactoring steps
4. Apply changes (with permission)
5. Run tests to ensure nothing broke

Keep changes focused. Don't refactor everything at once.""",
        required_tools=["read_file", "edit_file", "bash"],
    ),
    # Named `codesearch` (not `search`) so it doesn't collide with the built-in
    # `/search` command (which searches past SESSIONS, not the codebase).
    Skill(
        name="codesearch",
        description="Deep search through the codebase",
        prompt="""Search the codebase thoroughly for what the user is looking for.

Steps:
1. Use glob to find relevant file types
2. Use grep to search for patterns, function names, imports
3. Read the most relevant files
4. Trace the code flow if needed
5. Summarize findings with file paths and line numbers

Be thorough — check multiple naming conventions, related files, tests.""",
        required_tools=["glob", "grep", "read_file"],
    ),
    # Note: `/continue` and `/clean` are handled directly as built-in commands
    # in cli.py (checkpoint resume / delete session files). They intentionally
    # are NOT skills here — a skill of the same name would be unreachable and
    # would duplicate the command in /help.
]
