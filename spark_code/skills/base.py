"""Skill system — slash commands that inject specialized prompts."""

import os
from pathlib import Path

import yaml


class Skill:
    """A skill is a pre-written prompt triggered by a slash command."""

    def __init__(self, name: str, description: str, prompt: str,
                 required_tools: list[str] | None = None,
                 requires_args: bool = False):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.required_tools = required_tools or []
        self.requires_args = requires_args

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

    def names(self) -> list[str]:
        return [f"/{name}" for name in self._skills.keys()]

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

    def load_all(self):
        """Load built-in + global + project skills."""
        self.load_builtin()
        self.load_from_dir(os.path.expanduser("~/.spark/skills"))
        self.load_from_dir(".spark/skills")


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
    Skill(
        name="search",
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
    Skill(
        name="continue",
        description="Resume from the last checkpoint after hitting the tool round limit",
        prompt="Load the latest checkpoint and continue where you left off.",
        required_tools=[],
    ),
    Skill(
        name="clean",
        description="Delete files created during this session",
        prompt="List files created during this session and offer to delete them.",
        required_tools=[],
    ),
]
