"""Smart project detection — scan for project markers and frameworks."""

import json
import os
import re


def detect_project_type(directory: str = ".") -> str:
    """Detect project type and frameworks from directory markers.

    Returns a one-liner like "Python + pytest + FastAPI project"
    or empty string if nothing detected.
    """
    markers: list[str] = []
    frameworks: list[str] = []

    def exists(name: str) -> bool:
        return os.path.exists(os.path.join(directory, name))

    def read_file(name: str) -> str:
        path = os.path.join(directory, name)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read(8192)
        except OSError:
            return ""

    # Python
    if exists("pyproject.toml") or exists("setup.py") or exists("setup.cfg"):
        markers.append("Python")
        content = read_file("pyproject.toml")
        if "pytest" in content or exists("pytest.ini") or exists("conftest.py"):
            frameworks.append("pytest")
        if "fastapi" in content.lower():
            frameworks.append("FastAPI")
        if "django" in content.lower():
            frameworks.append("Django")
        if "flask" in content.lower():
            frameworks.append("Flask")
        if "ruff" in content:
            frameworks.append("ruff")
        if "mypy" in content:
            frameworks.append("mypy")
    elif exists("requirements.txt"):
        markers.append("Python")
        content = read_file("requirements.txt")
        if "pytest" in content:
            frameworks.append("pytest")
        if "fastapi" in content.lower():
            frameworks.append("FastAPI")
        if "django" in content.lower():
            frameworks.append("Django")
        if "flask" in content.lower():
            frameworks.append("Flask")

    # JavaScript / TypeScript
    if exists("package.json"):
        if "Python" not in markers:
            markers.append("JavaScript")
        pkg = read_file("package.json")
        try:
            pkg_data = json.loads(pkg)
        except (json.JSONDecodeError, ValueError):
            pkg_data = {}
        all_deps = {
            **pkg_data.get("dependencies", {}),
            **pkg_data.get("devDependencies", {}),
        }
        if exists("tsconfig.json") or "typescript" in all_deps:
            # Replace JavaScript with TypeScript
            if "JavaScript" in markers:
                markers[markers.index("JavaScript")] = "TypeScript"
            else:
                markers.append("TypeScript")
        if "react" in all_deps:
            frameworks.append("React")
        if "next" in all_deps:
            frameworks.append("Next.js")
        if "vue" in all_deps:
            frameworks.append("Vue")
        if "svelte" in all_deps:
            frameworks.append("Svelte")
        if "express" in all_deps:
            frameworks.append("Express")
        if "jest" in all_deps:
            frameworks.append("Jest")
        if "vitest" in all_deps:
            frameworks.append("Vitest")
        if "tailwindcss" in all_deps:
            frameworks.append("Tailwind")

    # Rust
    if exists("Cargo.toml"):
        markers.append("Rust")
        content = read_file("Cargo.toml")
        if "tokio" in content:
            frameworks.append("Tokio")
        if "actix" in content:
            frameworks.append("Actix")
        if "axum" in content:
            frameworks.append("Axum")

    # Go
    if exists("go.mod"):
        markers.append("Go")
        content = read_file("go.mod")
        if "gin-gonic" in content:
            frameworks.append("Gin")
        if "echo" in content:
            frameworks.append("Echo")

    # Swift
    if exists("Package.swift"):
        markers.append("Swift")
    elif any(
        f.endswith(".xcodeproj") or f.endswith(".xcworkspace")
        for f in os.listdir(directory)
        if not f.startswith(".")
    ):
        markers.append("Swift/Xcode")

    # Java / Kotlin
    if exists("build.gradle") or exists("build.gradle.kts"):
        if exists("build.gradle.kts"):
            markers.append("Kotlin")
        else:
            markers.append("Java")
        content = read_file("build.gradle.kts") or read_file("build.gradle")
        if "compose" in content.lower():
            frameworks.append("Compose")
        if "spring" in content.lower():
            frameworks.append("Spring")
    elif exists("pom.xml"):
        markers.append("Java")
        content = read_file("pom.xml")
        if "spring" in content.lower():
            frameworks.append("Spring")

    # Docker
    if exists("Dockerfile") or exists("docker-compose.yml") or exists("docker-compose.yaml"):
        frameworks.append("Docker")

    if not markers:
        return ""

    parts = markers.copy()
    if frameworks:
        parts.extend(frameworks)
    return " + ".join(parts) + " project"


# Known test-command literal patterns (Task 6: verification habit). Checked
# against CLAUDE.md/SPARK.md instructions text FIRST — an explicit
# human-authored testing instruction always wins over inference from project
# markers. Order is the priority when more than one pattern appears in the
# instructions text.
_KNOWN_TEST_PATTERNS = ("pytest", "npm test", "xcodebuild", "cargo test", "go test")


def _contains_word(text: str, phrase: str) -> bool:
    """True iff ``phrase`` appears in ``text`` at word boundaries.

    A plain substring check would let "pytest" match inside an unrelated
    filename like "mypytest.py" or "test_pytest_helpers.py" — this anchors the
    match so it only fires on the literal command/word, not a substring of a
    longer identifier.
    """
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def _has_makefile_test_target(cwd: str) -> bool:
    """True iff ``cwd``/Makefile exists and defines a ``test:`` target.

    Matched with ``^test:`` (anchored to the start of a line, ``re.MULTILINE``)
    rather than a substring check — a substring check would false-match
    unrelated targets/text that merely contain "test", e.g. a ``pytest:`` or
    ``unittest:`` target, or ``.PHONY: test`` (which doesn't start the line
    with "test:"). Read errors (missing file, permissions) are treated as "no
    Makefile test target" rather than raised.
    """
    try:
        with open(os.path.join(cwd, "Makefile"), encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return False
    return re.search(r"^test:", content, re.MULTILINE) is not None


def detect_test_command(cwd: str = ".", instructions_text: str = "") -> str | None:
    """Detect the project's test command, or ``None`` if nothing is detectable.

    Checked in order (Task 6,
    docs/superpowers/plans/2026-07-07-phase3-polish-ide.md):
      1. ``instructions_text`` (already-loaded CLAUDE.md/SPARK.md content, see
         ``spark_code/instructions.py``) for a literal known pattern
         (``pytest``, ``npm test``, ``xcodebuild``, ``cargo test``, ``go
         test``) — an explicit project instruction overrides inference.
      2. A ``Makefile`` in ``cwd`` with a ``test:`` target -> ``make test``.
         This wraps whatever the project actually needs (including a full
         ``xcodebuild`` invocation with scheme/destination) so it beats every
         type-specific guess below.
      3. Project-type markers in ``cwd`` (pyproject.toml/setup.py/setup.cfg/
         pytest.ini/conftest.py -> pytest; package.json -> npm test;
         Cargo.toml -> cargo test; go.mod -> go test; Package.swift ->
         ``swift test``, which is runnable as-is).

    Bare Xcode projects (an .xcodeproj/.xcworkspace with no Makefile test
    target and no Package.swift) intentionally return ``None``: bare
    ``xcodebuild`` needs -scheme/-destination arguments to run at all, so
    nudging with it would be worse than not nudging.

    ``None`` means the caller should treat the verification-nudge feature as
    silently off (no test command to nudge about) rather than guessing.
    """
    if instructions_text:
        for pattern in _KNOWN_TEST_PATTERNS:
            if _contains_word(instructions_text, pattern):
                return pattern

    def exists(name: str) -> bool:
        return os.path.exists(os.path.join(cwd, name))

    if _has_makefile_test_target(cwd):
        return "make test"

    if (exists("pyproject.toml") or exists("setup.py") or exists("setup.cfg")
            or exists("pytest.ini") or exists("conftest.py")):
        return "pytest"
    if exists("package.json"):
        return "npm test"
    if exists("Cargo.toml"):
        return "cargo test"
    if exists("go.mod"):
        return "go test"
    if exists("Package.swift"):
        return "swift test"
    return None
