"""Tests for smart project detection."""

import json
import os
import shutil
import tempfile

import pytest

from spark_code.project_detect import detect_project_type, detect_test_command


@pytest.fixture
def project_dir():
    d = tempfile.mkdtemp(prefix="spark_detect_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_empty_directory(project_dir):
    assert detect_project_type(project_dir) == ""


def test_python_pyproject(project_dir):
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n')
    result = detect_project_type(project_dir)
    assert "Python" in result
    assert "project" in result


def test_python_with_pytest(project_dir):
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n[tool.pytest]\n')
    result = detect_project_type(project_dir)
    assert "Python" in result
    assert "pytest" in result


def test_python_with_fastapi(project_dir):
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('dependencies = ["fastapi"]\n')
    result = detect_project_type(project_dir)
    assert "FastAPI" in result


def test_python_requirements(project_dir):
    with open(os.path.join(project_dir, "requirements.txt"), "w") as f:
        f.write("flask\npytest\n")
    result = detect_project_type(project_dir)
    assert "Python" in result
    assert "Flask" in result
    assert "pytest" in result


def test_javascript_package_json(project_dir):
    pkg = {"name": "myapp", "dependencies": {}}
    with open(os.path.join(project_dir, "package.json"), "w") as f:
        json.dump(pkg, f)
    result = detect_project_type(project_dir)
    assert "JavaScript" in result


def test_typescript_detection(project_dir):
    pkg = {"name": "myapp", "devDependencies": {"typescript": "^5.0"}}
    with open(os.path.join(project_dir, "package.json"), "w") as f:
        json.dump(pkg, f)
    with open(os.path.join(project_dir, "tsconfig.json"), "w") as f:
        f.write("{}")
    result = detect_project_type(project_dir)
    assert "TypeScript" in result
    assert "JavaScript" not in result


def test_react_detection(project_dir):
    pkg = {"name": "myapp", "dependencies": {"react": "^18.0"}}
    with open(os.path.join(project_dir, "package.json"), "w") as f:
        json.dump(pkg, f)
    result = detect_project_type(project_dir)
    assert "React" in result


def test_nextjs_detection(project_dir):
    pkg = {"name": "myapp", "dependencies": {"next": "^14.0", "react": "^18"}}
    with open(os.path.join(project_dir, "package.json"), "w") as f:
        json.dump(pkg, f)
    result = detect_project_type(project_dir)
    assert "Next.js" in result
    assert "React" in result


def test_rust_detection(project_dir):
    with open(os.path.join(project_dir, "Cargo.toml"), "w") as f:
        f.write('[package]\nname = "myapp"\n')
    result = detect_project_type(project_dir)
    assert "Rust" in result


def test_rust_with_tokio(project_dir):
    with open(os.path.join(project_dir, "Cargo.toml"), "w") as f:
        f.write('[dependencies]\ntokio = "1"\n')
    result = detect_project_type(project_dir)
    assert "Rust" in result
    assert "Tokio" in result


def test_go_detection(project_dir):
    with open(os.path.join(project_dir, "go.mod"), "w") as f:
        f.write("module example.com/myapp\n")
    result = detect_project_type(project_dir)
    assert "Go" in result


def test_docker_detection(project_dir):
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n')
    with open(os.path.join(project_dir, "Dockerfile"), "w") as f:
        f.write("FROM python:3.12\n")
    result = detect_project_type(project_dir)
    assert "Docker" in result


def test_kotlin_gradle(project_dir):
    with open(os.path.join(project_dir, "build.gradle.kts"), "w") as f:
        f.write('plugins { id("org.jetbrains.kotlin.jvm") }\n')
    result = detect_project_type(project_dir)
    assert "Kotlin" in result


def test_java_maven(project_dir):
    with open(os.path.join(project_dir, "pom.xml"), "w") as f:
        f.write('<project><groupId>com.example</groupId></project>\n')
    result = detect_project_type(project_dir)
    assert "Java" in result


def test_swift_package(project_dir):
    with open(os.path.join(project_dir, "Package.swift"), "w") as f:
        f.write("// swift-tools-version:5.9\n")
    result = detect_project_type(project_dir)
    assert "Swift" in result


def test_multiple_markers(project_dir):
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n[tool.ruff]\n')
    pkg = {"name": "frontend", "dependencies": {"react": "^18"}}
    with open(os.path.join(project_dir, "package.json"), "w") as f:
        json.dump(pkg, f)
    result = detect_project_type(project_dir)
    assert "Python" in result
    assert "React" in result


# ---------------------------------------------------------------------------
# detect_test_command (Task 6: verification habit)
# ---------------------------------------------------------------------------


def test_detect_test_command_empty_directory(project_dir):
    assert detect_test_command(project_dir, "") is None


def test_detect_test_command_pyproject(project_dir):
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n')
    assert detect_test_command(project_dir, "") == "pytest"


def test_detect_test_command_package_json(project_dir):
    with open(os.path.join(project_dir, "package.json"), "w") as f:
        f.write('{"name": "myapp"}')
    assert detect_test_command(project_dir, "") == "npm test"


def test_detect_test_command_cargo(project_dir):
    with open(os.path.join(project_dir, "Cargo.toml"), "w") as f:
        f.write('[package]\nname = "myapp"\n')
    assert detect_test_command(project_dir, "") == "cargo test"


def test_detect_test_command_go(project_dir):
    with open(os.path.join(project_dir, "go.mod"), "w") as f:
        f.write("module example.com/myapp\n")
    assert detect_test_command(project_dir, "") == "go test"


def test_detect_test_command_swift_package(project_dir):
    """Package.swift (SwiftPM) -> 'swift test', which is runnable as-is —
    changed from bare 'xcodebuild' per Task 6,
    docs/superpowers/plans/2026-07-07-phase3-polish-ide.md (bare xcodebuild
    needs -scheme/-destination args it doesn't have here)."""
    with open(os.path.join(project_dir, "Package.swift"), "w") as f:
        f.write("// swift-tools-version:5.9\n")
    assert detect_test_command(project_dir, "") == "swift test"


def test_detect_test_command_xcodeproj_bare_returns_none(project_dir):
    """A bare .xcodeproj (no Makefile test target, no Package.swift) -> None,
    not bare 'xcodebuild' — per Task 6, docs/superpowers/plans/
    2026-07-07-phase3-polish-ide.md, nudging an unrunnable command (bare
    xcodebuild has no -scheme/-destination) is worse than no nudge."""
    os.mkdir(os.path.join(project_dir, "MyApp.xcodeproj"))
    assert detect_test_command(project_dir, "") is None


def test_detect_test_command_package_swift_with_xcodeproj_prefers_swift_test(project_dir):
    """A SwiftPM package that also ships an .xcodeproj wrapper still resolves
    to 'swift test' — Package.swift is checked before the bare-xcodeproj
    fallthrough, so SwiftPM wins."""
    with open(os.path.join(project_dir, "Package.swift"), "w") as f:
        f.write("// swift-tools-version:5.9\n")
    os.mkdir(os.path.join(project_dir, "MyApp.xcodeproj"))
    assert detect_test_command(project_dir, "") == "swift test"


def test_detect_test_command_makefile_with_test_target_beats_pyproject(project_dir):
    """A Makefile with a 'test:' target outranks the pyproject.toml ->
    pytest inference (Task 6: 'make test' wraps whatever the project actually
    needs, e.g. an xcodebuild invocation with scheme/destination)."""
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n')
    with open(os.path.join(project_dir, "Makefile"), "w") as f:
        f.write("test:\n\tpytest -q\n")
    assert detect_test_command(project_dir, "") == "make test"


def test_detect_test_command_makefile_without_test_target_falls_through(project_dir):
    """A Makefile that exists but has no 'test:' target must not short-circuit
    detection — falls through to the pyproject.toml -> pytest inference."""
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n')
    with open(os.path.join(project_dir, "Makefile"), "w") as f:
        f.write("build:\n\tswift build\n")
    assert detect_test_command(project_dir, "") == "pytest"


def test_detect_test_command_makefile_test_target_not_fooled_by_substring(project_dir):
    """'^test:' is anchored to the start of a line — a 'pytest:' or
    '.PHONY: test' line must not false-match as a real 'test:' target."""
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n')
    with open(os.path.join(project_dir, "Makefile"), "w") as f:
        f.write(".PHONY: test\npytest:\n\tpytest -q\n")
    assert detect_test_command(project_dir, "") == "pytest"


def test_detect_test_command_makefile_for_xcode_project(project_dir):
    """A bare .xcodeproj with a Makefile 'test:' target resolves to
    'make test' — the documented iOS convention where Make wraps xcodebuild
    with the correct scheme/destination args."""
    os.mkdir(os.path.join(project_dir, "MyApp.xcodeproj"))
    with open(os.path.join(project_dir, "Makefile"), "w") as f:
        f.write("test:\n\txcodebuild test -scheme MyApp -destination 'generic/platform=iOS Simulator'\n")
    assert detect_test_command(project_dir, "") == "make test"


def test_detect_test_command_instructions_override_wins_over_makefile(project_dir):
    """An explicit instructions-text override still wins even when a Makefile
    test target is present — instructions are checked first, unconditionally."""
    with open(os.path.join(project_dir, "Makefile"), "w") as f:
        f.write("test:\n\tmake-specific-runner\n")
    instructions = "## Testing\n\nRun `npm test` before committing.\n"
    assert detect_test_command(project_dir, instructions) == "npm test"


def test_detect_test_command_instructions_override_wins(project_dir):
    """A pyproject.toml (→ pytest by inference) is overridden by an explicit
    'npm test' instruction in CLAUDE.md/SPARK.md content — instructions win."""
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write('[project]\nname = "myapp"\n')
    instructions = "## Testing\n\nRun `npm test` before committing.\n"
    assert detect_test_command(project_dir, instructions) == "npm test"


def test_detect_test_command_instructions_used_without_project_markers(project_dir):
    """Instructions alone (no project markers on disk) still resolve a command."""
    assert detect_test_command(project_dir, "test: cargo test") == "cargo test"


def test_detect_test_command_word_boundary_not_fooled_by_substring(project_dir):
    """'pytest' must not match inside an unrelated filename/identifier in the
    instructions text (e.g. mentioning 'test_pytest_helpers.py')."""
    with open(os.path.join(project_dir, "package.json"), "w") as f:
        f.write('{"name": "myapp"}')
    instructions = "See test_pytest_helpers.py and mypytest_util for helpers."
    # No word-boundary "pytest" match in the instructions -> falls back to the
    # project-type marker (package.json -> npm test), NOT a false "pytest" hit.
    assert detect_test_command(project_dir, instructions) == "npm test"
