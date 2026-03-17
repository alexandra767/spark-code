"""Tests for smart project detection."""

import json
import os
import shutil
import tempfile

import pytest

from spark_code.project_detect import detect_project_type


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
