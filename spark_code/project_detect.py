"""Smart project detection — scan for project markers and frameworks."""

import json
import os


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
