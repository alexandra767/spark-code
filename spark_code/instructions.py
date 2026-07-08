"""Unified project-instruction loading: SPARK.md + CLAUDE.md hierarchy.

Precedence (first = highest): project SPARK.md, project CLAUDE.md (nearest
ancestor), ~/.spark/SPARK.md (global Spark-Code context, loads from any cwd),
~/.claude/CLAUDE.md. Mirrors Claude Code's CLAUDE.md discovery so the same
repos work in both tools; ~/.spark/SPARK.md is the Spark-Code-only global
equivalent (doesn't touch Claude Code's ~/.claude/CLAUDE.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PER_FILE_BUDGET = 8000


@dataclass
class Instructions:
    text: str = ""
    sources: list[str] = field(default_factory=list)


def _read_budgeted(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read(PER_FILE_BUDGET + 1)
    except OSError:
        return None
    if not content.strip():
        return None
    if len(content) > PER_FILE_BUDGET:
        content = content[:PER_FILE_BUDGET] + "\n[... truncated]"
    return content


def _find_project_claude_md(cwd: str) -> str | None:
    current = os.path.realpath(cwd)
    while True:
        candidate = os.path.join(current, "CLAUDE.md")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def load_instructions(cwd: str) -> Instructions:
    """Load and merge all project-instruction files for ``cwd``."""
    candidates: list[tuple[str, str | None]] = []

    for rel in ("SPARK.md", os.path.join(".spark", "SPARK.md")):
        path = os.path.join(cwd, rel)
        if os.path.isfile(path):
            candidates.append(("SPARK.md", path))
            break
    else:
        candidates.append(("SPARK.md", None))

    candidates.append(("CLAUDE.md", _find_project_claude_md(cwd)))
    # Global Spark-Code context — loads from ANY cwd (like ~/.claude/CLAUDE.md)
    # but Spark-Code-only, so it never bloats Claude Code's global instructions.
    global_spark = os.path.expanduser(os.path.join("~", ".spark", "SPARK.md"))
    candidates.append(("~/.spark/SPARK.md",
                       global_spark if os.path.isfile(global_spark) else None))
    global_path = os.path.expanduser(os.path.join("~", ".claude", "CLAUDE.md"))
    candidates.append(("~/.claude/CLAUDE.md",
                       global_path if os.path.isfile(global_path) else None))

    parts: list[str] = []
    sources: list[str] = []
    for name, path in candidates:
        if path is None:
            continue
        content = _read_budgeted(path)
        if content is None:
            continue
        parts.append(f"## Instructions from {name}\n\n{content}")
        sources.append(name)
    return Instructions(text="\n\n".join(parts), sources=sources)
