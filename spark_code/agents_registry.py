"""Custom subagent definitions — user-defined agent types (Phase 5 Task 4).

Mirrors the skills bridge (``spark_code/skills/base.py``): a definition is a
markdown file with YAML frontmatter + a markdown body, living in
``.spark/agents/*.md`` (project) or ``~/.spark/agents/*.md`` (global). The
frontmatter parser is REUSED from ``skills.base`` (``_FRONTMATTER_RE``) rather
than reimplemented here.

Frontmatter fields:
  * ``name`` (required) — the ``agent_type`` value dispatch_agent will
    accept. A def without one is skipped (matching the Claude-skill bridge's
    ``_parse_claude_skill``, which has the same requirement).
  * ``description`` — one-liner shown in the dispatch_agent tool's index.
  * ``tools`` — an allowlist (list, or comma-separated string) of tool names
    the sub-agent may use. ``None``/absent means "the base type's full
    default set". This can only NARROW what the base type already allows —
    see ``dispatch.py``'s ``_build_custom_registry`` for the intersection.
  * ``type`` (alias ``base``) — which built-in agent class governs this
    def's PERMISSION ceiling: ``explore``/``reviewer`` (read-only) or
    ``implementer`` (may write). Defaults to ``explore`` — i.e. a custom def
    is read-only unless it explicitly opts into ``implementer``. This is the
    field the brief's "default read-only unless the def opts into writes"
    rule keys off; it is intentionally a separate axis from ``tools`` so a
    read-only-class def can't sneak write access in just by NAMING a write
    tool in its allowlist (the intersection in dispatch.py drops it).
  * ``model`` (alias ``model_hint``) — ``primary`` or ``utility``; which
    model client should run this sub-agent (Phase 4 Task 2 dual-model
    routing). Defaults to ``primary`` behavior (``None``).

The markdown body (everything after the closing ``---``) becomes the
sub-agent's ``system_prompt`` verbatim.

Malformed files (no frontmatter, unparseable YAML, missing name/body, a name
colliding with a built-in agent type) are SKIPPED with a logged warning —
never raised. A broken ``.spark/agents/*.md`` must never crash startup, same
contract as the Claude-skill bridge.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .skills.base import _FRONTMATTER_RE

logger = logging.getLogger(__name__)

# Reserved — a custom def may not redefine one of the built-in agent types
# (dispatch.py.AGENT_TYPES). Duplicated here as a plain tuple (not imported)
# to avoid a cli.py<->dispatch.py<->agents_registry import cycle; the two are
# guarded to stay in sync by test_custom_agents.py.
_BUILTIN_AGENT_TYPES = ("explore", "reviewer", "implementer")
_READ_ONLY_BASE_TYPES = ("explore", "reviewer")
_VALID_MODEL_HINTS = ("primary", "utility")


@dataclass(frozen=True)
class AgentDef:
    """A user-defined subagent type loaded from ``.spark/agents/*.md``."""

    name: str
    description: str
    system_prompt: str
    tools: list[str] | None = None
    model_hint: str | None = None
    # Permission ceiling — which built-in AGENT_TYPES class this def is
    # filtered through. Not part of the brief's minimal field list, but
    # required to make "read-only unless opted into writes" a checkable,
    # non-bypassable property (see module docstring + dispatch.py).
    base_type: str = "explore"
    source: str = "project"

    @property
    def is_read_only(self) -> bool:
        return self.base_type in _READ_ONLY_BASE_TYPES


def _coerce_tools(raw) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    if isinstance(raw, list) and all(isinstance(t, str) for t in raw):
        return list(raw)
    raise ValueError(f"tools must be a list of strings or a comma-separated string, got {raw!r}")


def _parse_agent_def(path: Path, source: str) -> AgentDef | None:
    """Parse one ``.md`` file into an :class:`AgentDef`, or ``None`` if
    malformed — logs a warning in every skip case, never raises."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read agent definition %s: %s", path, e)
        return None

    match = _FRONTMATTER_RE.match(text)
    if not match:
        logger.warning(
            "Skipping malformed agent definition %s: missing YAML frontmatter "
            "(expected '---\\n...\\n---\\n<body>')", path)
        return None

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        logger.warning("Skipping malformed agent definition %s: invalid YAML: %s", path, e)
        return None

    if not isinstance(meta, dict):
        logger.warning(
            "Skipping malformed agent definition %s: frontmatter is not a mapping", path)
        return None

    body = match.group(2).strip()
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        logger.warning("Skipping malformed agent definition %s: missing/invalid name", path)
        return None
    name = name.strip()
    if not body:
        logger.warning("Skipping malformed agent definition %s: empty body (no system prompt)", path)
        return None

    if name in _BUILTIN_AGENT_TYPES:
        logger.warning(
            "Skipping agent definition %s: name %r collides with a built-in "
            "agent type (%s) and cannot be redefined.",
            path, name, ", ".join(_BUILTIN_AGENT_TYPES))
        return None

    try:
        tools = _coerce_tools(meta.get("tools"))
    except ValueError as e:
        logger.warning("Skipping malformed agent definition %s: %s", path, e)
        return None

    base_type = meta.get("type") or meta.get("base") or "explore"
    if base_type not in _BUILTIN_AGENT_TYPES:
        logger.warning(
            "Agent definition %s: unknown type %r — defaulting to 'explore' "
            "(read-only). Valid values: %s.",
            path, base_type, ", ".join(_BUILTIN_AGENT_TYPES))
        base_type = "explore"

    model_hint = meta.get("model") or meta.get("model_hint")
    if model_hint is not None and model_hint not in _VALID_MODEL_HINTS:
        logger.warning(
            "Agent definition %s: unknown model hint %r — ignoring (using "
            "default). Valid values: %s.",
            path, model_hint, ", ".join(_VALID_MODEL_HINTS))
        model_hint = None

    description = meta.get("description", "") or ""
    if not isinstance(description, str):
        description = str(description)

    return AgentDef(
        name=name,
        description=description.strip(),
        system_prompt=body,
        tools=tools,
        model_hint=model_hint,
        base_type=base_type,
        source=source,
    )


def load_agent_defs(cwd: str | None = None) -> dict[str, AgentDef]:
    """Load custom subagent definitions: global (``~/.spark/agents/*.md``)
    then project (``.spark/agents/*.md`` under *cwd*) — project wins on a
    name clash, matching the skills bridge's project-over-global convention.

    Every ``.md`` file is parsed independently; a malformed one is skipped
    with a logged warning (see ``_parse_agent_def``) rather than raising, so
    one bad file can never crash startup. Returns ``{}`` when neither
    directory exists.
    """
    cwd = cwd or os.getcwd()
    defs: dict[str, AgentDef] = {}

    global_dir = Path(os.path.expanduser("~/.spark/agents"))
    if global_dir.is_dir():
        for f in sorted(global_dir.glob("*.md")):
            d = _parse_agent_def(f, source="global")
            if d is not None:
                defs[d.name] = d

    project_dir = Path(cwd) / ".spark" / "agents"
    if project_dir.is_dir():
        for f in sorted(project_dir.glob("*.md")):
            d = _parse_agent_def(f, source="project")
            if d is not None:
                defs[d.name] = d  # overwrites a global def of the same name

    return defs


def build_agent_index(defs: dict[str, AgentDef], cap: int = 1500) -> str:
    """Lean one-line-per-def index (``name — description``), name-sorted for
    a stable/cache-friendly result — mirrors ``SkillRegistry.build_index``.
    Capped to *cap* characters; the full ``system_prompt`` of a def NEVER
    appears here (only on actual dispatch).
    """
    lines: list[str] = []
    for name in sorted(defs):
        d = defs[name]
        line = f"- {name}: {d.description}" if d.description else f"- {name}"
        candidate = lines + [line]
        if len("\n".join(candidate)) > cap:
            break
        lines = candidate
    return "\n".join(lines)
