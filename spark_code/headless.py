"""Headless mode: the machine-readable JSON output contract for one-shot runs.

Phase 4 Task 1. This is the integration surface scripts/cron/JARVIS use to
drive Spark as an engine instead of an interactive assistant: `_one_shot` in
cli.py builds a :class:`HeadlessResult` after the agent loop finishes and, in
``--output json`` mode, prints exactly ONE line of JSON to stdout via
:func:`build_headless_json` (see cli.py's ``_one_shot`` for the wiring).

Kept dependency-free (stdlib only) so it can be imported without pulling in
the rest of the CLI — useful for tests and for anything that wants to build
or parse the contract without importing cli.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class HeadlessResult:
    """Machine-readable summary of a single headless (one-shot) run.

    Field set and order match the documented JSON contract exactly:
    result, files_changed, commands_run, tool_calls, rounds, input_tokens,
    output_tokens, error. ``error`` is ``None`` on a clean completion.
    """

    result: str
    files_changed: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    tool_calls: int = 0
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


def build_headless_json(result: HeadlessResult) -> str:
    """Serialize a :class:`HeadlessResult` to a single JSON line.

    ``json.dumps`` escapes embedded newlines in string values as the two
    characters ``\\n`` rather than emitting a literal line break, so the
    output is always exactly one line — safe to pipe into ``jq``, read with
    a single ``readline()``, etc.
    """
    return json.dumps(asdict(result))
