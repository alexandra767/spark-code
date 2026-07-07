"""``/doctor`` health check (Phase 5 Task 3).

One command that checks the whole local stack — primary engine, served-model
alias, utility model, RAG service, MCP servers, host editor, install/git
state, and configured cloud keys — and renders a single green/yellow/red
table. Read-only: nothing here writes a file, touches config, or mutates
process state; it only probes and reports.

Design: each of the 8 checks is its own small ``async def _check_*`` that
returns a :class:`DoctorCheck` — never raises in the *expected* failure
modes (unreachable server, missing config, no git repo, etc. all become a
``"warn"``/``"fail"`` row, exactly like :func:`spark_code.preflight.
fetch_server_models`'s "return None on any error" contract). :func:`run_doctor`
additionally wraps every check in :func:`_wrap`, a last-resort safety net —
so even a check with a genuine bug (an unanticipated exception type) turns
into a ``"fail"`` row instead of crashing a read-only diagnostic command that
is specifically meant to be safe to run when everything else is on fire. The
``Install`` check deliberately does NOT self-catch its ``git_runner`` calls
(missing ``git`` binary, a timeout, an injected test failure) — that's what
proves ``_wrap`` actually catches, not just each check's own internal
try/except (see tests/test_doctor.py's isolation test).

Network probes accept injectable ``httpx`` transports (mirroring
tests/test_preflight.py's ``httpx.MockTransport`` idiom) so ``run_doctor``
never touches the network in tests. Away from the DGX — engine unreachable,
RAG down, utility model unreachable — every affected check degrades to a
``"warn"``/``"fail"`` row; it never raises and never blocks the command.

SECURITY: the cloud-keys check (:func:`_check_keys`) only ever puts
provider NAMES and :func:`spark_code.keys.mask`-ed values into a
``DoctorCheck.detail`` — never a raw key value. See that function's
docstring.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from . import __version__
from .codesearch import resolve_service_url
from .config import get
from .editor import detect_editor
from .keys import load_keys, mask
from .mcp.client import MCPClient
from .preflight import _extract_max_context, context_window_warning, fetch_server_models
from .routing import DEFAULT_UTILITY_MODEL

logger = logging.getLogger(__name__)

# Short timeouts everywhere — this is an interactive, on-demand diagnostic;
# a hung probe must never make `/doctor` itself feel broken. The primary/
# utility checks reuse fetch_server_models' own 5s default; RAG gets a
# tighter budget per the plan ("brief GET, short timeout").
_RAG_TIMEOUT = 3.0
_GIT_TIMEOUT = 5.0
# Per-server ceiling for the MCP check. StdioTransport/HTTPTransport default
# to ~30s, so a server that accepts the connection but never answers
# `initialize` would otherwise block `/doctor` up to 30s PER server,
# sequentially — reproducibly minutes in exactly the away-from-home/DGX-down
# scenario this command exists to diagnose. asyncio.wait_for bounds each
# connect (see _check_mcp).
_MCP_TIMEOUT = 6.0

Status = str  # "ok" | "warn" | "fail"


@dataclass
class DoctorCheck:
    """One row of the `/doctor` table."""

    name: str
    status: Status  # "ok" | "warn" | "fail"
    detail: str


async def _wrap(name: str, coro: Awaitable[DoctorCheck]) -> DoctorCheck:
    """Run one check; any exception becomes a ``"fail"`` row, never a crash.

    Every ``_check_*`` below already handles its own expected failure modes
    (unreachable server → None from fetch_server_models, missing config →
    get()'s default, etc.) and returns a DoctorCheck either way. This is the
    outer safety net for the unexpected case — a check with a genuine bug,
    or (deliberately, for ``Install``) a probe that isn't self-catching.
    """
    try:
        return await coro
    except Exception as e:
        logger.warning("doctor check %r raised", name, exc_info=True)
        return DoctorCheck(name=name, status="fail", detail=f"check error: {e}")


# ---------------------------------------------------------------------------
# 1 & 2: primary engine reachability + context-window drift + alias presence
# ---------------------------------------------------------------------------

async def _check_primary_engine(config: dict, transport) -> DoctorCheck:
    endpoint = get(config, "model", "endpoint", default="") or ""
    if not endpoint:
        return DoctorCheck("Primary engine", "fail", "no model.endpoint configured")
    api_key = get(config, "model", "api_key", default="") or ""
    data = await fetch_server_models(endpoint, api_key, transport=transport)
    if data is None:
        return DoctorCheck("Primary engine", "fail", f"unreachable at {endpoint}")

    configured_ctx = get(config, "model", "context_window", default=None)
    if isinstance(configured_ctx, int):
        model_name = get(config, "model", "name", default="") or ""
        server_max = _extract_max_context(data, model_name)
        warning = context_window_warning(configured_ctx, server_max)
        if warning:
            return DoctorCheck("Primary engine", "warn", warning)

    return DoctorCheck(
        "Primary engine", "ok",
        f"reachable at {endpoint} ({len(data)} model(s) served)")


async def _check_model_alias(config: dict, transport) -> DoctorCheck:
    endpoint = get(config, "model", "endpoint", default="") or ""
    model_name = get(config, "model", "name", default="") or ""
    if not endpoint or not model_name:
        return DoctorCheck("Model alias", "fail", "no model.endpoint/model.name configured")
    api_key = get(config, "model", "api_key", default="") or ""
    data = await fetch_server_models(endpoint, api_key, transport=transport)
    if data is None:
        return DoctorCheck(
            "Model alias", "fail", f"could not verify — engine unreachable at {endpoint}")
    served_ids = {item.get("id") for item in data}
    if model_name in served_ids:
        return DoctorCheck("Model alias", "ok", f"'{model_name}' present in /v1/models")
    return DoctorCheck(
        "Model alias", "warn",
        f"'{model_name}' not found in /v1/models ({len(served_ids)} model(s) served)")


# ---------------------------------------------------------------------------
# 3: utility model (Phase 4 Task 2 — spark_code/routing.py)
# ---------------------------------------------------------------------------

async def _check_utility_model(config: dict, transport) -> DoctorCheck:
    if "utility_model" not in config:
        return DoctorCheck("Utility model", "warn", "not configured — primary handles everything")
    block = config["utility_model"]
    if block is None:
        return DoctorCheck("Utility model", "warn", "disabled (utility_model: null)")
    if not isinstance(block, dict):
        return DoctorCheck("Utility model", "warn", "utility_model config is malformed — disabled")

    # Same merge-over-default DEFAULT_UTILITY_MODEL applies (routing.
    # get_utility_client) — reused directly rather than building a real
    # ModelClient, so this stays testable via the same MockTransport idiom
    # as the primary-engine check instead of needing to patch a client's
    # internal httpx.AsyncClient.
    merged = {**DEFAULT_UTILITY_MODEL, **block}
    endpoint = merged.get("endpoint", "")
    model_name = merged.get("model", "")
    api_key = merged.get("api_key", "") or ""

    data = await fetch_server_models(endpoint, api_key, transport=transport)
    if data is None:
        return DoctorCheck("Utility model", "fail", f"unreachable at {endpoint}")
    served_ids = {item.get("id") for item in data}
    if model_name and model_name not in served_ids:
        return DoctorCheck(
            "Utility model", "warn",
            f"reachable at {endpoint}, but '{model_name}' not listed")
    return DoctorCheck("Utility model", "ok", f"reachable at {endpoint} ({model_name})")


# ---------------------------------------------------------------------------
# 4: RAG service (spark_code/codesearch.py — rag.service_url)
# ---------------------------------------------------------------------------

async def _check_rag(config: dict, transport) -> DoctorCheck:
    service_url = resolve_service_url(config).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=_RAG_TIMEOUT, transport=transport) as client:
            resp = await client.get(f"{service_url}/health")
    except Exception as e:
        # Matches codesearch.py's own stance: an unreachable RAG service is
        # an "away from home" degrade, not a broken install — code_search
        # just returns its friendly unavailable_message() per query.
        return DoctorCheck("RAG service", "warn", f"unreachable at {service_url}: {e}")
    if resp.status_code != 200:
        return DoctorCheck(
            "RAG service", "warn", f"{service_url} returned HTTP {resp.status_code}")
    return DoctorCheck("RAG service", "ok", f"reachable at {service_url}")


# ---------------------------------------------------------------------------
# 5: MCP servers (config["mcp_servers"] — see cli.py's connect_all wiring)
# ---------------------------------------------------------------------------

async def _default_mcp_connect(name: str, server_conf: dict) -> None:
    """Best-effort real connect: spin the server up, discover tools, tear
    it down. Raises on failure — the caller (:func:`_check_mcp`) is the one
    that turns that into a per-server result, never a crash."""
    client = MCPClient()
    try:
        await client.connect(name, server_conf)
    finally:
        await client.disconnect_all()


async def _check_mcp(
    config: dict,
    connector: Callable[[str, dict], Awaitable[None]] | None,
) -> DoctorCheck:
    mcp_configs = get(config, "mcp_servers", default={})
    if not isinstance(mcp_configs, dict) or not mcp_configs:
        return DoctorCheck("MCP servers", "ok", "none configured")

    connect = connector or _default_mcp_connect
    launched: list[str] = []
    failed: list[str] = []
    for name, server_conf in mcp_configs.items():
        try:
            # Bound each server independently: a server stuck mid-handshake
            # times out into a row and the loop moves on, rather than one
            # unresponsive server blocking the whole check (and every server
            # after it) for the transport's default ~30s.
            await asyncio.wait_for(
                connect(name, server_conf), timeout=_MCP_TIMEOUT)
            launched.append(name)
        except (asyncio.TimeoutError, TimeoutError):
            failed.append(f"{name} (did not respond within {_MCP_TIMEOUT:g}s)")
        except Exception as e:
            failed.append(f"{name} ({e})")

    total = len(mcp_configs)
    if not launched:
        return DoctorCheck("MCP servers", "fail", f"0/{total} launched — {'; '.join(failed)}")
    if failed:
        return DoctorCheck(
            "MCP servers", "warn",
            f"{len(launched)}/{total} launched — failed: {'; '.join(failed)}")
    return DoctorCheck("MCP servers", "ok", f"{len(launched)}/{total} launched: {', '.join(launched)}")


# ---------------------------------------------------------------------------
# 6: host editor (Phase 3 Task 4 — spark_code/editor.py)
# ---------------------------------------------------------------------------

async def _check_editor(config: dict) -> DoctorCheck:
    configured = get(config, "ui", "editor", default="auto") or "auto"
    if configured == "none":
        return DoctorCheck("Editor", "warn", "disabled (ui.editor: none) — /open is a no-op")
    if configured != "auto":
        return DoctorCheck("Editor", "ok", f"configured: {configured}")
    editor = detect_editor()
    if editor:
        return DoctorCheck("Editor", "ok", f"detected: {editor}")
    return DoctorCheck(
        "Editor", "warn",
        "no editor detected (Cursor/VS Code/Xcode) — /open will be a no-op")


# ---------------------------------------------------------------------------
# 7: install version + git branch/dirty state
# ---------------------------------------------------------------------------

async def _check_install(cwd: str | None, git_runner: Callable[..., Any]) -> DoctorCheck:
    """Version + git state. Deliberately does NOT catch ``git_runner``
    failures itself (missing binary, timeout, a hostile injected fake in
    tests) — that's exactly what proves ``run_doctor``'s outer ``_wrap``
    turns a raising probe into a ``"fail"`` row rather than a crash."""
    parts = [f"v{__version__}"]
    branch_result = git_runner(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, timeout=_GIT_TIMEOUT, cwd=cwd,
    )
    if branch_result.returncode != 0:
        # Non-zero covers both "not a git repo" and a perms/corrupt-.git case
        # (git present but rev-parse failed) — don't mislabel the latter.
        parts.append("not a git repo or git unavailable")
        return DoctorCheck("Install", "ok", ", ".join(parts))

    branch = branch_result.stdout.strip()
    parts.append(f"branch {branch}" if branch else "detached HEAD")

    dirty_result = git_runner(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, timeout=_GIT_TIMEOUT, cwd=cwd,
    )
    if dirty_result.returncode == 0:
        parts.append("dirty" if dirty_result.stdout.strip() else "clean")

    return DoctorCheck("Install", "ok", ", ".join(parts))


# ---------------------------------------------------------------------------
# 8: cloud keys (Phase 5 Task 2 — spark_code/keys.py). NAMES + masked only.
# ---------------------------------------------------------------------------

async def _check_keys(config: dict) -> DoctorCheck:
    """Cloud keys present, by provider name — NEVER a raw value.

    ``mask()`` is the only display-safe form (first-3/last-4, or fully
    masked for anything short enough that the two windows could overlap) —
    same guarantee /setkey's confirmation prints already rely on.
    """
    keys = load_keys()
    if not keys:
        return DoctorCheck("Cloud keys", "warn", "none configured (see /setkey)")
    detail = ", ".join(f"{name}: {mask(value)}" for name, value in sorted(keys.items()))
    return DoctorCheck("Cloud keys", "ok", detail)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_doctor(
    config: dict,
    *,
    transport: httpx.BaseTransport | None = None,
    utility_transport: httpx.BaseTransport | None = None,
    rag_transport: httpx.BaseTransport | None = None,
    mcp_connector: Callable[[str, dict], Awaitable[None]] | None = None,
    git_runner: Callable[..., Any] = subprocess.run,
    cwd: str | None = None,
) -> list[DoctorCheck]:
    """Run all 8 checks, each isolated — nothing here ever raises.

    ``config`` is the only argument production callers need; every other
    parameter is a keyword-only injection point for tests (mirrors
    tests/test_preflight.py's ``httpx.MockTransport`` idiom — ``None``
    means "really touch the network"/"really run git").
    """
    return [
        await _wrap("Primary engine", _check_primary_engine(config, transport)),
        await _wrap("Model alias", _check_model_alias(config, transport)),
        await _wrap("Utility model", _check_utility_model(config, utility_transport)),
        await _wrap("RAG service", _check_rag(config, rag_transport)),
        await _wrap("MCP servers", _check_mcp(config, mcp_connector)),
        await _wrap("Editor", _check_editor(config)),
        await _wrap("Install", _check_install(cwd, git_runner)),
        await _wrap("Cloud keys", _check_keys(config)),
    ]


_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "ok": ("#a3be8c", "OK"),
    "warn": ("#ebcb8b", "WARN"),
    "fail": ("#bf616a", "FAIL"),
}


def render_doctor(console: Console, checks: list[DoctorCheck]) -> None:
    """Print the green/yellow/red `/doctor` table. Read-only — no return value."""
    table = Table(
        title="Spark Code Doctor",
        border_style="#4c566a",
        show_header=True,
        header_style="bold #88c0d0",
        box=ROUNDED,
    )
    table.add_column("Check", style="#d8dee9")
    table.add_column("Status", justify="center")
    table.add_column("Detail", style="#8899aa")
    for check in checks:
        color, label = _STATUS_STYLE.get(check.status, ("#d8dee9", check.status.upper()))
        table.add_row(check.name, f"[{color}]{label}[/{color}]", check.detail)
    console.print(table)
