"""stack_info tool — queries Orrery, the live map of the owner's own AI/
automation stack (applications, scheduled routines, memory stores, skills),
running on the Spark at :4200. Peer of Claude UI's agentic_stack tool
(claude-ui/backend/tools/handlers_stack.py, Task 9) — same HTTP contract,
same compact rendering, ported to this tool's str-returning interface.

This is the MAC checkout: Orrery runs on the Spark, not locally, so the
default target is the Spark's tailnet/LAN hostname rather than localhost.
Override with ORRERY_URL if that ever changes."""

import os
import re

import httpx

from .base import Tool

ORRERY_URL = os.environ.get("ORRERY_URL", "http://spark-4a54.local:4200")

SECTIONS = ("applications", "routines", "memory", "skills")

# Query "looks file-ish" heuristic: a path separator, a trailing
# dot-extension, or a word that's normally about files rather than stack
# components. When it matches, also fan the query out to Orrery's file index
# (/api/files/search) alongside the stack search.
_FILE_HINT_RE = re.compile(
    r"[\\/]|\.[a-zA-Z0-9]{1,5}$|\b(file|files|folder|directory|document|"
    r"photo|photos|screenshot|pdf|csv|spreadsheet|image|zip|download)\b",
    re.IGNORECASE,
)


def _looks_file_ish(query: str) -> bool:
    return bool(_FILE_HINT_RE.search(query))


def _freshness(node: dict) -> str:
    """Compact 'k=v, k=v' stamp from whichever freshness-shaped fields a node
    has — routines carry schedule/last_run/last_result/next_run, memory
    stores carry kind/count, skills carry family/source, applications carry
    kind/detail. Never assumes a field exists (ring shapes differ)."""
    parts = []
    for key in ("kind", "schedule", "last_run", "last_result", "next_run",
                "count", "family", "source", "department", "host"):
        val = node.get(key)
        if val not in (None, ""):
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def _render_nodes(nodes: list, limit: int) -> str:
    lines = []
    for n in nodes[:limit]:
        name = n.get("label") or n.get("id")
        fresh = _freshness(n)
        lines.append(f"- {name} — {fresh}" if fresh else f"- {name}")
    if len(nodes) > limit:
        lines.append(f"... ({len(nodes) - limit} more not shown)")
    return "\n".join(lines)


class StackInfoTool(Tool):
    name = "stack_info"
    description = (
        "Read Orrery, the live map of the owner's OWN AI/automation stack: "
        "assistant applications (Claude UI, JARVIS, MCP servers...), "
        "scheduled routines (systemd timers/cron with last/next run + "
        "result), memory stores (RAG collections, the SecondBrain vault), "
        "and skills/tools across the Spark and Mac. Use for questions ABOUT "
        "the stack itself — 'what routines do I have', 'what's connected to "
        "my email', 'when did the vault last sync', 'do I have a skill for "
        "X'. Not for the content those systems manage (use rag_search for "
        "what's IN memory, not what memory stores exist)."
    )
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-text search across stack component names/ids "
                        "(and, when the query looks file-ish, also fans out "
                        "to the file index). Omit for an overview or a full "
                        "section listing."
                    ),
                },
                "section": {
                    "type": "string",
                    "enum": ["applications", "routines", "memory", "skills", "summary"],
                    "description": (
                        "Narrow to one ring. summary (default when query is "
                        "also omitted) = counts + freshness only. Combine "
                        "with query to search within a section."
                    ),
                },
            },
        }

    async def execute(self, query: str | None = None, section: str | None = None, **kw) -> str:
        query = (query or "").strip()
        section = (section or "").strip().lower()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # section=applications|routines|memory|skills: pull the full
                # model and filter client-side by ring (the API has no ring
                # filter of its own), optionally narrowed further by query.
                if section and section != "summary":
                    if section not in SECTIONS:
                        return (
                            f"Unknown section '{section}'. Use one of: "
                            f"{', '.join(SECTIONS)}, summary."
                        )
                    resp = await client.get(f"{ORRERY_URL}/api/stack/model")
                    resp.raise_for_status()
                    nodes = [n for n in resp.json().get("nodes", []) if n.get("ring") == section]
                    if query:
                        needle = query.lower()
                        nodes = [
                            n for n in nodes
                            if needle in str(n.get("label") or "").lower()
                            or needle in str(n.get("id") or "").lower()
                        ]
                    if not nodes:
                        return f"No {section} match for '{query}'." if query else f"No {section} found."
                    header = f"{section} ({len(nodes)})" + (f" matching '{query}'" if query else "") + ":\n"
                    return header + _render_nodes(nodes, limit=60)

                # No section (or section=summary) + a query: general
                # cross-ring search, plus a file-index fan-out when the
                # query reads file-ish.
                if query:
                    resp = await client.get(f"{ORRERY_URL}/api/stack/search", params={"q": query})
                    resp.raise_for_status()
                    nodes = resp.json().get("nodes", [])
                    if not nodes:
                        text = f"No stack matches for '{query}'."
                    else:
                        text = f"Stack matches for '{query}' ({len(nodes)}):\n" + _render_nodes(nodes, limit=40)

                    if _looks_file_ish(query):
                        try:
                            fresp = await client.get(f"{ORRERY_URL}/api/files/search", params={"q": query})
                            if fresp.status_code == 200:
                                fdata = fresp.json()
                                hits = fdata.get("hits", [])[:20]
                                if hits:
                                    trunc = " (truncated)" if fdata.get("truncated") else ""
                                    file_lines = "\n".join(f"- [{h['host']}] {h['path']}" for h in hits)
                                    text += f"\n\nFile matches{trunc}:\n{file_lines}"
                        except httpx.HTTPError:
                            pass  # file index is a bonus here — a search still stands without it
                    return text

                # Default: the overview.
                resp = await client.get(f"{ORRERY_URL}/api/stack/summary")
                resp.raise_for_status()
                data = resp.json()
                counts = data.get("counts", {})
                generated = data.get("generated", {})
                warnings = data.get("warnings") or []
                lines = ["Orrery stack summary:"]
                for ring in SECTIONS:
                    lines.append(f"- {ring}: {counts.get(ring, 0)}")
                gen_str = ", ".join(f"{k}={v}" for k, v in generated.items() if v)
                if gen_str:
                    lines.append(f"generated: {gen_str}")
                lines.append(f"warnings: {'; '.join(warnings) if warnings else 'none'}")
                return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"Orrery is not running: {type(exc).__name__} — is the stack-mapper service up on {ORRERY_URL}?"
