"""RAG search tool — queries the RAG service on the Spark for indexed documents."""

import os

from .base import Tool

RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8010")

# When the caller doesn't name a collection, fan out across the general
# document library, the synced history of Claude Code sessions
# (claude_code_memory), and the synced Obsidian vault (second_brain), then
# merge and re-rank. Mirrors the Claude UI rag_search handler's default
# fan-out (claude-ui/backend/tools/handlers_rag.py, Task 8).
DEFAULT_COLLECTIONS = ["claude_documents", "claude_code_memory", "second_brain"]


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "Search your indexed knowledge base (documents, Swift docs, Google Docs, "
        "crawled websites, camera detections, browser history, Telegram messages, "
        "your second-brain vault). Returns relevant passages with source citations. "
        "Use this when the user asks about content from their personal documents, "
        "notes, or indexed reference material."
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
                    "description": "Search query — what to look for",
                },
                "collection": {
                    "type": "string",
                    "description": (
                        "Collection to search. Omit to search everything that matters "
                        "by default (claude_documents, claude_code_memory, second_brain). "
                        "Or name one directly: claude_documents, claude_code_memory, "
                        "second_brain, jarvis_detections, jarvis_browser, jarvis_telegram"
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results (default: 5)",
                },
                "search_type": {
                    "type": "string",
                    "description": "hybrid (default), semantic, or keyword",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, collection: str | None = None,
                      n_results: int = 5, search_type: str = "hybrid", **kw) -> str:
        try:
            import httpx
        except ImportError:
            return "Error: httpx not installed. Run: pip install httpx"

        collections = [collection] if collection else DEFAULT_COLLECTIONS

        base_payload = {
            "query": query,
            "n_results": n_results,
            "search_type": search_type,
            "user_role": "owner",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                merged = []
                elapsed = 0.0
                for coll in collections:
                    resp = await client.post(
                        f"{RAG_SERVICE_URL}/search",
                        json=dict(base_payload, collection=coll),
                    )
                    data = resp.json()
                    elapsed += data.get("elapsed_ms", 0) or 0
                    merged.extend(data.get("results", []))
        except httpx.ConnectError:
            return f"Error: RAG service not reachable at {RAG_SERVICE_URL}. Is it running on the Spark?"
        except Exception as e:
            return f"RAG search error: {e}"

        if not merged:
            return f"No results found for: {query}"

        if len(collections) > 1:
            merged.sort(key=lambda r: r.get("score") or 0, reverse=True)
            merged = merged[:max(n_results, 5)]

        output = [f"**{len(merged)} results** ({elapsed:.0f}ms)\n"]

        for i, r in enumerate(merged, 1):
            source = r.get("source", "unknown")
            text = r.get("text", "")
            citation = r.get("citation", {})
            page = citation.get("page")
            page_str = f", p.{page}" if page else ""
            score = r.get("score", 0)

            output.append(f"**[{i}]** `{source}{page_str}` (score: {score:.4f})")
            output.append(f"{text}\n")

        return "\n".join(output)
