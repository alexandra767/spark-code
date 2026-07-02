"""RAG search tool — queries the RAG service on the Spark for indexed documents."""

import os

from .base import Tool

RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8010")


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "Search your indexed knowledge base (documents, Swift docs, Google Docs, "
        "crawled websites, camera detections, browser history, Telegram messages). "
        "Returns relevant passages with source citations. Use this when the user "
        "asks about content from their personal documents or indexed reference material."
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
                    "description": "Collection to search: claude_documents (default), jarvis_detections, jarvis_browser, jarvis_telegram",
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

    async def execute(self, query: str, collection: str = "claude_documents",
                      n_results: int = 5, search_type: str = "hybrid", **kw) -> str:
        try:
            import httpx
        except ImportError:
            return "Error: httpx not installed. Run: pip install httpx"

        payload = {
            "query": query,
            "collection": collection,
            "n_results": n_results,
            "search_type": search_type,
            "user_role": "owner",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{RAG_SERVICE_URL}/search", json=payload)
                data = resp.json()
        except httpx.ConnectError:
            return f"Error: RAG service not reachable at {RAG_SERVICE_URL}. Is it running on the Spark?"
        except Exception as e:
            return f"RAG search error: {e}"

        results = data.get("results", [])
        if not results:
            return f"No results found for: {query}"

        elapsed = data.get("elapsed_ms", 0)
        output = [f"**{len(results)} results** ({elapsed:.0f}ms)\n"]

        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            text = r.get("text", "")
            citation = r.get("citation", {})
            page = citation.get("page")
            page_str = f", p.{page}" if page else ""
            score = r.get("score", 0)

            output.append(f"**[{i}]** `{source}{page_str}` (score: {score:.4f})")
            output.append(f"{text}\n")

        return "\n".join(output)
