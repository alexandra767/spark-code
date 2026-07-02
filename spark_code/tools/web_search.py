"""Web search tool using DuckDuckGo."""

import asyncio

from .base import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for current information. Returns search results with titles, URLs, and snippets."
    is_read_only = True

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 5)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_results: int = 5, **kw) -> str:
        # `duckduckgo_search` was renamed to `ddgs`. Prefer the new package
        # (avoids the deprecation RuntimeWarning the old one emits on every
        # call), fall back to the old name only if `ddgs` isn't installed.
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return "Error: ddgs not installed. Run: pip install ddgs"

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            # ddgs.text() is blocking network I/O — keep it off the event loop.
            results = await asyncio.to_thread(_search)
        except Exception as e:
            return f"Search error: {e}"

        if not results:
            return f"No results found for: {query}"

        output = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", r.get("link", ""))
            body = r.get("body", r.get("snippet", ""))
            output.append(f"{i}. **{title}**\n   {url}\n   {body}")

        return "\n\n".join(output)
