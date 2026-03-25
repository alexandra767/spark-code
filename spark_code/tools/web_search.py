"""Web search tool using DuckDuckGo."""

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
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
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
