"""Web fetch tool — fetch and parse web pages with streaming support."""

import httpx

from .base import Tool


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a web page and return its text content. Useful for reading documentation, articles, and API references."
    is_read_only = True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 10000)",
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, max_length: int = 10000, **kw) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SparkCode/1.0)"
                })
                response.raise_for_status()
        except Exception as e:
            return f"Error fetching URL: {e}"

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = response.text

        if len(text) > max_length:
            text = text[:max_length] + f"\n\n... (truncated at {max_length} chars)"

        return f"URL: {url}\n\n{text}"

    async def execute_streaming(self, url: str, max_length: int = 10000,
                                callback=None, **kw) -> str:
        """Fetch with streaming progress updates."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return "Error: beautifulsoup4 not installed."

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                async with client.stream("GET", url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SparkCode/1.0)"
                }) as response:
                    response.raise_for_status()

                    total = int(response.headers.get("content-length", 0))
                    chunks = []
                    downloaded = 0

                    async for chunk in response.aiter_bytes(1024):
                        chunks.append(chunk)
                        downloaded += len(chunk)
                        if callback and total > 0:
                            pct = downloaded / total * 100
                            callback(f"Downloading... {downloaded:,}/{total:,} bytes ({pct:.0f}%)")
                        elif callback:
                            callback(f"Downloading... {downloaded:,} bytes")

                    body = b"".join(chunks)
                    text = body.decode("utf-8", errors="replace")

        except Exception as e:
            return f"Error fetching URL: {e}"

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        if len(text) > max_length:
            text = text[:max_length] + f"\n\n... (truncated at {max_length} chars)"

        if callback:
            callback(f"Fetched {len(text):,} chars from {url}")

        return f"URL: {url}\n\n{text}"
