"""Regression test for web_search.py: uses the `ddgs` package and runs the
blocking search off the event loop (via asyncio.to_thread)."""

from unittest.mock import patch

from spark_code.tools.web_search import WebSearchTool


class _FakeDDGS:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, query, max_results=5):
        return [
            {"title": "Result One", "href": "https://example.com/1", "body": "snippet one"},
        ]


async def test_web_search_uses_ddgs_and_formats(monkeypatch):
    # Patch the ddgs.DDGS symbol the tool imports at call time.
    import ddgs
    with patch.object(ddgs, "DDGS", _FakeDDGS):
        result = await WebSearchTool().execute(query="python", max_results=1)
    assert "Result One" in result
    assert "https://example.com/1" in result
    assert "snippet one" in result


async def test_web_search_search_error_handled():
    class _Boom:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5):
            raise RuntimeError("network down")

    import ddgs
    with patch.object(ddgs, "DDGS", _Boom):
        result = await WebSearchTool().execute(query="x")
    assert "Search error" in result
