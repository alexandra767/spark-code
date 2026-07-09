"""Regression tests for rag_search.py: default collection fan-out (Task 9,
second-brain retrieval) vs explicit single-collection search."""

from unittest.mock import AsyncMock, patch

from spark_code.tools.rag_search import RagSearchTool


def _resp(results, elapsed_ms=10.0):
    m = AsyncMock()
    m.json = lambda: {"results": results, "elapsed_ms": elapsed_ms}
    return m


async def test_no_collection_fans_out_to_default_three_and_merges_by_score():
    calls = []

    async def fake_post(url, json):
        calls.append(json["collection"])
        by_coll = {
            "claude_documents": [{"source": "doc.md", "text": "low", "score": 1.0}],
            "claude_code_memory": [{"source": "mem.md", "text": "mid", "score": 2.0}],
            "second_brain": [{"source": "second-brain/nomad/rav4-build.md", "text": "high", "score": 3.0}],
        }
        return _resp(by_coll[json["collection"]])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="what is my rav4 power setup")

    assert calls == ["claude_documents", "claude_code_memory", "second_brain"]
    assert "second-brain/nomad/rav4-build.md" in result
    # Highest score (second_brain hit) should be ranked first.
    assert result.index("second-brain/nomad/rav4-build.md") < result.index("mem.md")


async def test_explicit_collection_searches_only_that_one():
    calls = []

    async def fake_post(url, json):
        calls.append(json["collection"])
        return _resp([{"source": "det.jpg", "text": "a detection", "score": 1.0}])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="raccoon", collection="jarvis_detections")

    assert calls == ["jarvis_detections"]
    assert "det.jpg" in result


async def test_no_results_across_all_default_collections():
    async def fake_post(url, json):
        return _resp([])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="nonexistent thing")

    assert "No results found" in result
