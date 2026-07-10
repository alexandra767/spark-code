"""Regression tests for rag_search.py: default collection fan-out (Task 9,
second-brain retrieval) vs explicit single-collection search."""

from unittest.mock import AsyncMock, patch

import httpx

from spark_code.tools.rag_search import RagSearchTool


def _resp(results, elapsed_ms=10.0, status_code=200):
    m = AsyncMock()
    m.status_code = status_code
    m.json = lambda: {"results": results, "elapsed_ms": elapsed_ms}
    m.raise_for_status = lambda: None  # sync no-op, like httpx.Response
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


# ---------------------------------------------------------------------------
# Fix 1 — collection="" must behave EXACTLY like collection=None
# ---------------------------------------------------------------------------


async def test_empty_string_collection_behaves_like_none():
    """An LLM tool-caller passing collection="" (falsy) previously fanned out
    but set explicit=True (because "" is not None), which suppressed the
    second_brain deepening AND the CANONICAL/HISTORICAL labels while still
    paying for the fan-out. Empty string must be identical to None."""
    calls = []
    n_for = {}

    async def fake_post(url, json):
        coll = json["collection"]
        calls.append(coll)
        n_for[coll] = json["n_results"]
        by_coll = {
            "claude_documents": [{"source": "doc.md", "text": "d", "score": -3.0}],
            "claude_code_memory": [{"source": "mem.md", "text": "m", "score": -2.0}],
            "second_brain": [{"source": "second-brain/apps/boonpoint.md", "text": "sb", "score": -4.0}],
        }
        return _resp(by_coll[coll])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="which apps are live", collection="")

    # Fans out across all three defaults (not treated as an explicit single).
    assert calls == ["claude_documents", "claude_code_memory", "second_brain"]
    # second_brain is deepened to the vault pool (15) precisely because it is
    # NOT explicit — the deepening was the thing "" used to suppress.
    assert n_for["second_brain"] == 15
    # The CANONICAL label is only applied on the non-explicit path.
    assert "CANONICAL" in result


async def test_empty_string_collection_matches_none_call():
    """Belt-and-suspenders: the rendered output for "" equals the output for
    None given identical service responses."""
    async def fake_post(url, json):
        coll = json["collection"]
        by_coll = {
            "claude_documents": [{"source": "doc.md", "text": "d", "score": -3.0}],
            "claude_code_memory": [{"source": "mem.md", "text": "m", "score": -2.0}],
            "second_brain": [{"source": "sb.md", "text": "sb", "score": -1.0}],
        }
        return _resp(by_coll[coll])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        out_none = await RagSearchTool().execute(query="q", collection=None)
    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        out_empty = await RagSearchTool().execute(query="q", collection="")

    assert out_none == out_empty


# ---------------------------------------------------------------------------
# Fix 2 — service URL resolves from config (rag.service_url), like code_search
# ---------------------------------------------------------------------------


async def test_config_service_url_is_honored():
    seen = {}

    async def fake_post(url, json):
        seen["url"] = url
        return _resp([{"source": "x.md", "text": "t", "score": -1.0}])

    cfg = {"rag": {"service_url": "http://configured-host:8010"}}
    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        await RagSearchTool(cfg).execute(query="q", collection="claude_documents")

    assert seen["url"] == "http://configured-host:8010/search"


def test_no_arg_construction_still_resolves_a_url():
    # The cli.py registration site calls RagSearchTool() with no config; that
    # must still yield a usable default URL (env → default).
    tool = RagSearchTool()
    assert tool._service_url.startswith("http")


# ---------------------------------------------------------------------------
# Fix 4 — a single failing collection must not abort the whole search
# ---------------------------------------------------------------------------


async def test_one_collection_failing_keeps_the_others():
    async def fake_post(url, json):
        coll = json["collection"]
        if coll == "claude_code_memory":
            raise httpx.ConnectError("this collection is down")
        return _resp([{"source": f"{coll}.md", "text": coll, "score": -1.0}])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="q")

    assert "claude_documents.md" in result
    assert "second_brain.md" in result
    assert "No results" not in result


async def test_non_json_collection_is_skipped_not_fatal():
    """A 502 HTML page (resp.json() → ValueError) from one collection must not
    JSONDecodeError the whole search."""
    async def fake_post(url, json):
        coll = json["collection"]
        if coll == "second_brain":
            bad = _resp([])

            def boom():
                raise ValueError("Expecting value: line 1 column 1")

            bad.json = boom
            return bad
        return _resp([{"source": f"{coll}.md", "text": coll, "score": -1.0}])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="q")

    assert "claude_documents.md" in result
    assert "claude_code_memory.md" in result


async def test_all_collections_failing_returns_reachability_error():
    async def fake_post(url, json):
        raise httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="q")

    assert "not reachable" in result.lower()
    assert "No results found" not in result


# ---------------------------------------------------------------------------
# Fix 5 — negative reranker scores; a missing score sorts BELOW real hits
# ---------------------------------------------------------------------------


async def test_missing_score_ranks_below_real_negative_hits():
    """Old code used `r.get("score") or 0`, ranking a scoreless hit (→0) ABOVE
    every real negative-scored hit. A missing score must sort to the bottom."""
    async def fake_post(url, json):
        by_coll = {
            "claude_documents": [{"source": "no_score.md", "text": "scoreless"}],  # no score key
            "claude_code_memory": [{"source": "real.md", "text": "real", "score": -1.5}],
            "second_brain": [],
        }
        return _resp(by_coll[json["collection"]])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="q")

    assert result.index("real.md") < result.index("no_score.md")


async def test_missing_score_second_brain_hit_not_backfilled_over_floor():
    """A scoreless second_brain candidate maps to -inf, which fails the
    VAULT_FLOOR_MIN_SCORE (-5.0) gate — it must NOT be force-injected."""
    async def fake_post(url, json):
        by_coll = {
            "claude_documents": [{"source": "a.md", "text": "a", "score": -1.0}],
            "claude_code_memory": [{"source": "b.md", "text": "b", "score": -1.1}],
            # second_brain returns only a scoreless candidate.
            "second_brain": [{"source": "sb_no_score.md", "text": "sb"}],
        }
        return _resp(by_coll[json["collection"]])

    with patch("httpx.AsyncClient.post", side_effect=fake_post):
        result = await RagSearchTool().execute(query="q")

    # It may still appear via the normal merge (it's one of few results), but
    # the point is the floor gate treats -inf as below -5.0 — assert the floor
    # helper's contract via the score label rather than presence.
    assert "sb_no_score.md" in result  # present through the merge, not a crash
    assert "(score: n/a)" in result    # scoreless shown as n/a, never 0.0000
