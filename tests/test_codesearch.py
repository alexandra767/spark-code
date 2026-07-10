"""Tests for RAG code search (Phase 4 Task 3).

Mocks httpx via MockTransport, matching the pattern in tests/test_preflight.py
— no test here ever touches the network. The live RAG service investigation
(what /search, /ingest_text, /rebuild_bm25 actually accept, and that
extra_metadata round-trips into search hit metadata) was done separately
against the real service at spark-4a54.local:8010 — see
.superpowers/sdd/task-3-report.md.
"""

import json

import httpx
import pytest

from spark_code.codesearch import (
    MAX_INDEX_FILE_BYTES,
    CodeSearchTool,
    _chunk_text,
    collection_name_for,
    format_code_search_results,
    index_project,
    resolve_service_url,
    search_project,
    unavailable_message,
)

# ---------------------------------------------------------------------------
# collection_name_for
# ---------------------------------------------------------------------------


def test_collection_name_deterministic(tmp_path):
    a = collection_name_for(str(tmp_path))
    b = collection_name_for(str(tmp_path))
    assert a == b
    assert a.startswith("spark_code_")


def test_collection_name_differs_per_project(tmp_path):
    p1 = tmp_path / "proj1"
    p2 = tmp_path / "proj2"
    p1.mkdir()
    p2.mkdir()
    assert collection_name_for(str(p1)) != collection_name_for(str(p2))


def test_collection_name_resolves_symlinks(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert collection_name_for(str(real)) == collection_name_for(str(link))


# ---------------------------------------------------------------------------
# resolve_service_url
# ---------------------------------------------------------------------------


def test_resolve_service_url_uses_config_override():
    config = {"rag": {"service_url": "http://custom:9999"}}
    assert resolve_service_url(config) == "http://custom:9999"


def test_resolve_service_url_falls_back_without_config():
    assert resolve_service_url(None).startswith("http")


def test_resolve_service_url_falls_back_on_empty_config():
    assert resolve_service_url({}).startswith("http")


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_empty_returns_nothing():
    assert _chunk_text("") == []


def test_chunk_text_single_small_chunk_starts_at_line_1():
    chunks = _chunk_text("line one\nline two\nline three")
    assert len(chunks) == 1
    start_line, text = chunks[0]
    assert start_line == 1
    assert "line one" in text and "line three" in text


def test_chunk_text_splits_at_target_and_tracks_start_line():
    # Each line ~20 chars; target 50 chars should force a split after ~2 lines.
    lines = [f"x = {i:03d}  # padding" for i in range(20)]
    text = "\n".join(lines)
    chunks = _chunk_text(text, target_chars=50)
    assert len(chunks) > 1
    # start_line values are strictly increasing and 1-indexed.
    starts = [s for s, _ in chunks]
    assert starts[0] == 1
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


# ---------------------------------------------------------------------------
# format_code_search_results / unavailable_message
# ---------------------------------------------------------------------------


def test_format_results_uses_metadata_path_and_start_line():
    results = [
        {
            "text": "def hello():\n    print('hi')",
            "source": "rag_ingest_synthetic.txt",
            "metadata": {"path": "spark_code/greet.py", "start_line": 42},
            "score": 0.9,
        }
    ]
    out = format_code_search_results(results)
    assert "spark_code/greet.py:42" in out
    assert "def hello():" in out
    assert "—" in out


def test_format_results_falls_back_to_source_without_metadata():
    results = [{"text": "some text", "source": "some_source.txt"}]
    out = format_code_search_results(results)
    assert "some_source.txt:1" in out


def test_format_results_empty_list_is_empty_string():
    assert format_code_search_results([]) == ""


def test_unavailable_message_mentions_index_and_url():
    msg = unavailable_message("http://spark-4a54.local:8010")
    assert "/index" in msg
    assert "http://spark-4a54.local:8010" in msg


# ---------------------------------------------------------------------------
# search_project (direct)
# ---------------------------------------------------------------------------


def _search_transport(results, status=200):
    def handler(request):
        return httpx.Response(status, json={"results": results, "query": "q",
                                            "collection": "c", "search_type": "hybrid",
                                            "total_results": len(results), "elapsed_ms": 1.0})
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_search_project_returns_results():
    results = [{"text": "hit", "metadata": {"path": "a.py", "start_line": 1}}]
    got, err = await search_project(
        "http://fake", "coll", "query", transport=_search_transport(results))
    assert err is None
    assert got == results


@pytest.mark.asyncio
async def test_search_project_connect_error_returns_error_not_raise():
    def handler(request):
        raise httpx.ConnectError("boom")

    got, err = await search_project(
        "http://fake", "coll", "query", transport=httpx.MockTransport(handler))
    assert got == []
    assert err is not None


@pytest.mark.asyncio
async def test_search_project_http_error_status_returns_error():
    got, err = await search_project(
        "http://fake", "coll", "query", transport=_search_transport([], status=500))
    assert got == []
    assert err is not None


# ---------------------------------------------------------------------------
# CodeSearchTool.execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_search_tool_formats_results():
    results = [{"text": "def foo():\n    pass", "metadata": {"path": "x/foo.py", "start_line": 7}}]
    tool = CodeSearchTool(cwd="/tmp/some-project", transport=_search_transport(results))
    out = await tool.execute(query="where is foo defined")
    assert "x/foo.py:7" in out
    assert "def foo():" in out


@pytest.mark.asyncio
async def test_code_search_tool_unreachable_returns_message_not_exception():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    tool = CodeSearchTool(cwd="/tmp/some-project", transport=httpx.MockTransport(handler))
    out = await tool.execute(query="anything")
    assert "unavailable" in out.lower()
    assert "/index" in out


@pytest.mark.asyncio
async def test_code_search_tool_empty_results_returns_unavailable_message():
    tool = CodeSearchTool(cwd="/tmp/some-project", transport=_search_transport([]))
    out = await tool.execute(query="nothing matches this")
    assert "unavailable" in out.lower()


@pytest.mark.asyncio
async def test_code_search_tool_disabled_never_calls_network():
    def handler(request):
        raise AssertionError("disabled tool must not make a network call")

    tool = CodeSearchTool(cwd="/tmp/some-project", enabled=False,
                          transport=httpx.MockTransport(handler))
    out = await tool.execute(query="anything")
    assert "unavailable" in out.lower()


@pytest.mark.asyncio
async def test_code_search_tool_surfaces_real_error_on_http_500():
    """Fix 8: a 500 from the service is NOT 'never indexed' — surface the real
    error rather than misdiagnosing the user with the bare re-index nudge."""
    tool = CodeSearchTool(cwd="/tmp/some-project", transport=_search_transport([], status=500))
    out = await tool.execute(query="anything")
    assert "500" in out
    assert "unavailable" in out.lower()


@pytest.mark.asyncio
async def test_code_search_tool_surfaces_transport_error_detail():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    tool = CodeSearchTool(cwd="/tmp/some-project", transport=httpx.MockTransport(handler))
    out = await tool.execute(query="anything")
    # The transport detail is surfaced (distinguishes service-down from
    # never-indexed) while still hinting /index.
    assert "route" in out.lower()
    assert "/index" in out


def test_code_search_tool_metadata():
    tool = CodeSearchTool(cwd="/tmp/some-project")
    assert tool.name == "code_search"
    assert tool.is_read_only is True
    assert tool.requires_permission is False
    assert "query" in tool.parameters["properties"]
    assert tool.parameters["required"] == ["query"]


# ---------------------------------------------------------------------------
# index_project
# ---------------------------------------------------------------------------


def _index_transport(record):
    """Records every /ingest_text payload; serves /health and /rebuild_bm25 too."""

    def handler(request):
        path = request.url.path
        if request.method == "GET" and path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "POST" and path == "/ingest_text":
            body = json.loads(request.content)
            record.append(body)
            return httpx.Response(200, json={
                "filename": "x", "chunks": 1, "collection": body.get("collection"),
            })
        if request.method == "POST" and path == "/rebuild_bm25":
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_index_project_indexes_small_text_file(tmp_path):
    (tmp_path / "real_file.py").write_text("def hello():\n    return 42\n")
    record = []
    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=_index_transport(record))
    assert "error" not in result
    assert result["indexed"] == 1
    assert result["skipped"] == 0
    assert result["collection"] == collection_name_for(str(tmp_path))
    assert len(record) == 1
    assert record[0]["extra_metadata"]["path"] == "real_file.py"
    assert record[0]["extra_metadata"]["start_line"] == 1


@pytest.mark.asyncio
async def test_index_project_respects_size_cap(tmp_path):
    (tmp_path / "small.py").write_text("x = 1\n")
    big_content = "x = 1\n" * (MAX_INDEX_FILE_BYTES // 6 + 100)
    (tmp_path / "huge.py").write_text(big_content)
    record = []
    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=_index_transport(record))
    assert result["indexed"] == 1
    assert result["skipped"] == 1
    indexed_paths = {r["extra_metadata"]["path"] for r in record}
    assert indexed_paths == {"small.py"}


@pytest.mark.asyncio
async def test_index_project_skips_binary_files(tmp_path):
    (tmp_path / "text.py").write_text("x = 1\n")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02binary\x00garbage")
    record = []
    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=_index_transport(record))
    assert result["indexed"] == 1
    assert result["skipped"] == 1
    assert {r["extra_metadata"]["path"] for r in record} == {"text.py"}


@pytest.mark.asyncio
async def test_index_project_skips_venv_node_modules_git(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")
    for skip_dir, fname in [(".git", "config"), ("node_modules", "index.js"),
                            ("venv", "site.py"), (".venv", "site.py")]:
        d = tmp_path / skip_dir
        d.mkdir()
        (d / fname).write_text("should never be indexed\n")
    record = []
    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=_index_transport(record))
    assert result["indexed"] == 1
    indexed_paths = {r["extra_metadata"]["path"] for r in record}
    assert indexed_paths == {"real.py"}


@pytest.mark.asyncio
async def test_index_project_respects_gitignore(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("ignored_dir\n")
    ignored = tmp_path / "ignored_dir"
    ignored.mkdir()
    (ignored / "skip_me.py").write_text("should never be indexed\n")
    record = []
    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=_index_transport(record))
    indexed_paths = {r["extra_metadata"]["path"] for r in record}
    assert indexed_paths == {"real.py"}
    assert result["indexed"] == 1


@pytest.mark.asyncio
async def test_index_project_empty_dir_returns_zero_indexed(tmp_path):
    record = []
    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=_index_transport(record))
    assert result == {"indexed": 0, "skipped": 0, "collection": collection_name_for(str(tmp_path))}


@pytest.mark.asyncio
async def test_index_project_unreachable_service_returns_error_not_exception(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")

    def handler(request):
        raise httpx.ConnectError("no route to host")

    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=httpx.MockTransport(handler))
    assert "error" in result
    assert isinstance(result["error"], str)


@pytest.mark.asyncio
async def test_index_project_non_200_health_returns_error(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")

    def handler(request):
        return httpx.Response(503, json={"detail": "starting up"})

    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=httpx.MockTransport(handler))
    assert "error" in result


@pytest.mark.asyncio
async def test_index_project_health_probe_runs_before_file_walk(tmp_path, monkeypatch):
    """Fix 7: an unreachable service must fail fast — the (potentially large)
    tree walk + per-file _is_binary probes must NOT happen before /health."""
    import spark_code.codesearch as cs

    (tmp_path / "real.py").write_text("x = 1\n")
    walked = []
    orig = cs._iter_source_files

    def spy(root):
        walked.append(root)
        return orig(root)

    monkeypatch.setattr(cs, "_iter_source_files", spy)

    def handler(request):
        raise httpx.ConnectError("no route to host")  # health probe fails

    result = await index_project(str(tmp_path), service_url="http://fake",
                                 transport=httpx.MockTransport(handler))
    assert "error" in result
    assert walked == []  # never walked the tree — failed fast on health


@pytest.mark.asyncio
async def test_index_project_uses_explicit_collection_override(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")
    record = []
    result = await index_project(str(tmp_path), service_url="http://fake",
                                 collection="my_custom_collection",
                                 transport=_index_transport(record))
    assert result["collection"] == "my_custom_collection"
    assert record[0]["collection"] == "my_custom_collection"
