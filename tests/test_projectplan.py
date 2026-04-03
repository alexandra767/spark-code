"""Tests for projectplan module — RAG-enhanced project planning."""

from spark_code.projectplan import extract_keywords


def test_extract_keywords_strips_stop_words():
    result = extract_keywords("add a settings screen to the app")
    assert result == ["settings", "screen", "app"]


def test_extract_keywords_empty_prompt():
    result = extract_keywords("")
    assert result == []


def test_extract_keywords_all_stop_words():
    result = extract_keywords("the a to for in with")
    assert result == []


def test_extract_keywords_preserves_order():
    result = extract_keywords("build navigation and tab bar")
    assert result == ["navigation", "tab", "bar"]


def test_extract_keywords_lowercases():
    result = extract_keywords("Add Settings Screen")
    assert result == ["settings", "screen"]


from spark_code.projectplan import build_rag_queries


def test_build_rag_queries_ios():
    queries = build_rag_queries(["settings", "screen"], "Swift project")
    assert any("HIG" in q for q in queries)
    assert any("SwiftUI" in q for q in queries)
    assert any("App Store" in q for q in queries)
    assert len(queries) >= 3


def test_build_rag_queries_python():
    queries = build_rag_queries(["auth", "middleware"], "Python + FastAPI project")
    assert any("auth middleware" in q.lower() for q in queries)
    assert len(queries) >= 2


def test_build_rag_queries_unknown():
    queries = build_rag_queries(["database", "schema"], "")
    assert len(queries) >= 1
    assert any("database" in q.lower() for q in queries)


def test_build_rag_queries_no_keywords():
    queries = build_rag_queries([], "Swift project")
    assert queries == []
