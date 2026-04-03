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


from spark_code.projectplan import format_references


def test_format_references_basic():
    raw_results = [
        {"source": "apple-hig.pdf", "text": "Use grouped lists for settings.",
         "citation": {"page": 42}, "score": 0.89},
        {"source": "swiftui-docs", "text": "NavigationStack replaces NavigationView.",
         "citation": {"page": 15}, "score": 0.85},
    ]
    output = format_references(raw_results)
    assert "[Ref 1]" in output
    assert "[Ref 2]" in output
    assert "apple-hig.pdf" in output
    assert "p.42" in output
    assert "Use grouped lists" in output


def test_format_references_deduplicates():
    raw_results = [
        {"source": "hig.pdf", "text": "Same content.", "citation": {"page": 1}, "score": 0.9},
        {"source": "hig.pdf", "text": "Same content.", "citation": {"page": 1}, "score": 0.7},
        {"source": "other.pdf", "text": "Different.", "citation": {"page": 2}, "score": 0.8},
    ]
    output = format_references(raw_results)
    assert output.count("[Ref") == 2  # deduplicated to 2


def test_format_references_caps_at_eight():
    raw_results = [
        {"source": f"doc{i}.pdf", "text": f"Content {i}.", "citation": {"page": i}, "score": 0.9 - i * 0.05}
        for i in range(12)
    ]
    output = format_references(raw_results)
    assert "[Ref 8]" in output
    assert "[Ref 9]" not in output


def test_format_references_empty():
    output = format_references([])
    assert output == ""
