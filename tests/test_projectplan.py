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


from spark_code.plan_executor import parse_references


def test_parse_references_basic():
    plan_text = """## Reference Material

[Ref 1] **apple-hig.pdf, p.42** (score: 0.89)
> Use grouped lists for settings. Each group should have a descriptive header.

[Ref 2] **swiftui-docs, p.15** (score: 0.85)
> NavigationStack replaces NavigationView in iOS 16+.

---

## Summary
Build a settings screen.

## Steps
1. **Create SettingsView** [see Ref 1, Ref 2]
"""
    refs = parse_references(plan_text)
    assert 1 in refs
    assert 2 in refs
    assert "grouped lists" in refs[1]["text"]
    assert "NavigationStack" in refs[2]["text"]


def test_parse_references_no_ref_section():
    plan_text = """## Summary
Build something.

## Steps
1. Do a thing
"""
    refs = parse_references(plan_text)
    assert refs == {}


from spark_code.plan_executor import extract_step_refs


def test_extract_step_refs_from_title():
    nums = extract_step_refs("Create SettingsView [see Ref 1, Ref 2]", "")
    assert nums == {1, 2}


def test_extract_step_refs_from_body():
    nums = extract_step_refs("Do something", "Follow guidelines [see Ref 3]\nMore text")
    assert nums == {3}


def test_extract_step_refs_combined():
    nums = extract_step_refs("Title [see Ref 1]", "Body [see Ref 4, Ref 5]")
    assert nums == {1, 4, 5}


def test_extract_step_refs_none():
    nums = extract_step_refs("No refs here", "Just plain text")
    assert nums == set()
