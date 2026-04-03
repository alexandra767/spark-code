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
