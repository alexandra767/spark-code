"""RAG-enhanced project planning — keyword extraction and RAG query builder."""

import os
import re

STOP_WORDS = frozenset({
    "a", "an", "the", "to", "for", "in", "with", "and", "or", "but",
    "is", "it", "of", "on", "at", "by", "from", "as", "be", "was",
    "are", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "that", "this", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "they", "them", "its", "not", "no",
    "so", "if", "then", "than", "too", "very", "just", "about",
    "up", "out", "into", "over", "after", "before", "between",
    "add", "create", "make", "build", "implement", "write", "set",
})


def extract_keywords(prompt: str) -> list[str]:
    """Extract meaningful keywords from a user prompt, stripping stop words."""
    words = re.findall(r"[a-zA-Z]+", prompt.lower())
    return [w for w in words if w not in STOP_WORDS]


def build_rag_queries(keywords: list[str], project_type: str) -> list[str]:
    """Build RAG search queries based on keywords and detected project type.

    Returns 2-4 queries tailored to the project type.
    """
    if not keywords:
        return []

    kw_str = " ".join(keywords)
    project_lower = project_type.lower()

    if "swift" in project_lower or "xcode" in project_lower:
        return [
            f"HIG {kw_str}",
            f"SwiftUI {kw_str}",
            f"App Store guidelines {kw_str}",
            f"Swift {kw_str} best practices",
        ]
    elif "python" in project_lower:
        return [
            f"{kw_str} patterns",
            f"{kw_str} best practices",
        ]
    elif "javascript" in project_lower or "typescript" in project_lower:
        return [
            f"{kw_str} patterns",
            f"{kw_str} best practices",
        ]
    else:
        # Unknown project type — generic queries
        return [
            kw_str,
            f"{kw_str} best practices",
        ]
