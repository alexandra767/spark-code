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
