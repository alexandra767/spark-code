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


def _detect_platform_from_prompt(prompt: str) -> str:
    """Detect platform/language hints from the user's prompt text.

    Returns a synthetic project_type string if a platform is mentioned,
    or empty string if none detected.
    """
    lower = prompt.lower()
    # Check for iOS/Swift hints
    if any(kw in lower for kw in ["ios", "swiftui", "swift", "iphone", "ipad",
                                   "xcode", "apple", "watchos", "macos", "tvos",
                                   "uikit", "appkit", "widget"]):
        return "Swift project"
    # Check for Android/Kotlin hints
    if any(kw in lower for kw in ["android", "kotlin", "jetpack", "compose"]):
        return "Kotlin project"
    # Check for Python hints
    if any(kw in lower for kw in ["python", "django", "flask", "fastapi", "pytest"]):
        return "Python project"
    # Check for JS/TS hints
    if any(kw in lower for kw in ["react", "nextjs", "next.js", "vue", "angular",
                                   "typescript", "javascript", "node", "express"]):
        return "TypeScript project"
    return ""


def build_rag_queries(keywords: list[str], project_type: str,
                      prompt: str = "") -> list[str]:
    """Build RAG search queries based on keywords and detected project type.

    Falls back to checking the prompt for platform hints if project_type
    is empty (e.g. empty directory for a new project).

    Returns 2-4 queries tailored to the project type.
    """
    if not keywords:
        return []

    kw_str = " ".join(keywords)
    project_lower = project_type.lower()

    # If no project detected from files, check the prompt for platform hints
    if not project_lower and prompt:
        project_lower = _detect_platform_from_prompt(prompt).lower()

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


RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://192.168.1.187:8010")
MAX_REFS = 5


def format_references(raw_results: list[dict]) -> str:
    """Format RAG results as numbered [Ref N] blocks. Deduplicates and caps at MAX_REFS."""
    if not raw_results:
        return ""

    # Deduplicate by (source, page) keeping highest score
    seen = {}
    for r in raw_results:
        source = r.get("source", "unknown")
        page = r.get("citation", {}).get("page", "")
        key = (source, str(page))
        if key not in seen or r.get("score", 0) > seen[key].get("score", 0):
            seen[key] = r

    # Sort by score descending, cap at MAX_REFS
    unique = sorted(seen.values(), key=lambda r: r.get("score", 0), reverse=True)
    unique = unique[:MAX_REFS]

    lines = ["## Reference Material\n"]
    for i, r in enumerate(unique, 1):
        source = r.get("source", "unknown")
        text = r.get("text", "").strip()
        citation = r.get("citation", {})
        page = citation.get("page")
        page_str = f", p.{page}" if page else ""
        score = r.get("score", 0)

        lines.append(f"[Ref {i}] **{source}{page_str}** (score: {score:.2f})")
        lines.append(f"> {text}\n")

    return "\n".join(lines)


def fetch_rag_context(keywords: list[str], project_type: str,
                      prompt: str = "") -> str:
    """Fire RAG queries and return formatted reference material.

    Uses synchronous httpx since this runs in the slash command handler
    which is called from within an already-running async event loop.

    Returns a formatted '## Reference Material' section string,
    or empty string if RAG is unreachable or returns no results.
    """
    import httpx

    queries = build_rag_queries(keywords, project_type, prompt=prompt)
    if not queries:
        return ""

    all_results = []

    try:
        with httpx.Client(timeout=30.0) as client:
            for query in queries:
                payload = {
                    "query": query,
                    "collection": "claude_documents",
                    "n_results": 3,
                    "search_type": "hybrid",
                    "user_role": "owner",
                }
                try:
                    resp = client.post(f"{RAG_SERVICE_URL}/search", json=payload)
                    data = resp.json()
                    all_results.extend(data.get("results", []))
                except Exception:
                    continue
    except httpx.ConnectError:
        return ""
    except Exception:
        return ""

    return format_references(all_results)
