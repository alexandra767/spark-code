# `/projectplan` Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/projectplan` command to Spark Code that researches the RAG knowledge base before generating implementation plans, and injects matched references into worker prompts during execution.

**Architecture:** New `projectplan.py` module handles keyword extraction and RAG querying. The `/projectplan` CLI handler orchestrates: detect project type → fire RAG queries → inject results into model prompt → model writes `projectplan.md`. Enhanced `plan_executor.py` parses `[Ref N]` tags and prepends matched documentation to worker task descriptions.

**Tech Stack:** Python, httpx (existing dep), RAG service at `192.168.1.187:8010`

---

### Task 1: Create `projectplan.py` — keyword extraction

**Files:**
- Create: `spark_code/projectplan.py`
- Test: `tests/test_projectplan.py`

- [ ] **Step 1: Write failing test for `extract_keywords`**

```python
# tests/test_projectplan.py
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
    assert result == ["build", "navigation", "tab", "bar"]


def test_extract_keywords_lowercases():
    result = extract_keywords("Add Settings Screen")
    assert result == ["settings", "screen"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spark_code.projectplan'`

- [ ] **Step 3: Implement `extract_keywords`**

```python
# spark_code/projectplan.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add spark_code/projectplan.py tests/test_projectplan.py
git commit -m "feat(projectplan): add keyword extraction with stop word filtering"
```

---

### Task 2: Add RAG query builder and fetcher to `projectplan.py`

**Files:**
- Modify: `spark_code/projectplan.py`
- Modify: `tests/test_projectplan.py`

- [ ] **Step 1: Write failing test for `build_rag_queries`**

Append to `tests/test_projectplan.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: 4 new tests FAIL — `ImportError: cannot import name 'build_rag_queries'`

- [ ] **Step 3: Implement `build_rag_queries`**

Add to `spark_code/projectplan.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add spark_code/projectplan.py tests/test_projectplan.py
git commit -m "feat(projectplan): add RAG query builder with project type detection"
```

---

### Task 3: Add `fetch_rag_context` async function to `projectplan.py`

**Files:**
- Modify: `spark_code/projectplan.py`
- Modify: `tests/test_projectplan.py`

- [ ] **Step 1: Write failing test for `format_references`**

We test the formatting function separately from the HTTP call. Append to `tests/test_projectplan.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py::test_format_references_basic -v`
Expected: FAIL — `ImportError: cannot import name 'format_references'`

- [ ] **Step 3: Implement `format_references` and `fetch_rag_context`**

Add to `spark_code/projectplan.py`:

```python
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://192.168.1.187:8010")
MAX_REFS = 8


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


def fetch_rag_context(keywords: list[str], project_type: str) -> str:
    """Fire RAG queries and return formatted reference material.

    Uses synchronous httpx since this runs in the slash command handler
    which is called from within an already-running async event loop.

    Returns a formatted '## Reference Material' section string,
    or empty string if RAG is unreachable or returns no results.
    """
    import httpx

    queries = build_rag_queries(keywords, project_type)
    if not queries:
        return ""

    all_results = []

    try:
        with httpx.Client(timeout=30.0) as client:
            for query in queries:
                payload = {
                    "query": query,
                    "collection": "claude_documents",
                    "n_results": 5,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add spark_code/projectplan.py tests/test_projectplan.py
git commit -m "feat(projectplan): add RAG context fetcher with dedup and formatting"
```

---

### Task 4: Add `/projectplan` command handler to `cli.py`

**Files:**
- Modify: `spark_code/cli.py:31` (add import)
- Modify: `spark_code/cli.py:483` (add to help text)
- Modify: `spark_code/cli.py:1007` (add command handler after `/plan`)

- [ ] **Step 1: Add import**

At `spark_code/cli.py:31`, after the existing `from .plan_executor import execute_plan` line, add:

```python
from .projectplan import extract_keywords, fetch_rag_context
```

- [ ] **Step 2: Add to help text**

At `spark_code/cli.py:483`, after the line `- /plan <prompt> — Create plan.md / /plan show / /plan go`, add:

```python
- `/projectplan <prompt>` — RAG-researched plan / `/projectplan show` / `/projectplan go`
```

- [ ] **Step 3: Add the `/projectplan` command handler**

At `spark_code/cli.py:1007`, after the `/plan` `elif` block ends and before the `elif command == "/publish":` line, insert:

```python
    elif command == "/projectplan":
        if not args:
            console.print("[#ebcb8b]Usage: /projectplan <prompt> — create a RAG-researched plan[/#ebcb8b]")
            console.print("[#8899aa]  /projectplan show    — show current plan[/#8899aa]")
            console.print("[#8899aa]  /projectplan copy    — copy plan to clipboard[/#8899aa]")
            console.print("[#8899aa]  /projectplan go      — execute the approved plan[/#8899aa]")
            return None

        sub = args.strip().split(maxsplit=1)
        sub_cmd = sub[0].lower()

        pp_path = os.path.join(os.getcwd(), "projectplan.md")

        if sub_cmd == "show":
            if not os.path.exists(pp_path):
                console.print("[#8899aa]No projectplan.md found. Create one with /projectplan <prompt>[/#8899aa]")
                return None
            with open(pp_path) as f:
                content = f.read()
            console.print()
            console.print(Panel(
                Markdown(content),
                title="[bold #88c0d0] projectplan.md [/bold #88c0d0]",
                border_style="#4c566a",
                box=ROUNDED,
                padding=(1, 2),
            ))
            _copy_to_clipboard(content)
            console.print()
            console.print("[#a3be8c]  ✓ Copied to clipboard  ·  /projectplan go to execute  ·  /projectplan <prompt> to redo[/#a3be8c]")
            return None

        elif sub_cmd == "copy":
            if not os.path.exists(pp_path):
                console.print("[#8899aa]No projectplan.md found. Create one with /projectplan <prompt>[/#8899aa]")
                return None
            with open(pp_path) as f:
                content = f.read()
            if _copy_to_clipboard(content):
                console.print("[#a3be8c]✓ Project plan copied to clipboard[/#a3be8c]")
            else:
                console.print("[#bf616a]Could not copy — pbcopy not available[/#bf616a]")
            return None

        elif sub_cmd == "go":
            if not os.path.exists(pp_path):
                console.print("[#bf616a]No projectplan.md found. Create one first with /projectplan <prompt>[/#bf616a]")
                return None
            with open(pp_path) as f:
                plan_content = f.read()
            return f"__PLAN_EXECUTE__{plan_content}"

        else:
            # /projectplan <prompt> — research RAG, then create plan
            plan_prompt = args
            project_type = detect_project_type(os.getcwd())

            # Extract keywords and fetch RAG context
            keywords = extract_keywords(plan_prompt)
            console.print(f"[#88c0d0]▸ Researching docs for: {', '.join(keywords) or plan_prompt}[/#88c0d0]")

            rag_context = fetch_rag_context(keywords, project_type)

            if rag_context:
                console.print(f"[#a3be8c]  ✓ Found relevant documentation[/#a3be8c]")
            else:
                console.print(f"[#ebcb8b]  ⚠ No RAG results (service down or no matches)[/#ebcb8b]")

            rag_section = ""
            if rag_context:
                rag_section = (
                    "IMPORTANT — I have pre-researched the following documentation from the knowledge base. "
                    "You MUST include this as the '## Reference Material' section at the top of the plan, "
                    "and tag relevant steps with [see Ref N] markers.\n\n"
                    f"{rag_context}\n\n"
                    "---\n\n"
                )

            create_plan_prompt = (
                f"The user wants you to create a detailed, RAG-informed project plan. "
                f"Their request: {plan_prompt}\n\n"
                f"Detected project type: {project_type or 'unknown'}\n\n"
                f"{rag_section}"
                "RULES:\n"
                "1. Use read-only tools to explore the codebase first (read_file, glob, grep, list_dir)\n"
                "2. Do NOT create project files yet — only create projectplan.md\n"
                "3. Create a detailed implementation plan\n"
                "4. You MUST use the write_file tool to save the plan to 'projectplan.md' in the current directory\n"
                "   This is the ONE file you must write. Do not skip this step.\n\n"
                "The projectplan.md MUST include (in this order):\n"
                "- ## Reference Material — the pre-researched docs above (keep the [Ref N] format exactly)\n"
                "- --- (horizontal rule separator)\n"
                "- ## Summary — what will be done\n"
                "- ## Steps — numbered steps with clear descriptions. Tag steps with [see Ref N, Ref M] "
                "where the referenced documentation is relevant to that step's implementation\n"
                "- ## Parallelization — which steps can run in parallel\n"
                "- ## Files — files to be modified or created\n"
                "- ## Risks — any risks or considerations\n\n"
                "You may also use the rag_search tool to find additional documentation if the "
                "pre-researched material doesn't cover everything you need.\n\n"
                "After writing projectplan.md, tell the user: Review with /projectplan show, then /projectplan go to execute."
            )
            return create_plan_prompt
```

- [ ] **Step 4: Run Spark Code to verify the command is recognized**

Run: `cd /Users/alexandratitus/spark-code && python -m spark_code.cli --help 2>/dev/null || echo "check syntax"`

Manually test: launch spark, type `/projectplan` with no args, verify usage text appears.

- [ ] **Step 5: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add spark_code/cli.py
git commit -m "feat(projectplan): add /projectplan command with RAG orchestration"
```

---

### Task 5: Enhance `plan_executor.py` — parse references from plan text

**Files:**
- Modify: `spark_code/plan_executor.py`
- Modify: `tests/test_projectplan.py`

- [ ] **Step 1: Write failing test for `parse_references`**

Append to `tests/test_projectplan.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py::test_parse_references_basic -v`
Expected: FAIL — `ImportError: cannot import name 'parse_references'`

- [ ] **Step 3: Implement `parse_references`**

Add to `spark_code/plan_executor.py`, after the existing `parse_plan` function (after line 83):

```python
def parse_references(plan_text: str) -> dict[int, dict]:
    """Parse [Ref N] blocks from ## Reference Material section.

    Returns {ref_number: {"title": str, "text": str}} dict.
    """
    refs = {}
    lines = plan_text.split("\n")
    in_ref_section = False
    current_ref_num = None
    current_title = ""
    current_text_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Detect start of reference material section
        if stripped.startswith("## ") or stripped.startswith("# "):
            header_lower = stripped.lower()
            if "reference material" in header_lower:
                in_ref_section = True
                continue
            elif in_ref_section:
                # Hit the next section — save last ref and stop
                if current_ref_num is not None:
                    refs[current_ref_num] = {
                        "title": current_title,
                        "text": "\n".join(current_text_lines).strip(),
                    }
                break

        if not in_ref_section:
            continue

        # Horizontal rule = end of ref section
        if stripped == "---":
            if current_ref_num is not None:
                refs[current_ref_num] = {
                    "title": current_title,
                    "text": "\n".join(current_text_lines).strip(),
                }
            break

        # Match [Ref N] header line
        ref_match = re.match(r"\[Ref\s+(\d+)\]\s*\*{0,2}(.+?)\*{0,2}\s*(?:\(.*\))?\s*$", stripped)
        if ref_match:
            # Save previous ref
            if current_ref_num is not None:
                refs[current_ref_num] = {
                    "title": current_title,
                    "text": "\n".join(current_text_lines).strip(),
                }
            current_ref_num = int(ref_match.group(1))
            current_title = ref_match.group(2).strip()
            current_text_lines = []
        elif current_ref_num is not None and stripped:
            # Collect blockquote text (strip leading >)
            text = stripped.lstrip("> ").strip()
            if text:
                current_text_lines.append(text)

    # Save last ref if we didn't hit a break
    if current_ref_num is not None and current_ref_num not in refs:
        refs[current_ref_num] = {
            "title": current_title,
            "text": "\n".join(current_text_lines).strip(),
        }

    return refs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add spark_code/plan_executor.py tests/test_projectplan.py
git commit -m "feat(plan_executor): add reference material parser for projectplan.md"
```

---

### Task 6: Enhance `plan_executor.py` — extract ref tags from steps

**Files:**
- Modify: `spark_code/plan_executor.py`
- Modify: `tests/test_projectplan.py`

- [ ] **Step 1: Write failing test for `extract_step_refs`**

Append to `tests/test_projectplan.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py::test_extract_step_refs_from_title -v`
Expected: FAIL — `ImportError: cannot import name 'extract_step_refs'`

- [ ] **Step 3: Implement `extract_step_refs`**

Add to `spark_code/plan_executor.py`, after `parse_references`:

```python
def extract_step_refs(title: str, body: str) -> set[int]:
    """Extract [see Ref N] numbers from a step's title and body."""
    combined = f"{title} {body}"
    return {int(m.group(1)) for m in re.finditer(r"Ref\s+(\d+)", combined)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: All 19 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add spark_code/plan_executor.py tests/test_projectplan.py
git commit -m "feat(plan_executor): add ref tag extraction from step title/body"
```

---

### Task 7: Enhance `plan_executor.py` — inject refs into worker prompts

**Files:**
- Modify: `spark_code/plan_executor.py:93-229` (the `execute_plan` function)

- [ ] **Step 1: Write failing test for ref-aware task description building**

Append to `tests/test_projectplan.py`:

```python
from spark_code.plan_executor import build_task_desc


def test_build_task_desc_with_refs():
    refs = {
        1: {"title": "HIG Settings", "text": "Use grouped lists for settings."},
        2: {"title": "SwiftUI Nav", "text": "NavigationStack replaces NavigationView."},
    }
    step = {"number": 1, "title": "Create SettingsView [see Ref 1, Ref 2]", "body": "Build the UI."}
    desc = build_task_desc(step, refs)
    assert "## Relevant Documentation" in desc
    assert "Use grouped lists" in desc
    assert "NavigationStack" in desc
    assert "## Task:" in desc


def test_build_task_desc_no_refs():
    refs = {
        1: {"title": "HIG Settings", "text": "Use grouped lists."},
    }
    step = {"number": 2, "title": "Add data model", "body": "Create the model."}
    desc = build_task_desc(step, refs)
    assert "## Relevant Documentation" not in desc
    assert "## Task:" in desc


def test_build_task_desc_empty_refs_dict():
    step = {"number": 1, "title": "Do something [see Ref 1]", "body": "Details."}
    desc = build_task_desc(step, {})
    assert "## Relevant Documentation" not in desc
    assert "## Task:" in desc
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py::test_build_task_desc_with_refs -v`
Expected: FAIL — `ImportError: cannot import name 'build_task_desc'`

- [ ] **Step 3: Implement `build_task_desc` and update `execute_plan`**

Add `build_task_desc` to `spark_code/plan_executor.py`, after `extract_step_refs`:

```python
def build_task_desc(step: dict, refs: dict[int, dict]) -> str:
    """Build a task description for a worker, injecting matched references.

    If the step has [see Ref N] tags and those refs exist, prepends a
    '## Relevant Documentation' block. Otherwise returns a plain task desc.
    """
    ref_nums = extract_step_refs(step["title"], step["body"])
    matched = {n: refs[n] for n in ref_nums if n in refs}

    parts = []
    if matched:
        parts.append("## Relevant Documentation\n")
        for n in sorted(matched):
            r = matched[n]
            parts.append(f"**[Ref {n}] {r['title']}**")
            parts.append(f"> {r['text']}\n")
        parts.append("---\n")
        parts.append("Follow the documentation above when implementing this task.\n")

    parts.append(f"## Task: {step['title']}\n")
    parts.append(f"{step['body']}\n")
    parts.append("Instructions:")
    parts.append("- Create the file(s) described above")
    parts.append("- Write complete, working code with imports")
    parts.append("- Include docstrings and basic error handling")
    parts.append("- If the task mentions tests, write real tests with pytest")

    return "\n".join(parts)
```

Now modify the `execute_plan` function to use ref-aware task descriptions. Replace the existing worker task_desc building (lines 147-155) and the sequential task_desc building (lines 202-208).

In `execute_plan`, after `steps, parallel_nums = parse_plan(plan_text)` on line 100, add:

```python
    # Parse references if this is a projectplan with ## Reference Material
    refs = parse_references(plan_text) if "## Reference Material" in plan_text else {}
```

Replace the parallel worker task_desc block (lines 147-155) with:

```python
                task_desc = build_task_desc(s, refs)
```

Replace the sequential task_desc block (lines 202-208) with:

```python
            if refs:
                task_desc = (
                    "Execute this step of the project plan:\n\n"
                    + build_task_desc(step, refs)
                    + "\n\nComplete this step fully. "
                    "Create any files or run any commands needed."
                )
            else:
                task_desc = (
                    f"Execute this step of the project plan:\n\n"
                    f"## {step['title']}\n\n"
                    f"{step['body']}\n\n"
                    f"Complete this step fully. "
                    f"Create any files or run any commands needed."
                )
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: All 22 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add spark_code/plan_executor.py tests/test_projectplan.py
git commit -m "feat(plan_executor): inject matched references into worker task descriptions"
```

---

### Task 8: Update system prompt and add `/projectplan` to help

**Files:**
- Modify: `spark_code/context.py:20` (system prompt)

- [ ] **Step 1: Add `/projectplan` mention to system prompt**

In `spark_code/context.py`, in the `SYSTEM_PROMPT` string after line 20 (`- rag_search: ...`), the RAG section already exists. After the existing line about rag_search, no change needed there. Instead, find the line:

```
- Plan before executing complex tasks
```

And change it to:

```
- Plan before executing complex tasks (use /projectplan for RAG-researched plans)
```

- [ ] **Step 2: Verify no syntax errors**

Run: `cd /Users/alexandratitus/spark-code && python -c "from spark_code.context import SYSTEM_PROMPT; print('OK')" `
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add spark_code/context.py
git commit -m "feat(context): mention /projectplan in system prompt"
```

---

### Task 9: Integration test — end-to-end verify

**Files:**
- Modify: `tests/test_projectplan.py`

- [ ] **Step 1: Write integration test for full pipeline**

Append to `tests/test_projectplan.py`:

```python
import pytest
from spark_code.projectplan import extract_keywords, build_rag_queries, format_references
from spark_code.plan_executor import parse_plan, parse_references, extract_step_refs, build_task_desc


def test_full_pipeline_ios():
    """End-to-end: prompt → keywords → queries → format refs → plan parse → ref inject."""
    # Step 1: Extract keywords
    keywords = extract_keywords("add a settings screen to GigLedger")
    assert "settings" in keywords
    assert "screen" in keywords

    # Step 2: Build queries
    queries = build_rag_queries(keywords, "Swift project")
    assert len(queries) >= 3
    assert any("HIG" in q for q in queries)

    # Step 3: Format mock RAG results
    mock_results = [
        {"source": "apple-hig.pdf", "text": "Use grouped lists for settings.",
         "citation": {"page": 42}, "score": 0.89},
        {"source": "swiftui-docs", "text": "Use NavigationStack for navigation.",
         "citation": {"page": 15}, "score": 0.85},
    ]
    ref_section = format_references(mock_results)
    assert "[Ref 1]" in ref_section
    assert "[Ref 2]" in ref_section

    # Step 4: Build a full projectplan.md
    plan_text = f"""# Project Plan: GigLedger Settings

{ref_section}

---

## Summary
Add a settings screen to GigLedger.

## Steps

1. **Create SettingsView** [see Ref 1, Ref 2]
   - Build the settings screen with grouped lists

2. **Add data model**
   - Create UserPreferences SwiftData model

3. **Wire to tab bar** [see Ref 2]
   - Add settings tab

## Parallelization
- 2
- 3

## Files
- SettingsView.swift
"""

    # Step 5: Parse the plan
    steps, parallel_nums = parse_plan(plan_text)
    assert len(steps) == 3
    assert parallel_nums == {2, 3}

    # Step 6: Parse references
    refs = parse_references(plan_text)
    assert 1 in refs
    assert 2 in refs

    # Step 7: Build ref-injected task desc for step 1
    desc = build_task_desc(steps[0], refs)
    assert "## Relevant Documentation" in desc
    assert "grouped lists" in desc
    assert "NavigationStack" in desc

    # Step 8: Step 2 has no refs — no injection
    desc2 = build_task_desc(steps[1], refs)
    assert "## Relevant Documentation" not in desc2
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/test_projectplan.py -v`
Expected: All 23 tests PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/alexandratitus/spark-code
git add tests/test_projectplan.py
git commit -m "test: add end-to-end integration test for /projectplan pipeline"
```

---

### Task 10: Final cleanup and manual smoke test

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/alexandratitus/spark-code && python -m pytest tests/ -v`
Expected: All tests PASS (including any pre-existing tests)

- [ ] **Step 2: Smoke test in live CLI**

Launch: `cd /Users/alexandratitus/spark-code && python -m spark_code.cli`

Test these commands:
1. `/projectplan` (no args) — should show usage
2. `/projectplan show` — should say "No projectplan.md found"
3. `/plan show` — should still work as before (unchanged)

- [ ] **Step 3: Final commit with any fixups**

If any fixes were needed during smoke test:

```bash
cd /Users/alexandratitus/spark-code
git add -A
git commit -m "fix: address smoke test issues in /projectplan"
```
