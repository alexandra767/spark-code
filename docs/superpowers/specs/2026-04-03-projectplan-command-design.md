# `/projectplan` Command — RAG-Enhanced Project Planning

**Date:** 2026-04-03
**Status:** Draft
**Scope:** New `/projectplan` command for Spark Code CLI with RAG integration

## Problem

The existing `/plan` command generates implementation plans by exploring the codebase, but it never queries the RAG knowledge base. For iOS projects especially, this means the AI writes plans without consulting Apple HIG, Swift documentation, App Store Review Guidelines, or SwiftUI patterns — all of which are indexed in the local RAG service (port 8010, `claude_documents` collection).

## Solution

A new `/projectplan` command that:
1. Extracts keywords from the user's prompt
2. Detects project type via existing `project_detect.py`
3. Fires adaptive RAG queries against `claude_documents`
4. Injects the results as pre-researched context into the model prompt
5. The model writes `projectplan.md` with a `## Reference Material` section and `[see Ref N]` tags on steps
6. The enhanced plan executor parses ref tags and injects matched references into worker prompts during execution

## Design

### Command Interface

| Command | Action |
|---------|--------|
| `/projectplan <prompt>` | Research RAG + generate `projectplan.md` |
| `/projectplan show` | Display current `projectplan.md` |
| `/projectplan copy` | Copy to clipboard |
| `/projectplan go` | Execute via enhanced plan executor |

Lives alongside `/plan` — they are independent commands. `/plan` stays fast and lightweight. `/projectplan` is the thorough, research-first version.

### Command Flow

```
User prompt
    │
    ▼
1. Parse prompt → extract keywords (strip stop words)
    │
    ▼
2. Detect project type via project_detect.py
    │
    ▼
3. Build RAG queries (2-3 initial, adaptive):
   iOS → "HIG {kw}", "SwiftUI {kw}", "App Store guidelines {kw}", "Swift {kw} best practices"
   Python → "{kw} patterns"
   Other → "{kw}"
    │
    ▼
4. Fire queries in parallel against claude_documents collection
   - Deduplicate by source + page
   - Keep highest-scored instance of duplicates
   - Cap at 8 references total
    │
    ▼
5. Format as numbered [Ref N] blocks with source, page, excerpt
    │
    ▼
6. Inject references + prompt into model prompt
   Model writes projectplan.md
    │
    ▼
7. User reviews with /projectplan show
   Executes with /projectplan go
```

### RAG Query Strategy

**Keyword extraction:** Split prompt, remove English stop words ("a", "the", "to", "add", "for", "in", "with", etc.), keep meaningful terms.

**Query templates by project type:**

- **iOS/Swift:** `"HIG {keywords}"`, `"SwiftUI {keywords}"`, `"App Store guidelines {keywords}"`, `"Swift {keywords} best practices"`
- **Python:** `"{keywords} patterns"`, `"{keywords} best practices"`
- **Unknown/other:** `"{keywords}"`

**Adaptive expansion:** After initial 2-3 queries, the model can fire additional `rag_search` tool calls if it needs broader coverage. Floor is 2-3 guaranteed programmatic queries, ceiling is model-driven.

**Deduplication:** Same source + page = keep highest score only.

**Result cap:** Maximum 8 references in the final `## Reference Material` section.

**Search type:** `hybrid` (default) — combines vector semantic + BM25 keyword matching for best recall.

**Collection:** Always `claude_documents`. Personal collections (`jarvis_detections`, `jarvis_browser`, `jarvis_telegram`) are excluded — they're not development references.

### `projectplan.md` Format

```markdown
# Project Plan: {title derived from prompt}

## Reference Material

[Ref 1] **HIG — Settings** (source: apple-hig.pdf, p.42)
> Use a grouped list style for settings. Each group should have
> a descriptive header...

[Ref 2] **SwiftUI — NavigationStack** (source: swiftui-docs, p.15)
> NavigationStack replaces NavigationView in iOS 16+. Use
> navigationDestination(for:) for type-safe routing...

[Ref 3] **App Store Guidelines — 2.3.1** (source: app-store-guidelines)
> Apps should use Settings bundles for system-level preferences.
> In-app settings are acceptable for app-specific configuration...

---

## Summary
Brief description of what will be built and why.

## Steps

1. **Create SettingsView with navigation structure** [see Ref 1, Ref 2]
   - Build NavigationStack-based settings screen
   - Use grouped List style per HIG guidelines

2. **Add SwiftData settings model**
   - Define UserPreferences model with @Model macro

3. **Wire up to existing tab bar** [see Ref 2]
   - Add settings tab to main TabView

## Parallelization
Steps 2 and 3 can run in parallel.
- 2
- 3

## Files
- SettingsView.swift (new)
- UserPreferences.swift (new)
- ContentView.swift (modified)

## Risks
- None significant for this scope
```

### Plan Executor Enhancement

The existing `plan_executor.py` is enhanced to handle ref injection. Detection is automatic — if `## Reference Material` exists in the plan text, it activates ref-aware mode. Otherwise it behaves identically to today.

**Step 1 — Parse references:**

Parse `## Reference Material` section into a dict keyed by ref number:
```python
refs = {
    1: {"title": "HIG — Settings", "text": "Use a grouped list style..."},
    2: {"title": "SwiftUI — NavigationStack", "text": "NavigationStack replaces..."},
}
```

**Step 2 — Parse ref tags from steps:**

Regex `Ref\s+(\d+)` against each step's title + body to find referenced numbers.

**Step 3 — Inject into worker prompts:**

Workers receive a `## Relevant Documentation` block prepended to their task description, containing only the refs their step references. Workers are instructed to follow the documentation when implementing.

**Step 4 — Sequential steps too:**

When the lead agent executes a sequential step, the same ref injection applies to the task description injected via `agent.run()`.

### CLI Implementation

**Location:** New `elif command == "/projectplan":` block in `cli.py`, placed next to the existing `/plan` block (~line 933).

**Subcommands:** `show`, `copy`, `go` are near-identical to `/plan`'s versions, reading `projectplan.md` instead of `plan.md`.

**Helper functions** (in a new `spark_code/projectplan.py` module):

- `extract_keywords(prompt: str) -> list[str]` — strip stop words, return meaningful terms
- `async fetch_rag_context(keywords: list[str], project_type: str) -> str` — fire parallel queries via httpx, dedupe, format as `## Reference Material` block
- `STOP_WORDS: set[str]` — common English stop words to filter

**Dependencies:** None new. Uses `httpx` (already installed) and `project_detect.py` (already exists).

### What This Does NOT Change

- `/plan` command — untouched, works exactly as before
- `rag_search` tool — untouched, model can still call it manually
- System prompt — untouched
- Existing `plan.md` files — not affected
- Tool registry — no new tools added

## Files Affected

| File | Change |
|------|--------|
| `spark_code/projectplan.py` | **New** — keyword extraction, RAG query builder, reference formatter |
| `spark_code/cli.py` | **Modified** — add `/projectplan` command handler, add to help text |
| `spark_code/plan_executor.py` | **Modified** — add reference parsing and injection logic |
| `spark_code/context.py` | **Modified** — add `/projectplan` to the slash command list in `SYSTEM_PROMPT` (line ~20 area, alongside existing tool descriptions). No behavioral changes. |
