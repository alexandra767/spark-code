"""RAG search tool — queries the RAG service on the Spark for indexed documents."""

from ..codesearch import resolve_service_url
from .base import Tool


def _score(r: dict) -> float:
    """Numeric sort/floor key for a result.

    Scores are reranker-scale NEGATIVES (e.g. -1.78 .. -10.24). A missing or
    non-numeric score must sort BELOW every real hit and fail any floor
    comparison, so it maps to ``-inf`` — never ``0``, which would rank a
    scoreless item ABOVE genuine negative-scored hits and corrupt both the
    merge ordering and the vault-floor eviction.
    """
    s = r.get("score")
    return s if isinstance(s, (int, float)) else float("-inf")


# When the caller doesn't name a collection, fan out across the general
# document library, the synced history of Claude Code sessions
# (claude_code_memory), and the synced Obsidian vault (second_brain), then
# merge and re-rank. Mirrors the Claude UI rag_search handler's default
# fan-out (claude-ui/backend/tools/handlers_rag.py, Task 8).
DEFAULT_COLLECTIONS = ["claude_documents", "claude_code_memory", "second_brain"]


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "Search your indexed knowledge base (documents, Swift docs, Google Docs, "
        "crawled websites, camera detections, browser history, Telegram messages, "
        "your second-brain vault). Returns relevant passages with source citations. "
        "Use this when the user asks about content from their personal documents, "
        "notes, or indexed reference material."
    )
    is_read_only = True
    requires_permission = False

    def __init__(self, config: dict | None = None):
        # Resolve the RAG service URL exactly like code_search does
        # (codesearch.resolve_service_url): a config `rag.service_url` wins,
        # else the RAG_SERVICE_URL env var, else a sensible default. The tool
        # is registered as `RagSearchTool()` (no args) in cli.py, so config
        # defaults to None (→ env → default) and that no-arg call still works.
        # Peer code_search takes config and honors rag.service_url; without it
        # this tool always hit the localhost default — live-broken on the Mac,
        # where localhost:8010 is down but the Spark host serves. Pass
        # `config=config` at the cli.py registration site to honor the user's
        # configured host.
        self._service_url = resolve_service_url(config)

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — what to look for",
                },
                "collection": {
                    "type": "string",
                    "description": (
                        "Collection to search. Omit to search everything that matters "
                        "by default (claude_documents, claude_code_memory, second_brain). "
                        "Or name one directly: claude_documents, claude_code_memory, "
                        "second_brain, jarvis_detections, jarvis_browser, jarvis_telegram"
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results (default: 5)",
                },
                "search_type": {
                    "type": "string",
                    "description": "hybrid (default), semantic, or keyword",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, collection: str | None = None,
                      n_results: int = 5, search_type: str = "hybrid", **kw) -> str:
        try:
            import httpx
        except ImportError:
            return "Error: httpx not installed. Run: pip install httpx"

        # Treat "" and None IDENTICALLY as "no explicit collection". LLM
        # tool-callers routinely pass collection="" for an optional string
        # param; deriving `collections` with `if collection` (fan-out) but
        # `explicit` with `is not None` (True for "") would fan out across the
        # three defaults yet suppress the second_brain deepening AND the
        # CANONICAL/HISTORICAL labels below, silently losing the
        # vault-is-source-of-truth guarantee while still paying for the
        # fan-out. Both derivations use the same truthiness.
        collections = [collection] if collection else DEFAULT_COLLECTIONS
        explicit = bool(collection)

        base_payload = {
            "query": query,
            "n_results": n_results,
            "search_type": search_type,
            "user_role": "owner",
        }

        # Vault-canonical floor config (default fan-out only). The underlying
        # hybrid+rerank service's candidate pool scales with n_results, so a
        # small outer n_results (e.g. 5) can fail to surface a second_brain
        # chunk that would clearly rank #1 at a wider pool — verified live
        # 2026-07-10: apps/boonpoint.md is invisible in second_brain's own
        # top 12 for "which apps are live" but rank 1 at n_results>=13.
        # Search second_brain deeper than the caller asked so the floor
        # guarantee below has real candidates to draw from.
        VAULT_POOL = 15
        # Absolute relevance gate on the backfill below — only inject a
        # second_brain candidate if it scores at or above this floor.
        # Deliberately NOT a relative (vault >= victim) rule: that would
        # re-break Q6, whose correct injection scored -1.78, below several
        # of the items it evicts — the whole point of the K=2 floor is
        # canonical presence despite a moderate score, not "only if it
        # wins on points." Calibration, live-verified 2026-07-10: Q6's
        # desired injection scored -1.78; confirmed-bad off-domain
        # injections (a Kubernetes readinessProbe query, a Honda Civic
        # torque-spec query) scored -10.24 / -10.28. -5.0 sits with clear
        # margin on both sides.
        VAULT_FLOOR_MIN_SCORE = -5.0

        merged = []
        per_collection = {}
        failed = []
        elapsed = 0.0
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for coll in collections:
                    n_for_coll = n_results
                    if coll == "second_brain" and not explicit:
                        n_for_coll = max(n_results, VAULT_POOL)
                    # Guard each collection independently: one collection
                    # returning non-JSON (a 502 HTML page → JSONDecodeError) or
                    # raising mid-loop (ConnectError) must NOT abort the whole
                    # search or discard results already gathered from the
                    # others. A failed collection is skipped and noted.
                    try:
                        resp = await client.post(
                            f"{self._service_url}/search",
                            json=dict(base_payload, collection=coll, n_results=n_for_coll),
                        )
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception:
                        failed.append(coll)
                        continue
                    elapsed += data.get("elapsed_ms", 0) or 0
                    results = data.get("results", [])
                    for r in results:
                        r["_collection"] = coll
                    per_collection[coll] = results  # service returns these score-sorted
                    merged.extend(results)
        except Exception as e:
            return f"RAG search error: {e}"

        if not merged:
            if failed:
                # Every queried collection failed — surface a reachability
                # error, not a misleading "no results" (which reads as "the
                # service is fine, your query just matched nothing").
                return (f"Error: RAG service not reachable at {self._service_url} "
                        f"(all {len(failed)} collection queries failed). "
                        f"Is it running on the Spark?")
            return f"No results found for: {query}"

        cap = max(n_results, 5)
        if len(collections) > 1:
            merged.sort(key=_score, reverse=True)
            merged = merged[:cap]

            # Vault-canonical floor: default fan-out only, never for an
            # explicit single-collection call (len(collections) > 1 already
            # implies no explicit collection was passed — see DEFAULT_COLLECTIONS
            # above). second_brain is the current source of truth for
            # status-shaped questions ("which apps are live"), but a broad
            # query can let claude_code_memory/claude_documents out-score it
            # on pure relevance and crowd it out of the merged cap entirely
            # (see docs/verification/retrieval-battery.md Q6 in the
            # second-brain repo). Guarantee the top VAULT_FLOOR second_brain
            # hits always survive the cap, evicting the lowest-scored
            # non-second_brain items to make room. This is NOT the earlier
            # "one slot per every collection" egalitarian quota (rejected
            # 2026-07-09 as a proven no-op for the Q2 gap) — it is targeted
            # specifically at second_brain, not every collection.
            if "second_brain" in per_collection:
                VAULT_FLOOR = 2
                vault_sorted = per_collection["second_brain"]
                already_ids = {id(r) for r in merged}
                vault_count = sum(1 for r in merged if r.get("_collection") == "second_brain")
                if vault_count < VAULT_FLOOR:
                    missing = VAULT_FLOOR - vault_count
                    backfill = [
                        r for r in vault_sorted
                        if id(r) not in already_ids
                        and _score(r) >= VAULT_FLOOR_MIN_SCORE
                    ][:missing]
                    if backfill:
                        room = cap - len(merged)
                        if room < len(backfill):
                            to_evict = len(backfill) - room
                            evicted = 0
                            i = len(merged) - 1
                            while evicted < to_evict and i >= 0:
                                if merged[i].get("_collection") != "second_brain":
                                    del merged[i]
                                    evicted += 1
                                i -= 1
                        merged.extend(backfill)
                        merged = merged[:cap]

        output = [f"**{len(merged)} results** ({elapsed:.0f}ms)\n"]

        # Label each item's source status on the default fan-out path so the
        # model doesn't treat a dated memory note as equally authoritative as
        # the vault's current page. Explicit single-collection calls are left
        # unlabeled/unmodified.
        SOURCE_LABELS = {
            "second_brain": "[CANONICAL — current]",
            "claude_code_memory": "[HISTORICAL note — may be stale; prefer second-brain on conflicts]",
        }

        for i, r in enumerate(merged, 1):
            source = r.get("source", "unknown")
            text = r.get("text", "")
            citation = r.get("citation", {})
            page = citation.get("page")
            page_str = f", p.{page}" if page else ""
            raw_score = r.get("score")
            score_str = f"{raw_score:.4f}" if isinstance(raw_score, (int, float)) else "n/a"
            label = ""
            if not explicit:
                marker = SOURCE_LABELS.get(r.get("_collection"))
                if marker:
                    label = f" {marker}"

            output.append(f"**[{i}]** `{source}{page_str}` (score: {score_str}){label}")
            output.append(f"{text}\n")

        return "\n".join(output)
