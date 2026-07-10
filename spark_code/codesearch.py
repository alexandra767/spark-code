"""RAG-backed per-project code search (Phase 4 Task 3).

Investigation findings (Step 1 — read before touching anything here):

The RAG service at ``rag.service_url`` (default
:data:`spark_code.projectplan.DEFAULT_RAG_SERVICE_URL`) is a small FastAPI
app (see its ``/openapi.json``) that exposes BOTH a query surface and several
INDEX/write surfaces:

  * ``POST /search`` — the query endpoint projectplan.py's ``_run_query``
    already uses: ``{query, collection, n_results, search_type, user_role}``
    → ``{results: [...], ...}``. Each result is ``{text, source, score,
    citation, metadata}`` — ``metadata`` echoes back whatever
    ``extra_metadata`` was supplied at ingest time (verified live).
  * ``POST /ingest`` / ``POST /ingest_directory`` — take a **server-side
    filesystem path** (``filepath`` / ``directory``). These assume the RAG
    service can read the file directly off its own disk, which only works
    when Spark Code happens to run co-located with the service (e.g. on the
    DGX itself). They are NOT used here — Spark Code must work when run from
    a laptop against a remote RAG host, so relying on a shared filesystem
    would silently break the common case.
  * ``POST /ingest_text`` — "Ingest raw text content (no file needed)":
    ``{text, title, collection, access_level, chunk_size, extra_metadata,
    skip_bm25_rebuild}``. This is the one used here: it transmits file
    content over HTTP, so it works regardless of where Spark Code runs
    relative to the RAG host. Live-verified that ``extra_metadata`` set at
    ingest time (e.g. ``{"path": ..., "start_line": ...}``) round-trips into
    each search hit's ``metadata`` dict — that's how ``path:line`` results
    are reconstructed below without needing the service to understand code
    at all.
  * ``POST /rebuild_bm25`` — manually triggers the BM25 side-index rebuild
    for a collection. ``ingest_text`` is called with
    ``skip_bm25_rebuild=True`` per-chunk (rebuilding after every one of
    potentially hundreds of chunks would be very slow) and rebuilt once at
    the end of :func:`index_project`.
  * ``DELETE /collections/{name}`` exists too (not used by this module, but
    means a bad/stale project index can be dropped manually via curl if
    ever needed).

Because ``/ingest_text`` chunks server-side using its own ``chunk_size``
(default 1500 chars) with NO way to recover per-server-chunk line offsets,
this module does its OWN line-aware chunking (:func:`_chunk_text`) and passes
a ``chunk_size`` request parameter larger than any chunk it ever sends
(``_INGEST_CHUNK_SIZE_PARAM``) so the server never re-splits a chunk further
— that keeps exactly one client-side chunk == exactly one server-side chunk,
which is what makes the ``extra_metadata["start_line"]`` accurate for every
hit returned by ``/search``.

Scope adjustment from the plan: the index half IS feasible (contra the
brief's fallback clause for a query-only service) — ``/ingest_text`` is
exactly the "content, not filepath" index endpoint the task needed. No
scope reduction was necessary.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import os

from .config import get as get_config
from .projectplan import get_rag_service_url
from .tools.base import Tool, _is_binary

# Directories always pruned while walking a project tree to index — dead
# weight (deps, VCS metadata, build output) that would otherwise dominate the
# index with noise no one will ever semantically search for.
SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".next", ".idea", ".vscode", "target", ".cache", "site-packages",
    ".build", "DerivedData", "Pods",
})

# Files larger than this are skipped outright rather than partially indexed
# (brief: "cap file size ~100KB").
MAX_INDEX_FILE_BYTES = 100_000

# Target size (chars) for each client-side chunk. Kept comfortably under the
# RAG service's own default chunk_size (1500) so a chunk this size would
# never have been re-split anyway even without _INGEST_CHUNK_SIZE_PARAM.
CHUNK_CHAR_TARGET = 1200

# Passed as the /ingest_text request's chunk_size so the server-side chunker
# never re-splits a chunk we already sized ourselves — see module docstring.
_INGEST_CHUNK_SIZE_PARAM = 1_000_000

INDEX_TIMEOUT = 15.0
SEARCH_TIMEOUT = 8.0
# How many concurrent /ingest_text requests index_project fires at once.
_INDEX_CONCURRENCY = 8


def collection_name_for(cwd: str) -> str:
    """Deterministic, collision-safe per-project collection name.

    Hashes the *resolved* (symlink-following) absolute path, so two ways of
    referring to the same project directory (a symlink vs. its target, a
    relative vs. absolute cwd) land in the same collection, while distinct
    projects — even same-named ones in different parents — never collide.
    """
    real = os.path.realpath(cwd)
    digest = hashlib.sha1(real.encode("utf-8")).hexdigest()[:12]
    return f"spark_code_{digest}"


def resolve_service_url(config: dict | None) -> str:
    """``rag.service_url`` from config if set, else the env-aware projectplan default."""
    if config:
        url = get_config(config, "rag", "service_url", default=None)
        if url:
            return url
    return get_rag_service_url()


def _load_gitignore_patterns(root: str) -> list[str]:
    """Best-effort ``.gitignore`` support.

    Whole-path-component glob matching via :mod:`fnmatch` — NOT full
    git-wildmatch semantics (no negation beyond dropping ``!`` lines
    entirely, no nested ``.gitignore`` files, no ``**``). This project has no
    ``pathspec`` dependency and Task 3's global constraint is "no new hard
    dependencies", so this trades precision for zero new deps; good enough to
    keep common ignored directories (build output, coverage, etc.) that
    aren't already in :data:`SKIP_DIRS` out of the index.
    """
    patterns: list[str] = []
    try:
        with open(os.path.join(root, ".gitignore"), encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue
                patterns.append(line.rstrip("/"))
    except OSError:
        pass
    return patterns


def _gitignore_skip(rel_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    parts = rel_path.split(os.sep)
    return any(fnmatch.fnmatch(part, pat) for part in parts for pat in patterns)


def _iter_source_files(root: str):
    """Yield ``(abs_path, rel_path)`` for every indexing candidate under root.

    Prunes :data:`SKIP_DIRS`, dot-directories, and ``.gitignore`` matches
    while walking (so it never descends into e.g. ``node_modules`` instead of
    filtering its contents out after the fact).
    """
    patterns = _load_gitignore_patterns(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)

        def _rel(name: str) -> str:
            return name if rel_dir == "." else os.path.join(rel_dir, name)

        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
            and not _gitignore_skip(_rel(d), patterns)
        ]
        for fname in filenames:
            # Dotfiles (.gitignore, .env, .DS_Store, ...) are config/noise,
            # not code — skip them the same way dot-directories are pruned
            # above.
            if fname.startswith("."):
                continue
            rel_path = _rel(fname)
            if _gitignore_skip(rel_path, patterns):
                continue
            yield os.path.join(dirpath, fname), rel_path


def _chunk_text(text: str, target_chars: int = CHUNK_CHAR_TARGET) -> list[tuple[int, str]]:
    """Split ``text`` into ``(start_line, chunk_text)`` pairs at line boundaries.

    Greedily accumulates whole lines until adding the next one would exceed
    ``target_chars``, then flushes. A single line longer than
    ``target_chars`` becomes its own (oversized) chunk rather than being
    split mid-line — code lines that long are rare and mid-line splits would
    produce useless snippets.
    """
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 1
    buf_len = 0
    for i, line in enumerate(lines, start=1):
        if buf and buf_len + len(line) + 1 > target_chars:
            chunks.append((buf_start, "\n".join(buf)))
            buf = []
            buf_start = i
            buf_len = 0
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        chunks.append((buf_start, "\n".join(buf)))
    return chunks


async def index_project(cwd: str, service_url: str | None = None,
                        collection: str | None = None, *,
                        max_file_bytes: int = MAX_INDEX_FILE_BYTES,
                        timeout: float = INDEX_TIMEOUT,
                        concurrency: int = _INDEX_CONCURRENCY,
                        transport=None) -> dict:
    """Walk ``cwd`` and index it into a per-project RAG collection.

    Returns ``{"indexed": int, "skipped": int, "collection": str}`` on
    success (``indexed``/``skipped`` count FILES, not chunks) or
    ``{"error": str}`` if the RAG service is unreachable — never raises.

    A cheap ``GET /health`` probe runs first specifically so an unreachable
    service fails fast (one request) instead of walking a potentially large
    tree first. Once the service is confirmed reachable, per-chunk POST
    failures during the actual walk degrade to "skipped" rather than a
    top-level error — a service that flakes mid-run still leaves you with a
    partial-but-useful index instead of nothing.
    """
    import httpx

    root = os.path.realpath(cwd)
    service_url = (service_url or resolve_service_url(None)).rstrip("/")
    collection = collection or collection_name_for(root)

    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            # Health-first so an unreachable service fails fast (one request)
            # instead of walking (and per-file _is_binary-probing) a
            # potentially large tree first — the docstring's fail-fast promise.
            try:
                health = await client.get(f"{service_url}/health")
                if health.status_code != 200:
                    return {"error": f"RAG service at {service_url} returned HTTP {health.status_code}"}
            except Exception as e:
                return {"error": f"RAG service unreachable at {service_url}: {e}"}

            # Service confirmed reachable — now walk the tree.
            candidates: list[tuple[str, str]] = []
            skipped = 0
            for abs_path, rel_path in _iter_source_files(root):
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    skipped += 1
                    continue
                if size == 0 or size > max_file_bytes:
                    skipped += 1
                    continue
                if _is_binary(abs_path):
                    skipped += 1
                    continue
                candidates.append((abs_path, rel_path))

            if not candidates:
                return {"indexed": 0, "skipped": skipped, "collection": collection}

            sem = asyncio.Semaphore(concurrency)
            indexed = 0

            async def _ingest_file(abs_path: str, rel_path: str) -> bool:
                try:
                    with open(abs_path, encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except OSError:
                    return False
                any_ok = False
                for start_line, chunk in _chunk_text(text):
                    if not chunk.strip():
                        continue
                    payload = {
                        "text": chunk,
                        "title": f"{rel_path}:{start_line}",
                        "collection": collection,
                        "chunk_size": _INGEST_CHUNK_SIZE_PARAM,
                        "extra_metadata": {"path": rel_path, "start_line": start_line},
                        "skip_bm25_rebuild": True,
                    }
                    async with sem:
                        try:
                            resp = await client.post(f"{service_url}/ingest_text", json=payload)
                        except Exception:
                            continue
                    if resp.status_code == 200:
                        any_ok = True
                return any_ok

            outcomes = await asyncio.gather(
                *(_ingest_file(a, r) for a, r in candidates),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if outcome is True:
                    indexed += 1
                else:
                    skipped += 1

            try:
                await client.post(f"{service_url}/rebuild_bm25", json={"collection": collection})
            except Exception:
                pass  # best-effort — search still works (semantic side) without it
    except Exception as e:
        return {"error": str(e)}

    return {"indexed": indexed, "skipped": skipped, "collection": collection}


async def search_project(service_url: str, collection: str, query: str,
                         k: int = 5, *, timeout: float = SEARCH_TIMEOUT,
                         transport=None) -> tuple[list[dict], str | None]:
    """Query a project's code collection. Returns ``(results, error)`` — never raises."""
    import httpx

    try:
        n_results = max(1, min(int(k or 5), 20))
    except (TypeError, ValueError):
        n_results = 5

    payload = {
        "query": query,
        "collection": collection,
        "n_results": n_results,
        "search_type": "hybrid",
        "user_role": "owner",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            resp = await client.post(f"{service_url.rstrip('/')}/search", json=payload)
            if resp.status_code != 200:
                return [], f"HTTP {resp.status_code}"
            data = resp.json()
            return data.get("results", []) or [], None
    except Exception as e:
        return [], str(e)


def unavailable_message(service_url: str) -> str:
    return (f"code search unavailable (index this project with /index; "
            f"RAG service at {service_url})")


def _format_snippet(text: str, max_lines: int = 4, max_chars: int = 320) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    head = lines[0].strip()
    extra = lines[1:max_lines]
    body = "\n".join(f"    {ln}" for ln in extra)
    snippet = head if not body else f"{head}\n{body}"
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3] + "..."
    return snippet


def format_code_search_results(results: list[dict]) -> str:
    """Format raw ``/search`` result dicts as numbered ``path:line — snippet`` blocks."""
    if not results:
        return ""
    blocks = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata") or {}
        path = meta.get("path") or r.get("source") or "unknown"
        line = meta.get("start_line") or 1
        snippet = _format_snippet(r.get("text", ""))
        blocks.append(f"{i}. {path}:{line} — {snippet}")
    return "\n\n".join(blocks)


class CodeSearchTool(Tool):
    """Semantic 'where/how is X handled' search over this project's RAG index.

    Registers unconditionally (unless ``rag.code_search_enabled`` is
    explicitly ``false``) — it does no network I/O at construction time, so
    an away-from-home session (RAG host unreachable) never fails to start
    because of this tool. Every failure mode of :meth:`execute` (service
    down, project never indexed, empty results) collapses to the same
    friendly :func:`unavailable_message` string — never an exception.
    """

    name = "code_search"
    is_read_only = True
    requires_permission = False

    def __init__(self, config: dict | None = None, cwd: str | None = None,
                enabled: bool | None = None, transport=None):
        self._cwd = os.path.realpath(cwd or os.getcwd())
        self._service_url = resolve_service_url(config)
        self._collection = collection_name_for(self._cwd)
        self._enabled = (
            get_config(config, "rag", "code_search_enabled", default=True)
            if enabled is None else enabled
        )
        # Test-only: an httpx transport override threaded through to
        # search_project (mirrors preflight.py's transport= pattern).
        self._transport = transport

    @property
    def description(self) -> str:
        return (
            "Semantic search over this project's RAG-indexed code — finds "
            "'where/how is X handled' by meaning, not just literal text "
            "(complements grep, which is exact-text). Run /index first to "
            "build this project's collection. Returns a clear unavailable "
            "message (never an error) if the project hasn't been indexed or "
            "the RAG service is unreachable."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, e.g. 'where is the retry backoff computed'",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, k: int = 5, **kwargs) -> str:
        if not self._enabled:
            return unavailable_message(self._service_url)
        results, error = await search_project(
            self._service_url, self._collection, query, k,
            transport=self._transport)
        if error:
            # A real service/transport failure (HTTP 5xx, auth, connection
            # refused, timeout) is NOT the same as "this project was never
            # indexed" — collapsing every failure to the re-index nudge
            # misdiagnoses the user (told to run /index when the service is
            # down). Surface the real error, still with the /index hint in
            # case it genuinely is unindexed. Never raises.
            return (f"code search unavailable: {error} "
                    f"(RAG service at {self._service_url}). "
                    f"If this project was never indexed, run /index.")
        if not results:
            # No error, just no hits — most likely never indexed (or nothing
            # matched). Keep the clean re-index nudge.
            return unavailable_message(self._service_url)
        return format_code_search_results(results)
