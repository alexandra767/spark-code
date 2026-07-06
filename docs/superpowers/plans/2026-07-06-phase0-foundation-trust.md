# Phase 0: Foundation Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Spark Code's current feature set works against the real local engine (vLLM 80B + Ollama 30B), fix the config drift and anything the live runs expose, and push `main` to GitHub.

**Architecture:** Three new artifacts — a preflight module that catches config/server context-window drift at startup, a standalone live smoke suite that drives the installed CLI end-to-end against the real engine, and a fresh-eyes audit round over the July 2 fix commits (which were never live-tested). Everything else is fixes with regression tests.

**Tech Stack:** Python 3.11+, httpx, click, pytest (+pytest-asyncio, follow existing async test patterns in `tests/test_agent.py`), ruff.

## Global Constraints

- All edits in `~/spark-code` (the editable install Spark runs from). `~/CodingProjects/spark-code` is a symlink — never treat them as two repos.
- The vLLM served-model alias `qwen3.5:122b` on `http://spark-4a54.local:30000` is a **hard contract** shared with JARVIS and Claude UI. Never "fix" the name, never touch server config (`~/spark-vllm-docker` is out of scope).
- Secondary profile: `coder` = Ollama `qwen3-coder:30b` on `http://spark-4a54.local:11434`.
- Server context window is 32,768 tokens (measured live 2026-07-06; KV cache 108,256 tokens).
- Test commands: `cd ~/spark-code && .venv/bin/python -m pytest -q` (572 tests green at start) and `.venv/bin/ruff check .` (clean at start). Both must be green before every commit.
- Live smoke runs happen while JARVIS/Claude UI share the engine — that's the realistic condition, don't schedule around them.
- Commits: small, per-fix, message style `fix:`/`feat:`/`test:` matching existing history.

---

### Task 1: Context-window preflight check + config repair

**Files:**
- Create: `spark_code/preflight.py`
- Create: `tests/test_preflight.py`
- Modify: `spark_code/cli.py` (startup banner section — locate with `grep -n "ping" spark_code/cli.py`, wire in after the existing successful-ping branch)
- Modify (user config, not repo): `~/.spark/config.yaml` — `providers.llm.context_window: 16384` → `32768`

**Interfaces:**
- Produces: `async fetch_server_max_context(endpoint: str, model: str, api_key: str = "", timeout: float = 5.0, transport=None) -> int | None` and `context_window_warning(configured: int, server_max: int | None) -> str | None` in `spark_code.preflight`. Task 2's smoke suite imports neither; Task 3 auditors may reference them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preflight.py
import httpx
import pytest

from spark_code.preflight import context_window_warning, fetch_server_max_context


def _transport(payload, status=200):
    def handler(request):
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_fetch_returns_max_model_len_for_matching_model():
    payload = {"data": [{"id": "qwen3.5:122b", "max_model_len": 32768}]}
    got = await fetch_server_max_context(
        "http://x:30000", "qwen3.5:122b", transport=_transport(payload))
    assert got == 32768


@pytest.mark.asyncio
async def test_fetch_returns_none_when_field_missing():
    # Ollama's OpenAI-compat /v1/models has no max_model_len
    payload = {"data": [{"id": "qwen3-coder:30b"}]}
    got = await fetch_server_max_context(
        "http://x:11434", "qwen3-coder:30b", transport=_transport(payload))
    assert got is None


@pytest.mark.asyncio
async def test_fetch_returns_none_when_model_absent():
    payload = {"data": [{"id": "other-model", "max_model_len": 4096}]}
    got = await fetch_server_max_context(
        "http://x:30000", "qwen3.5:122b", transport=_transport(payload))
    assert got is None


@pytest.mark.asyncio
async def test_fetch_returns_none_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("boom")
    got = await fetch_server_max_context(
        "http://x:30000", "m", transport=httpx.MockTransport(handler))
    assert got is None


@pytest.mark.asyncio
async def test_fetch_returns_none_on_http_error():
    got = await fetch_server_max_context(
        "http://x:30000", "m", transport=_transport({}, status=500))
    assert got is None


def test_warning_when_configured_below_server():
    msg = context_window_warning(16384, 32768)
    assert msg is not None
    assert "16384" in msg and "32768" in msg


def test_warning_when_configured_above_server():
    msg = context_window_warning(65536, 32768)
    assert msg is not None
    assert "exceed" in msg.lower()


def test_no_warning_on_match():
    assert context_window_warning(32768, 32768) is None


def test_no_warning_when_server_unknown():
    assert context_window_warning(32768, None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'spark_code.preflight'`

- [ ] **Step 3: Implement `spark_code/preflight.py`**

```python
"""Startup preflight checks against the live inference server."""

from __future__ import annotations

import httpx


async def fetch_server_max_context(
    endpoint: str,
    model: str,
    api_key: str = "",
    timeout: float = 5.0,
    transport=None,
) -> int | None:
    """Return the server-reported max context length for ``model``, or None.

    vLLM reports ``max_model_len`` per model on /v1/models; Ollama's
    OpenAI-compat endpoint omits the field. Any error means "unknown" —
    preflight must never block startup.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"{endpoint.rstrip('/')}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            for item in resp.json().get("data", []):
                if item.get("id") == model:
                    value = item.get("max_model_len")
                    return value if isinstance(value, int) else None
    except Exception:
        return None
    return None


def context_window_warning(configured: int, server_max: int | None) -> str | None:
    """Human-readable drift warning, or None when config matches reality."""
    if server_max is None or configured == server_max:
        return None
    if configured > server_max:
        return (
            f"context_window {configured} exceeds the server's max "
            f"{server_max} — long sessions will 400; lower context_window "
            f"in ~/.spark/config.yaml"
        )
    return (
        f"server supports {server_max} tokens but context_window is "
        f"{configured} — raise it in ~/.spark/config.yaml to use the full window"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q`
Expected: 9 passed

- [ ] **Step 5: Wire into startup**

In `spark_code/cli.py`, inside the interactive-startup path where the existing connection check reports success (find it: `grep -n "ping" spark_code/cli.py`), add — following the surrounding code's style for console output and config access (`get(config, "model", ...)`):

```python
from spark_code.preflight import context_window_warning, fetch_server_max_context

server_max = await fetch_server_max_context(
    get(config, "model", "endpoint"),
    get(config, "model", "name"),
    api_key=get(config, "model", "api_key", default=""),
)
warning = context_window_warning(
    get(config, "model", "context_window", default=32768), server_max)
if warning:
    console.print(f"[yellow]⚠ {warning}[/yellow]")
```

Note: verify the exact config key for the provider's context window by checking `load_config` in `spark_code/config.py` (the provider block is mapped into `config["model"]`). If the mapped key differs (e.g. `config["model"]["context_window"]` vs top-level), use the real one and mirror it in the code above.

- [ ] **Step 6: Fix the user config drift**

```bash
python3 - <<'EOF'
import re, pathlib
p = pathlib.Path.home() / ".spark/config.yaml"
text = p.read_text()
# Only the llm provider block's 16384 — verify visually after
new = text.replace("context_window: 16384", "context_window: 32768", 1)
p.write_text(new)
print("done" if new != text else "NO CHANGE — inspect manually")
EOF
grep -n -A2 "llm:" ~/.spark/config.yaml | head -8
```

Expected: `context_window: 32768` under the `llm:` provider. Leave `worker_model`/`max_workers` untouched (Phase 2 revisits them).

- [ ] **Step 7: Verify live, full suite, commit**

Run: `cd ~/spark-code && .venv/bin/spark --help >/dev/null && echo OK` (import sanity), then `.venv/bin/python -m pytest -q && .venv/bin/ruff check .`
Then start `spark` interactively once (engine up) and confirm no context-window warning appears; temporarily set the config to 16384 and confirm the warning DOES appear; restore 32768.

```bash
git add spark_code/preflight.py tests/test_preflight.py spark_code/cli.py
git commit -m "feat: preflight context-window drift check against live server"
```

---

### Task 2: Live smoke suite

**Files:**
- Create: `scripts/smoke_live.py`
- Test: the script IS the test (it needs a live engine, so it is not part of pytest). A dry `--help` invocation is the only CI-safe check.

**Interfaces:**
- Consumes: the installed CLI (`python -m spark_code.cli`), `spark_code.config.load_config`, `spark_code.model.ModelClient`, `spark_code.context.Context` (`.compact(keep_recent)`, `.save(path, label, cwd)`, `.load(path)`), `spark_code.agent.Agent(model, context, tools, permissions, console)` with `await agent.run(prompt)`, `spark_code.tools.build_tools` (verify import path: `grep -rn "def build_tools" spark_code/`), `spark_code.permissions.PermissionManager(mode="trust")`.
- Produces: `scripts/smoke_live.py --provider {llm|coder}` exit 0 = all green; used by Task 4 as the acceptance gate.

- [ ] **Step 1: Verify CLI flag spelling before writing the script**

Run: `.venv/bin/spark --help`
Record: the provider flag (expected `-p/--provider`), the trust flag (expected `--trust`), and that a positional prompt triggers one-shot mode. If any differ, adjust the constants at the top of the script accordingly.

- [ ] **Step 2: Write the script**

```python
#!/usr/bin/env python3
"""Live smoke suite: drives the installed Spark CLI against the real engine.

Usage:
    .venv/bin/python scripts/smoke_live.py --provider llm
    .venv/bin/python scripts/smoke_live.py --provider coder

Runs while JARVIS/Claude UI share the engine — that is the realistic
condition. Exit 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONE_SHOT_TIMEOUT = 300  # seconds; 80B is slow and the engine is shared

HELLO = 'GREETING = "smoke-7f3a9"\n\nprint(GREETING)\n'


def one_shot(provider: str, prompt: str, cwd: str) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "spark_code.cli", "--trust",
           "-p", provider, prompt]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=ONE_SHOT_TIMEOUT)
    return proc.returncode, proc.stdout + proc.stderr


class Smoke:
    def __init__(self, provider: str):
        self.provider = provider
        self.failures: list[str] = []
        sys.path.insert(0, REPO)
        from spark_code.config import load_config
        cfg = load_config(os.getcwd(), provider=provider)
        self.endpoint = cfg["model"]["endpoint"]
        self.model_name = cfg["model"]["name"]
        self.tmp = tempfile.mkdtemp(prefix="spark-smoke-")

    def run_check(self, name, fn):
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {name}  ({time.time() - t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001 — report, don't crash the suite
            self.failures.append(f"{name}: {exc}")
            print(f"  FAIL  {name}: {exc}")

    # -- checks ------------------------------------------------------------

    def c1_models_endpoint(self):
        with urllib.request.urlopen(f"{self.endpoint}/v1/models", timeout=10) as r:
            ids = [m["id"] for m in json.load(r)["data"]]
        assert self.model_name in ids, f"{self.model_name} not in {ids}"

    def c2_read(self):
        with open(os.path.join(self.tmp, "hello.py"), "w") as f:
            f.write(HELLO)
        rc, out = one_shot(self.provider,
                           "Read hello.py and reply with the exact value of "
                           "the GREETING constant.", self.tmp)
        assert rc == 0, f"exit {rc}: {out[-500:]}"
        assert "smoke-7f3a9" in out, f"marker missing: {out[-500:]}"

    def c3_edit(self):
        rc, out = one_shot(self.provider,
                           "In hello.py, change the GREETING value from "
                           "'smoke-7f3a9' to 'spark-ok-2026' by editing the "
                           "file.", self.tmp)
        content = open(os.path.join(self.tmp, "hello.py")).read()
        assert "spark-ok-2026" in content, \
            f"file unchanged (exit {rc}): {out[-500:]}"

    def c4_bash(self):
        rc, out = one_shot(self.provider,
                           'Use the bash tool to run: python3 -c "print(6*7*101)" '
                           "and report the number it prints.", self.tmp)
        assert "4242" in out, f"bash marker missing (exit {rc}): {out[-500:]}"

    def c5_multiround(self):
        rc, out = one_shot(self.provider,
                           "Create fizzbuzz.py with a fizzbuzz(n) function "
                           "(returns 'Fizz'/'Buzz'/'FizzBuzz'/str(n)). Create "
                           "test_fizzbuzz.py with pytest tests for n=3,5,15,7. "
                           "Then run pytest with the bash tool and report the "
                           "result.", self.tmp)
        assert os.path.exists(os.path.join(self.tmp, "fizzbuzz.py")), \
            f"fizzbuzz.py not created (exit {rc}): {out[-500:]}"
        assert os.path.exists(os.path.join(self.tmp, "test_fizzbuzz.py")), \
            "test file not created"
        assert "passed" in out.lower(), f"tests not run/passed: {out[-500:]}"

    def c6_compact_survival(self):
        asyncio.run(self._compact_survival())

    async def _compact_survival(self):
        from rich.console import Console
        from spark_code.agent import Agent
        from spark_code.config import load_config
        from spark_code.context import Context
        from spark_code.model import ModelClient
        from spark_code.permissions import PermissionManager
        from spark_code.tools import build_tools  # adjust if defined elsewhere

        cfg = load_config(self.tmp, provider=self.provider)
        model = ModelClient(
            endpoint=cfg["model"]["endpoint"],
            model=cfg["model"]["name"],
            temperature=0.3,
            max_tokens=cfg["model"].get("max_tokens", 4096),
            api_key=cfg["model"].get("api_key", ""),
            provider=cfg["model"].get("provider", "ollama"),
            timeout=float(cfg["model"].get("timeout", 300)),
        )
        context = Context()
        agent = Agent(model, context, build_tools(),
                      PermissionManager(mode="trust"),
                      Console(quiet=True))
        try:
            os.chdir(self.tmp)
            r1 = await agent.run("Read hello.py and summarize it in one line.")
            assert r1, "empty response before compact"
            # compact right after a tool exchange — the historical wedge case
            context.compact(keep_recent=2)
            r2 = await agent.run("Reply with the single word: ok")
            assert r2, "empty response after compact (session wedged?)"
        finally:
            await model.close()
            os.chdir(REPO)

    def c7_session_roundtrip(self):
        from spark_code.context import Context
        ctx = Context()
        ctx.add_user("hello")
        ctx.add_assistant("world")
        path = os.path.join(self.tmp, "session.json")
        ctx.save(path, label="smoke", cwd=self.tmp)
        fresh = Context()
        assert fresh.load(path), "load() returned False"
        assert len(fresh.get_messages()) == len(ctx.get_messages()), \
            "message count changed across save/load"

    # -- driver ------------------------------------------------------------

    def run(self) -> int:
        print(f"Smoke: provider={self.provider} model={self.model_name} "
              f"endpoint={self.endpoint}")
        self.run_check("1 models endpoint", self.c1_models_endpoint)
        self.run_check("2 one-shot read", self.c2_read)
        self.run_check("3 one-shot edit", self.c3_edit)
        self.run_check("4 one-shot bash", self.c4_bash)
        self.run_check("5 multi-round build+test", self.c5_multiround)
        self.run_check("6 compact survival (in-process)", self.c6_compact_survival)
        self.run_check("7 session save/load", self.c7_session_roundtrip)
        print(f"\n{'ALL GREEN' if not self.failures else 'FAILURES:'}")
        for f in self.failures:
            print(f"  - {f}")
        return 1 if self.failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="llm", choices=["llm", "coder"])
    args = ap.parse_args()
    sys.exit(Smoke(args.provider).run())


if __name__ == "__main__":
    main()
```

Adjust the `build_tools` import to its real location (Step 1 of Task 2 in practice: `grep -rn "def build_tools" spark_code/`), and the `ModelClient` kwargs to the real constructor (`grep -n "def __init__" spark_code/model.py`) — the shapes above came from `cli.py:_one_shot` and must match exactly.

- [ ] **Step 3: Dry-run the script's scaffolding (no engine dependency)**

Run: `.venv/bin/python scripts/smoke_live.py --help`
Expected: usage text, exit 0.

- [ ] **Step 4: First live run against the 80B**

Run: `.venv/bin/python scripts/smoke_live.py --provider llm`
Expected on first run: any mix of PASS/FAIL — failures here are FINDINGS, not script bugs (triage in Task 3). If a failure is a script defect (wrong flag, wrong import), fix the script and re-run until failures are all genuine product findings.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_live.py
git commit -m "feat: live smoke suite against real vLLM/Ollama engines"
```

---

### Task 3: Fresh-eyes audit of main + fix round

The July 2 fix round (~55 bugs, commits `19443d7` + `1f61f34`) rewrote streaming, MCP, compaction, sessions, and permissions but never ran against a live engine. This task is a discovery loop: audit, verify, fix with regression tests.

**Files:**
- Create: `tests/test_audit_2026_07_06.py` (all regression tests for this round)
- Modify: whatever the verified findings implicate

**Interfaces:**
- Consumes: smoke-suite failures from Task 2 Step 4 (each is automatically a P0 finding).
- Produces: green smoke suite precondition for Task 4.

- [ ] **Step 1: Dispatch three reviewer agents over the current code (max 3 concurrent — matches `agents.max_concurrent` reality on the shared engine)**

Lens A — engine/streaming: "Review `spark_code/model.py` and `spark_code/agent.py` at commit `1f61f34` for defects that only manifest against a live OpenAI-compatible streaming server (vLLM with the qwen3_coder tool parser, Ollama): SSE chunk handling, usage accounting, retry paths, `<think>` filtering keyed to served-model aliases, cancellation actually closing the stream. Report file:line, defect, concrete failure scenario."

Lens B — sessions/context: "Review `spark_code/context.py` (especially `compact()` and `_safe_split_index()`), session save/load in `spark_code/cli.py` (`_sessions_dir`, resume paths) at `1f61f34` for sequences that produce OpenAI-invalid message lists (tool result without its call), data loss on save/load, or resume regressions from the July 2 `~/.spark/history` FILE→DIR migration. Report file:line, defect, failure scenario."

Lens C — fix-round regressions: "Diff `dd55c8c..1f61f34` in `~/spark-code`. The round claimed ~55 fixes. Look for fixes that introduced new bugs, half-applied changes (fixed in one code path, not its parallel twin — e.g. single vs parallel tool execution), and claims in commit messages not matched by the code. Report file:line, defect, failure scenario."

- [ ] **Step 2: Verify every finding before accepting it**

For each reported finding, confirm it in the actual code (read the cited lines yourself or dispatch a skeptic agent per finding: "Try to refute this claim against the code as written"). Discard anything that doesn't survive. Merge with Task 2's live-run failures. Triage: P0 = breaks a mainline flow (one-shot, edit, compact, resume) or corrupts data; P1 = wrong behavior with workaround; P2 = hygiene (note in `docs/superpowers/plans/phase0-deferred.md`, don't fix now).

- [ ] **Step 3: Fix each P0/P1 with the TDD cycle**

For each fix, in order: write the failing regression test in `tests/test_audit_2026_07_06.py` (name it `test_<symptom>`, docstring cites the finding); run it, see it fail; make the minimal fix; run it, see it pass; run `.venv/bin/python -m pytest -q && .venv/bin/ruff check .`; commit as `fix: <symptom> (audit 2026-07-06)`. One commit per fix.

- [ ] **Step 4: Re-run the smoke suite until green**

Run: `.venv/bin/python scripts/smoke_live.py --provider llm`
Expected: ALL GREEN, exit 0. Loop back to Step 3 for anything still red.

---

### Task 4: Acceptance run + push

**Files:** none new — this is the gate.

**Interfaces:**
- Consumes: `scripts/smoke_live.py` (Task 2), green fix round (Task 3).

- [ ] **Step 1: Smoke vs the 80B (llm)**

Run: `.venv/bin/python scripts/smoke_live.py --provider llm`
Expected: ALL GREEN.

- [ ] **Step 2: Smoke vs the 30B (coder)**

First check the model is loadable: `curl -s http://spark-4a54.local:11434/v1/models | python3 -m json.tool | grep qwen3-coder` — if Ollama doesn't list it (the healthcheck may have evicted pins), SKIP with a note in the run log rather than force-loading 25GB next to vLLM (memory-budget policy on the Spark evicts it anyway).
Run: `.venv/bin/python scripts/smoke_live.py --provider coder`
Expected: ALL GREEN (or documented SKIP).

- [ ] **Step 3: Full local gate**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check . && git status --short`
Expected: all tests pass (572 + new), ruff clean, working tree clean.

- [ ] **Step 4: Push (authorized by the approved Phase 0 spec)**

```bash
cd ~/spark-code && git push origin main
```
Expected: `main -> main` on `github.com/alexandra767/spark-code`. Verify: `git log origin/main --oneline -1` matches local HEAD.

---

## Amendment (2026-07-06, post-Task-2 review)

Task 2's reviewer found two false-PASS risks in the plan's own smoke-script
code (plan-mandated defects). Amended requirements for `scripts/smoke_live.py`:

- **c4_bash:** the script generates a random 8-digit integer R at runtime and
  writes `mystery.py` containing `print({R} ^ 0xA5A5)`; the prompt tells the
  model to run it with bash and report the number; the assertion requires
  `str(R ^ 0xA5A5)` in output. (Old `6*7*101=4242` was mentally computable —
  the check could pass without bash ever running.)
- **c5_multiround:** in addition to the file-existence asserts and the model's
  claimed result matching `r"\d+ passed"`, the script itself runs
  `sys.executable -m pytest -q` in the temp project and asserts exit 0 — the
  authoritative signal that the generated code passes its own tests. (Old
  `"passed" in out` matched "1 failed, 3 passed" too.)
- **rc asserts:** c3/c4/c5 assert the one-shot exit code is 0.
- **c7:** compares full message content across save/load, not just count.
