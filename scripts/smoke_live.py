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
        from spark_code.cli import build_tools  # build_tools lives in cli.py, not tools/
        from spark_code.config import load_config
        from spark_code.context import Context
        from spark_code.model import ModelClient
        from spark_code.permissions import PermissionManager

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
