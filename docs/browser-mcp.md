# Browser automation via MCP (Playwright MCP)

Spark Code can drive a real web browser — navigate, click, type, read the
accessibility tree, screenshot — by connecting to the
[Playwright MCP](https://github.com/microsoft/playwright-mcp) server over the
built-in **stdio** MCP transport (`spark_code/mcp/transport.py`). This is the
"Claude-in-Chrome"-style capability: the model gets `browser_*` tools that
return text-first accessibility snapshots the 80B can reason over directly.

This page has the exact config, the prerequisites, and a smoke procedure. It was
written after the transport's **first real-server trial** — see
[Shakeout result](#shakeout-result) at the bottom.

## Prerequisites

- **Node.js + npx** on `PATH`. Verified working on Node `v25.2.1` / npx
  `11.12.0`. Check with:
  ```sh
  which npx node && node --version && npx --version
  ```
- **A browser for Playwright to drive.** Two options:
  - *Use system Google Chrome* (no extra download) — pass `--browser chrome`.
    This is what the config below does; it just needs `/Applications/Google
    Chrome.app` (macOS) or Chrome on `PATH`.
  - *Let Playwright manage Chromium* — drop `--browser chrome` and run
    `npx playwright install chromium` once (~120 MB download).
- **First run downloads the MCP package.** `npx -y @playwright/mcp@latest`
  fetches the server on first use (a few seconds on a warm npm cache, longer
  cold). Spark's `initialize` handshake uses a generous timeout, but if your
  network is slow, pre-warm it once outside Spark:
  ```sh
  npx -y @playwright/mcp@latest --help
  ```

## Config

Spark reads runtime MCP servers from the **`mcp_servers`** block of your config
(`~/.spark/config.yaml`, or a project-local `.spark/config.yaml`). Add:

```yaml
mcp_servers:
  playwright:
    transport: stdio          # optional; stdio is the default
    command: npx
    args:
      - "-y"
      - "@playwright/mcp@latest"
      - "--browser"
      - "chrome"              # drive system Chrome; omit to use bundled Chromium
      - "--headless"          # no visible window; drop this to watch it work
      - "--isolated"          # fresh profile each run, nothing persisted to disk
      - "--output-dir"
      - "/tmp/spark-playwright-mcp"   # IMPORTANT: keep artifacts out of your repo
    # env:                    # optional; supports ${VAR} expansion
    #   HTTP_PROXY: "${HTTP_PROXY}"
```

> **Why `--output-dir` matters.** Without it, Playwright MCP writes snapshot
> `.yml` / screenshot files into a `.playwright-mcp/` folder **in the current
> working directory** — i.e. your repo. Pointing `--output-dir` at a temp path
> (and using `--isolated`) keeps your working tree clean. This was a real
> finding from the shakeout.

Keeping `mcp_servers` at its default (`null` / `{}`) leaves browser automation
off. Nothing is spawned until you add the block above.

### Tool names

Tools are exposed to the model as `<server>__<tool>`, so with the server named
`playwright` above you get `playwright__browser_navigate`,
`playwright__browser_snapshot`, `playwright__browser_click`, etc. The server
advertises 23 tools; the useful text-first core:

| Tool | What it does |
|------|--------------|
| `browser_navigate` | Go to a URL |
| `browser_snapshot` | Accessibility-tree snapshot of the page (text) |
| `browser_click` / `browser_type` / `browser_fill_form` | Interact |
| `browser_press_key` / `browser_hover` / `browser_select_option` | Interact |
| `browser_navigate_back` / `browser_wait_for` / `browser_tabs` | Navigation |
| `browser_console_messages` / `browser_network_requests` | Diagnostics |
| `browser_take_screenshot` | PNG — routed to a vision-capable model (see note below) |
| `browser_evaluate` / `browser_run_code_unsafe` | Run JS in the page |

Image content from MCP tools (e.g. `browser_take_screenshot`) is routed
through the same vision path `simulator_screenshot` uses (Phase 5 Task 5):
with a vision-capable model configured (`model.vision` / the active
provider's `vision` key in config — same setting the iOS Simulator
screenshot tool checks), the PNG reaches the model as an image turn. Without
a vision-capable model, you get back the original `[Image: <mime>]` text
placeholder — for page *content* prefer **`browser_snapshot`** regardless
(the accessibility tree is text the primary text model reasons over
directly, and is far cheaper).

## Result budgets (important)

Browser tools return **huge** dumps: a single `browser_snapshot` of a Wikipedia
article measured **~300 KB** in testing — 20× the default 15 K tool-result cap,
enough to swamp the context window on its own.

Spark caps any MCP tool whose bare name starts with `browser_` at
**8 000 chars** (`BROWSER_MCP_RESULT_BUDGET` in `spark_code/agent.py`), applied
by `Agent._budget_for` regardless of what you name the server. Head+tail
truncation is preserved (you see the top of the tree and the tail).

To override for a specific tool, set an exact `server__tool` key — it wins over
the prefix rule:

```yaml
tools:
  result_budgets:
    playwright__browser_snapshot: 20000   # give snapshots more room
```

## Smoke procedure

1. Add the `mcp_servers.playwright` block above to `~/.spark/config.yaml`.
2. Start Spark. On startup you should see
   `Connecting to MCP servers...` and no MCP warning. If a server fails to
   connect, Spark prints `Warning: MCP server 'playwright' failed: …` and keeps
   running without the tools (startup is never blocked by an MCP server).
3. Ask Spark to drive a page, e.g.:
   > *Navigate to https://example.com and tell me the page heading.*
   Spark calls `playwright__browser_navigate`, reads the snapshot, and should
   report **"Example Domain"**.
4. Confirm nothing leaked into your repo: there should be **no**
   `.playwright-mcp/` directory in your working tree (artifacts go to
   `--output-dir`).

### Transport-level smoke (no Spark UI)

To verify just the handshake against the real server, drive the transport
directly:

```python
import asyncio
from spark_code.mcp.transport import StdioTransport

async def main():
    t = StdioTransport("npx", [
        "-y", "@playwright/mcp@latest",
        "--browser", "chrome", "--headless", "--isolated",
        "--output-dir", "/tmp/spark-playwright-mcp",
    ])
    await t.start()
    init = await t.send("initialize", {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "spark-code", "version": "0.1.0"}}, timeout=60)
    print("serverInfo:", init.get("serverInfo"))
    await t.notify("notifications/initialized")
    tools = (await t.send("tools/list", timeout=60)).get("tools", [])
    print(f"{len(tools)} tools")
    r = await t.send("tools/call", {
        "name": "browser_navigate",
        "arguments": {"url": "https://example.com"}}, timeout=90)
    text = "\n".join(c["text"] for c in r.get("content", []) if c.get("type") == "text")
    print("Example Domain present:", "Example Domain" in text)
    await t.stop()

asyncio.run(main())
```

## Shakeout result

This was the first time Spark's rewritten stdio MCP transport (July-2 audit:
reader-task + `id → Future` map, stderr drain, `notifications/initialized`,
10 MB stream limit) ran against a **real** MCP server rather than
`tests/_mcp_fake_server.py`. It passed clean, first try:

- `initialize` completed in **~0.9 s** → `serverInfo` Playwright
  `1.62.0-alpha`, protocol `2024-11-05`.
- `notifications/initialized` accepted; `tools/list` returned **23 tools**.
- **Concurrent id-matching:** three parallel `tools/list` calls each got their
  own correct response — no cross-talk.
- **Live flow:** `browser_navigate` → `https://example.com` returned a snapshot
  asserting **"Example Domain"**; the full `MCPClient` → `MCPTool.execute` path
  (what the CLI drives) worked end-to-end and `disconnect_all()` was clean.
- None of the July-2 bug classes (stderr deadlock, id/response mismatch,
  timeout stream-poisoning, missing `notifications/initialized`, 64 KB readline
  limit) reproduced. No transport changes were needed.

The one operational gotcha — `.playwright-mcp/` written into the cwd — is a
Playwright MCP default, not a transport bug, and is handled by the
`--output-dir` flag in the config above.
