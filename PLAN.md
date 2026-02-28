# Spark Code — Implementation Plan

A local AI coding assistant CLI, powered by Qwen2.5-VL-72B + LoRA on DGX Spark.
Like Claude Code, but fully local, private, and free.

## Name & Branding
- **Name**: Spark Code
- **CLI Command**: `spark`
- **Tagline**: "Your local AI coding assistant"
- **Logo**: Generate with nano-banana MCP (electric spark icon, terminal-style)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    YOUR MACBOOK                       │
│                                                      │
│  Terminal ──▶ spark (CLI) ──▶ Agent Loop              │
│                  │                │                   │
│                  │          ┌─────┴──────┐            │
│                  │          │   Tools    │            │
│                  │          │ read/write │            │
│                  │          │ edit/bash  │            │
│                  │          │ grep/glob  │            │
│                  │          │ web_search │            │
│                  │          │ web_fetch  │            │
│                  │          │ MCP client │            │
│                  │          └────────────┘            │
│                  │                                    │
│                  ▼ (HTTPS / API)                      │
├──────────────────────────────────────────────────────┤
│                    DGX SPARK                          │
│                                                      │
│  Ollama (port 11434) ◀── Qwen2.5-VL-72B + LoRA      │
│  or LLM Server (port 8003)                           │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
spark-code/
├── spark_code/                  # Main package
│   ├── __init__.py              # Version, metadata
│   ├── cli.py                   # Entry point, terminal UI (rich/textual)
│   ├── agent.py                 # Agent loop: prompt → model → tool → repeat
│   ├── model.py                 # Model client (Ollama API / OpenAI-compatible)
│   ├── config.py                # Configuration loading (.spark/config.yaml)
│   ├── context.py               # Context management, conversation history
│   ├── memory.py                # Persistent memory across sessions
│   ├── permissions.py           # Permission system (allow/deny tool calls)
│   ├── streaming.py             # Streaming response handler
│   │
│   ├── tools/                   # Built-in tools
│   │   ├── __init__.py          # Tool registry
│   │   ├── base.py              # Base tool class / interface
│   │   ├── read_file.py         # Read file contents
│   │   ├── write_file.py        # Create new files
│   │   ├── edit_file.py         # Find & replace edits
│   │   ├── bash.py              # Execute shell commands
│   │   ├── glob_search.py       # Find files by pattern
│   │   ├── grep_search.py       # Search file contents (ripgrep)
│   │   ├── list_dir.py          # List directory contents
│   │   ├── web_search.py        # Web search (DuckDuckGo/Brave)
│   │   ├── web_fetch.py         # Fetch & parse web pages
│   │   └── notebook.py          # Jupyter notebook read/edit
│   │
│   ├── mcp/                     # MCP (Model Context Protocol) client
│   │   ├── __init__.py
│   │   ├── client.py            # MCP client (stdio + SSE transport)
│   │   ├── registry.py          # Discover & register MCP server tools
│   │   └── transport.py         # Transport layer (stdio, SSE, HTTP)
│   │
│   ├── skills/                  # Skill system (slash commands)
│   │   ├── __init__.py          # Skill loader & registry
│   │   ├── base.py              # Base skill class
│   │   ├── commit.py            # /commit — git diff → commit message
│   │   ├── review.py            # /review — code review
│   │   ├── test.py              # /test — run tests, report results
│   │   ├── explain.py           # /explain — explain code
│   │   ├── fix.py               # /fix — diagnose and fix errors
│   │   ├── refactor.py          # /refactor — suggest refactoring
│   │   ├── plan.py              # /plan — enter plan mode
│   │   └── search.py            # /search — deep codebase search
│   │
│   └── ui/                      # Terminal UI components
│       ├── __init__.py
│       ├── input.py             # Multi-line input, history, autocomplete
│       ├── output.py            # Markdown rendering, syntax highlighting
│       ├── spinner.py           # Progress spinners & status
│       ├── diff.py              # Diff display for file edits
│       ├── permission_prompt.py # Allow/deny permission dialogs
│       └── theme.py             # Colors, styles, dark/light mode
│
├── skills/                      # User-defined custom skills (YAML/MD)
│   └── example.yaml             # Example custom skill template
│
├── tests/                       # Test suite
│   ├── test_agent.py
│   ├── test_tools.py
│   ├── test_mcp.py
│   ├── test_skills.py
│   └── test_cli.py
│
├── docs/                        # Documentation
│   └── ARCHITECTURE.md
│
├── .spark/                      # Default project config template
│   └── config.yaml              # Default configuration
│
├── pyproject.toml               # Package config (pip install -e .)
├── setup.py                     # Setup script
├── PLAN.md                      # This file
└── LICENSE                      # MIT
```

---

## Features (mirrors Claude Code)

### 1. Core Tools

| Tool | Command | What It Does |
|------|---------|-------------|
| `read_file` | — | Read file contents with line numbers |
| `write_file` | — | Create new files |
| `edit_file` | — | Find & replace in existing files |
| `bash` | — | Execute shell commands |
| `glob` | — | Find files by name pattern |
| `grep` | — | Search file contents (uses ripgrep) |
| `list_dir` | — | List directory contents |
| `web_search` | — | Search the web (DuckDuckGo) |
| `web_fetch` | — | Fetch & parse web pages |
| `notebook` | — | Read/edit Jupyter notebooks |

### 2. Slash Commands (Skills)

| Command | What It Does |
|---------|-------------|
| `/commit` | Read git diff, generate commit message, commit |
| `/review` | Review changed files, find bugs/issues |
| `/test` | Run test suite, report results, fix failures |
| `/explain` | Explain a file or function |
| `/fix` | Diagnose and fix an error |
| `/refactor` | Suggest and apply refactoring |
| `/plan` | Enter plan mode for complex tasks |
| `/search` | Deep codebase search |
| `/help` | Show all commands |
| `/clear` | Clear conversation |
| `/compact` | Summarize conversation to save context |
| `/config` | Show/edit configuration |
| `/model` | Switch model |
| `/cost` | Show token usage stats |

### 3. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message (single line) |
| `Shift+Enter` | New line (multi-line input) |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit Spark Code |
| `Ctrl+L` | Clear screen |
| `Ctrl+R` | Search command history |
| `Tab` | Autocomplete file paths / commands |
| `↑/↓` | Navigate command history |
| `Esc` | Cancel current input |

### 4. Plan Mode

When user types `/plan` or model detects complex task:
1. Model explores codebase (read-only tools)
2. Creates numbered implementation plan
3. Shows plan in formatted box
4. User approves/rejects/modifies
5. Model executes approved plan step by step
6. Each step shows progress indicator

### 5. Permission System

Three modes (like Claude Code):
- **Ask** (default) — prompt before each tool call
- **Auto-allow** — allow all read operations, ask for writes
- **Trust** — allow everything (for experienced users)

```
┌─ Permission Required ──────────────────────────┐
│ spark wants to edit: src/main.py                │
│                                                 │
│ - Line 23: old_code → new_code                  │
│                                                 │
│ [A]llow  [D]eny  [A]lways allow edits           │
└─────────────────────────────────────────────────┘
```

### 6. MCP Server Support

Connect to external MCP servers (like nano-banana):
```yaml
# .spark/config.yaml
mcp_servers:
  nano-banana:
    command: npx nano-banana-mcp
    env:
      GEMINI_API_KEY: ${GEMINI_API_KEY}

  database:
    command: python db_mcp_server.py
    transport: stdio
```

Model can use MCP tools alongside built-in tools.

### 7. Memory System

Persistent memory across sessions:
```
~/.spark/memory/
├── MEMORY.md          # Auto-loaded context
├── projects.md        # Project-specific notes
└── preferences.md     # User preferences
```

Also per-project memory:
```
project/.spark/
├── config.yaml        # Project config
├── memory.md          # Project memory
└── skills/            # Project-specific skills
```

### 8. Context Management

- Conversation history with token counting
- Auto-compact when approaching context limit (32K for Qwen2.5)
- `/compact` manual compaction
- Smart context injection (relevant files, git status)

### 9. Streaming Output

- Real-time token streaming from Ollama API
- Syntax-highlighted code blocks
- Markdown rendering in terminal
- Progress spinners for tool execution
- Diff display for file edits

### 10. Configuration

```yaml
# ~/.spark/config.yaml (global)
model:
  endpoint: https://spark-4a54.local:11434
  name: qwen2.5-vl:72b  # or your LoRA GGUF model
  temperature: 0.7
  max_tokens: 4096
  context_window: 32768

permissions:
  mode: ask  # ask | auto | trust
  always_allow:
    - read_file
    - glob
    - grep
    - list_dir

ui:
  theme: dark
  syntax_highlighting: true
  show_token_count: true
  markdown_rendering: true

mcp_servers:
  nano-banana:
    command: npx nano-banana-mcp
    env:
      GEMINI_API_KEY: ${GEMINI_API_KEY}

memory:
  enabled: true
  path: ~/.spark/memory/
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Basic CLI with rich terminal UI
- [ ] Ollama API client (chat completions + streaming)
- [ ] Conversation history management
- [ ] Configuration system (.spark/config.yaml)

### Phase 2: Core Tools (Week 1-2)
- [ ] read_file, write_file, edit_file
- [ ] bash execution with output capture
- [ ] glob and grep (using ripgrep)
- [ ] list_dir
- [ ] Tool calling format (Qwen2.5 function calling)

### Phase 3: Agent Loop (Week 2)
- [ ] Full agent loop: prompt → model → tool call → execute → feed back
- [ ] Multi-step tool chains (model calls multiple tools in sequence)
- [ ] Error handling and recovery in the loop
- [ ] Permission system (ask/auto/trust)

### Phase 4: UI Polish (Week 2-3)
- [ ] Streaming output with syntax highlighting
- [ ] Diff display for edits
- [ ] Permission prompts (allow/deny)
- [ ] Progress spinners
- [ ] Markdown rendering
- [ ] Keyboard shortcuts
- [ ] Command history

### Phase 5: Skills System (Week 3)
- [ ] Skill loader and registry
- [ ] /commit, /review, /test, /explain
- [ ] /plan (plan mode)
- [ ] /fix, /refactor, /search
- [ ] Custom skill support (YAML/MD templates)

### Phase 6: Web & MCP (Week 3-4)
- [ ] web_search (DuckDuckGo API)
- [ ] web_fetch (requests + beautifulsoup)
- [ ] MCP client (stdio transport)
- [ ] MCP server discovery and tool registration
- [ ] nano-banana integration

### Phase 7: Memory & Context (Week 4)
- [ ] Persistent memory system
- [ ] Per-project .spark/ config
- [ ] Auto-compact conversation
- [ ] Smart context injection
- [ ] Token counting and budget management

### Phase 8: Polish & Package (Week 4+)
- [ ] `pip install spark-code` packaging
- [ ] `spark` command in PATH
- [ ] Logo and branding (nano-banana)
- [ ] Error messages and help text
- [ ] Test suite
- [ ] Documentation

---

## Dependencies

```
# Core
rich>=13.0          # Terminal UI, syntax highlighting, markdown
textual>=0.50       # TUI framework (optional, for advanced UI)
httpx>=0.27         # Async HTTP client (Ollama API)
pyyaml>=6.0         # Config files
click>=8.0          # CLI framework

# Tools
ripgrepy>=2.0       # Python ripgrep wrapper (or subprocess rg)
beautifulsoup4      # Web page parsing
duckduckgo-search   # Web search

# MCP
mcp>=1.0            # MCP Python SDK

# Optional
prompt-toolkit>=3.0 # Advanced input handling
pygments>=2.17      # Syntax highlighting
```

---

## Key Design Decisions

1. **Ollama API (OpenAI-compatible)** — Use `/v1/chat/completions` endpoint so it works with any OpenAI-compatible server, not just Ollama.

2. **Tool calling via Qwen2.5 function calling** — Qwen2.5 supports native tool/function calling. Define tools as JSON schemas, model returns structured tool calls.

3. **Rich library for UI** — Rich handles syntax highlighting, markdown, tables, spinners, prompts. Lighter than textual for a CLI.

4. **Ripgrep for search** — Shell out to `rg` for fast grep. Fall back to Python regex if rg not installed.

5. **Skills as prompt templates** — Skills are just pre-written system prompts + tool sequences. Easy to add new ones.

6. **MCP for extensibility** — MCP lets users add any external tool without modifying Spark Code. nano-banana, databases, APIs, etc.

---

## Usage Examples

```bash
# Start spark in current directory
spark

# Start with specific model
spark --model qwen2.5-vl:72b

# Start with specific endpoint
spark --endpoint https://spark-4a54.local:11434

# Quick one-shot question
spark -m "How do I fix the auth middleware?"

# Run a skill directly
spark commit
spark review
spark test
```

### Interactive Session
```
$ spark

  ⚡ Spark Code v1.0.0
  Model: qwen2.5-vl:72b @ spark-4a54.local
  Type /help for commands

> Fix the bug in src/auth.py where tokens expire too early

  I'll investigate the token expiration issue.

  📖 Reading src/auth.py...
  🔍 Searching for expiration logic...

  I found the issue on line 47. The token lifetime is set to
  60 seconds instead of 3600:

  ┌─ Edit: src/auth.py ──────────────────────────┐
  │ - TOKEN_LIFETIME = 60    # seconds            │
  │ + TOKEN_LIFETIME = 3600  # 1 hour             │
  └───────────────────────────────────────────────┘

  Allow this edit? [Y]es / [N]o: y

  ✅ Fixed. Token lifetime changed from 60s to 1 hour.

> /test

  🧪 Running tests...
  $ pytest tests/ -v
  ✅ 23 passed, 0 failed

> /commit

  📝 Generating commit message...

  fix: increase token lifetime from 60s to 3600s (1 hour)

  Commit? [Y]es / [N]o / [E]dit: y
  ✅ Committed: a1b2c3d
```

---

## How It Connects to Training

The coding LoRA training (running now) teaches the model:
- **Tool calling patterns** — 200 agentic examples teach when/how to call tools
- **Planning** — model learns to plan before executing
- **Error recovery** — model learns to retry and adapt
- **Your stack** — model knows Jarvis, Docker, Spark, your directories
- **All languages** — balanced training across Python, JS, TS, Swift, Rust, etc.

After training completes:
1. Convert LoRA to GGUF → load in Ollama
2. Spark Code connects to Ollama
3. Model uses tools naturally because it learned the patterns in training
