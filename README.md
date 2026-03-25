# Spark Code

Your local AI coding assistant — like Claude Code, powered by your own model.

## Features

- **Agent loop** with streaming, tool calling, and multi-round execution
- **12+ built-in tools**: file ops, bash, grep/glob, web search, worker agents
- **Multi-provider**: Ollama, Gemini, OpenAI, Groq, DeepSeek, OpenRouter
- **Team system**: spawn background workers, inter-agent messaging
- **Plan mode**: create plans, execute with parallel workers
- **Smart UX**: slash commands, autocomplete, Nord theme, inline diffs
- **Persistence**: session history, memory, pinned files, snippets
- **MCP support**: connect external tool servers
- **Skills**: built-in /commit, /review, /test, /fix, /refactor, /explain

## Install

```bash
pip install -e .
```

## Usage

```bash
# Interactive mode
spark

# One-shot
spark "explain this project"

# With options
spark --provider gemini --trust
spark --yolo    # autonomous agent mode
spark --resume  # continue last session

# First-time setup
spark --setup
```

## Configuration

Config files: `~/.spark/config.yaml` (global) and `.spark/config.yaml` (project).

```yaml
providers:
  ollama:
    endpoint: http://localhost:11434
    model: qwen2.5:72b
    context_window: 32768
  gemini:
    endpoint: https://generativelanguage.googleapis.com/v1beta/openai
    model: gemini-2.0-flash
    api_key: ${GEMINI_API_KEY}

active_provider: ollama
```

## Commands

Type `/help` in the CLI for the full list, including:

`/plan`, `/team`, `/commit`, `/review`, `/test`, `/publish`, `/new`, `/run`,
`/watch`, `/checkpoint`, `/rollback`, `/cost`, `/profile`, `/apply`, `/teach`,
`/share`, `/search`, `/branch`, `/switch`, and more.

## License

MIT — Alexandra Titus
