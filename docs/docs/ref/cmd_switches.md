---
title: Command Line Options
description: Command line option reference for fast-agent MCP applications
social:
  title: Command Line Options
  tagline: Reference every fast-agent command-line switch and runtime option.
  description: Reference every fast-agent command-line switch and runtime option.
  alt: fast-agent social card — Command Line Options
---


# Command Line Options

**fast-agent** offers flexible command line options for both running agent applications and using built-in CLI utilities.

## Agent Applications

When running a **fast-agent** application (typically `uv run agent.py`), you have access to the following command line options:

| Option | Description | Example |
|--------|-------------|---------|
| `--model MODEL` | Override the default model for the agent | `--model gpt-4o` |
| `--agent AGENT` | Specify which agent to use (default: "default") | `--agent researcher` |
| `-m, --message MESSAGE` | Send a single message to the agent and exit | `--message "Hello world"` |
| `-p, --prompt-file <path-or-uri>` | Load and apply a prompt file from a path, HTTP(S) URL, `file://` URI, or `hf://` URI | `--prompt-file conversation.txt` |
| `--quiet` | Disable progress display, tool and message logging | `--quiet` |
| `--version` | Show version and exit | `--version` |
| `--transport {http,stdio,acp,a2a}` | Transport protocol; enabling it also turns on server mode | `--transport http` |
| `--port PORT` | Port for HTTP server (default: 8000) | `--port 8080` |
| `--host HOST` | Host for HTTP/A2A server (default: 127.0.0.1; use 0.0.0.0 only to accept remote clients) | `--host 127.0.0.1` |
| `--instance-scope {shared,connection,request}` | Control server-side agent instancing (default: shared) | `--instance-scope connection` |
| `--workspace DIR` | Override the workspace root; default home becomes `DIR/.fast-agent` | `--workspace ./client-project` |
| `--home DIR` | Override the base fast-agent home | `--home ./agent-home` |
| `--skills DIR` | Override the default skills directory | `--skills ./skills` |
| `--dump DIR` | Export loaded agents as Markdown AgentCards | `--dump ./agent-cards` |
| `--dump-yaml DIR` | Export loaded agents as YAML AgentCards | `--dump-yaml ./agent-cards` |

`--transport` implies server mode when running a Python module directly. Without `--transport`, the module runs in interactive/message mode.

### Examples

```bash
# Run interactively with specified model
uv run agent.py --model sonnet

# Run specific agent
uv run agent.py --agent researcher

# Run with specific agent and model
uv run agent.py --agent researcher --model gpt-4o

# Send a message to an agent and exit
uv run agent.py --agent summarizer --message "Summarize this document"

# Apply a prompt file
uv run agent.py --prompt-file my_conversation.txt

# Apply a prompt file from Hugging Face Hub generic storage
uv run agent.py --prompt-file hf://buckets/evalstate/home/demo.md

# Run as an HTTP server on port 8080
uv run agent.py --transport http --port 8080

# Run as a stdio server
uv run agent.py --transport stdio

# Get minimal output (for scripting)
uv run agent.py --quiet --message "Generate a report"
```

### Programmatic Control of Command Line Parsing

When embedding FastAgent in other applications (like web frameworks or GUI applications), you can disable command line parsing by setting `parse_cli_args=False` in the constructor:

```python
# Create FastAgent without parsing command line arguments
fast = FastAgent("Embedded Agent", parse_cli_args=False)
```

This is particularly useful when:
- Integrating with frameworks like FastAPI/Uvicorn that have their own argument parsing
- Building GUI applications where command line arguments aren't relevant
- Creating applications with custom argument parsing requirements


## fast-agent go Command

The `fast-agent go` command lets you run an interactive agent directly without creating a Python file. Read the guide [here](go_command/)

For ephemeral runs with no implicit home side effects, use `--no-home`.

For machine-readable one-shot JSON output, use:

```bash
fast-agent go --no-home --message "..." --json-schema ./schema.json
```

This validates the response locally and writes only JSON to stdout.

For one-shot `fast-agent go` runs, attach local files or HTTP(S) URLs with
repeatable `--attach` / `-a` options:

```bash
fast-agent go --message "Summarize this" --attach ./report.pdf
fast-agent go --prompt-file review.md --attach ./evidence.pdf --attach https://example.com/chart.png
```

`--attach` requires `--message` or `--prompt-file`; with `--prompt-file`,
attachments are added to the last user message in the prompt.

For card-based loading and the distinction between `--agent-cards` and `--card-tool`, see [AgentCards and ToolCards](../agents/defining/agent_cards/).

## fast-agent export Command

Use `fast-agent export` to list persisted sessions or export one as a Codex-style
JSONL trace.

```bash
# List recent sessions
fast-agent export --list

# Export locally
fast-agent export latest --output trace.jsonl

# Upload to a Hugging Face URL
fast-agent export latest --hf-url hf://buckets/your-name/fast-agent-traces/
```

See the full reference [here](export_command/).

## fast-agent check Command

Use `fast-agent check` to diagnose your configuration:

```bash
# Show configuration summary
fast-agent check

# Display configuration file
fast-agent check show

# Display secrets file
fast-agent check show --secrets
```

## fast-agent model Command

Use `fast-agent model` for interactive model onboarding flows:

```bash
# Configure or update namespaced model references
fast-agent model setup

# Inspect reference resolution and onboarding readiness
fast-agent model doctor

# Interactively discover llama.cpp models and import one as a local overlay
fast-agent model llamacpp --url http://localhost:8080

# List discovered llama.cpp models
fast-agent model llamacpp list --url http://localhost:8080
```

For overlay manifests and llama.cpp import details, see [Model Overlays](../models/model_overlays/).

## fast-agent scaffold Command

Create a new agent project with configuration files:

```bash
# Set up in current directory
fast-agent scaffold

# Set up in a specific directory
fast-agent scaffold --config-dir ./my-agent

# Force overwrite existing files
fast-agent scaffold --force
```

## fast-agent quickstart Command

Create example applications to get started quickly:

```bash
# Show available examples
fast-agent quickstart

# Create workflow examples
fast-agent quickstart workflow .

# Create researcher example
fast-agent quickstart researcher .

# Create data analysis example
fast-agent quickstart data-analysis .

# Create state transfer example
fast-agent quickstart state-transfer .
```
