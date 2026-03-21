---
title: fast-agent go command
description: Describes how to use the fast-agent go command to quickly connect to and test MCP Servers, including STDIO, SSE and Streaming support.
---

## `fast-agent go` command

The `go` command allows you to run an interactive agent directly from the command line without
creating a dedicated agent.py file.

### Usage

```bash
fast-agent go [OPTIONS]
```

### Options

- `--name TEXT`: Name for the agent (default: "fast-agent")
- `--instruction`, `-i <path or url>`: File name or URL for [System Prompt](../agents/instructions.md)
- `--config-path`, `-c <path>`: Path to config file
- `--servers <server1>,<server2>`: Comma-separated list of server names to enable from config
- `--url TEXT`: Comma-separated list of HTTP/SSE URLs to connect to directly
- `--auth TEXT`: Bearer token for authorization with URL-based servers
- `--model`, `--models <model_string>`: Override the default model (e.g., haiku, sonnet, gpt-4)
- `--model`, `--models <model1>,<model2>,...`: Run one agent per model in parallel and print a side-by-side comparison of responses
- `--pack`, `--card-pack <name>`: Ensure a named card pack is installed in the selected environment before starting
- `--pack-registry <path or url>`: Marketplace file or URL used to resolve `--pack` when it is not already installed
- `--agent-cards`, `--card <path or url>`: Load AgentCards as runnable agents (repeatable)
- `--card-tool <path or url>`: Load AgentCards and attach them as tools to the selected/default agent (repeatable)
- `--agent <name>`: Target a specific loaded agent by name for `--message`, `--prompt-file`, and initial interactive mode
- `--message`, `-m TEXT`: Message to send to the agent (skips interactive mode)
- `--env <path>`: Override the base `.fast-agent` environment directory (where default `agent-cards/` and `tool-cards/` are discovered)
- `--noenv`, `--no-env`: Run in ephemeral mode (disable implicit environment card loading, session persistence/resume, and permission-store side effects)
- `--resume <id|latest>`: Resume the latest session (or a specific session id)
- `--smart`: Prefer a smart default agent when fast-agent creates the default agent
- `--prompt-file`, `-p <path>`: Path to a prompt file to use (either text or JSON)
- `--skills-dir`, `--skills <path>`: Override the default skills directory
- `--stdio "<command> <options>"`: Run the command to attach a STDIO server (enclose arguments in quotes)
- `--npx "@package/name <options>"`: Run an NPX package as a STDIO server (enclose arguments in quotes)
- `--uvx "@package/name <options>"`: Run an UVX package as a STDIO server (enclose arguments in quotes)
- `--shell`, `-x`: Enable a local shell runtime and expose the execute tool (bash or pwsh)

Global CLI options (apply to all subcommands):

- `--quiet`, `-q`: Disable progress display and logging
- `--verbose`, `-v`: Enable verbose mode

### Examples

Note - you may omit `go` when supplying command line options.

```bash
# Basic usage with interactive mode
fast-agent go --model=haiku

# Install or reuse a card pack, then launch immediately
fast-agent go --pack analyst --model haiku

# Resolve a pack from a specific marketplace file
fast-agent go --pack analyst --pack-registry ./marketplace.json --model haiku

# Multi-agent pack: target one specific agent
fast-agent go --pack analyst --agent planner --model haiku

# Basic usage with interactive mode (go omitted)
fast-agent --model haiku

# Use smart default agent (go omitted)
fast-agent --smart --model haiku

# Compare responses across multiple models (comparison mode)
fast-agent --models kimi,gpt-5-mini.low

# Specifying servers from configuration
fast-agent go --servers=fetch,filesystem --model=haiku

# Directly connecting to HTTP/SSE servers via URLs
fast-agent go --url=http://localhost:8001/mcp,http://api.example.com/sse

# Connecting to an authenticated API endpoint
fast-agent go --url=https://api.example.com/mcp --auth=YOUR_API_TOKEN

# Run an NPX package directly
fast-agent --npx @modelcontextprotocol/server-everything 

# Non-interactive mode with a single message
fast-agent go --message="What is the weather today?" --model=haiku

# Target one specific loaded agent when multiple agents are available
fast-agent go --agent-cards ./agents --agent researcher

# Send one message to a specific loaded agent and exit
fast-agent go --agent-cards ./agents --agent qa --message "run smoke checks"

# Using a prompt file
fast-agent go --prompt-file=my-prompt.txt --model=haiku

# Specify a system prompt file
fast-agent go -i my_system_prompt.md

# Specify a skills directory (overrides default search paths)
fast-agent go --skills ~/my-skills/

# Provider LLM shell access (use at your own risk)
fast-agent go -x

```

### Comparison mode (multiple models)

Pass a comma-separated list to `--models` (or `--model`) to run one agent per model in parallel and compare responses side-by-side.

How it works:

- `--instruction` / `-i`, `--servers`, `--url`, and other connection options apply to every model agent.
- Each model string becomes a separate agent name in the output.
- Interactive mode (default): every prompt is sent to all models and results are shown in a comparison view.
- Non-interactive: use `--message` or `--prompt-file` to run once and print results for each model.
- `--smart` is ignored when multiple models are provided.

```bash
fast-agent go --models sonnet,gpt-5-mini.low

# Route to one model agent directly (instead of side-by-side parallel output)
fast-agent go --models sonnet,gpt-5-mini.low --agent sonnet --message "Summarize this"
```

### Agent targeting notes

- Use `--agent` when you loaded multiple agents (for example with `--agent-cards`).
- If `--instruction` points to a local file, fast-agent may derive an internal default agent name
  from the file name (for example `research.md` -> `research`). `--agent` still takes precedence
  for explicit targeting.
- Explicit targeting can include tool-only agents when needed for testing.

### AgentCards vs ToolCards

`tool-cards` are not a separate file format. They are still AgentCards.

- `--agent-cards`: load cards as normal runnable agents.
- `--card-tool`: load cards, then attach those loaded agents as tools to a parent agent.

By default, `fast-agent go` will auto-discover cards from your environment folder when present:

- `<env>/agent-cards/` for runnable agents
- `<env>/tool-cards/` for cards to attach as tools

When `--noenv` is set, this implicit discovery is disabled. Explicit `--agent-cards` and
`--card-tool` values still work.

Cards loaded via `--agent-cards` / `--card-tool` can include `mcp_connect` entries;
those runtime MCP servers are resolved and attached automatically during startup.

See [AgentCards and ToolCards reference](agent_cards.md) for details and recommended layout.

### Card packs

Use `--pack` when you want fast-agent to install or reuse a named card pack in
the selected environment and then launch immediately.

Typical flow:

1. Resolve the environment from `--env` or normal fast-agent settings
2. Reuse the pack if it is already installed there
3. Otherwise resolve it from the configured marketplace (or `--pack-registry`)
4. Install it into the environment
5. Continue through the normal `fast-agent go` startup path

Notes:

- `--pack` cannot be combined with `--noenv`
- `--model` is a fallback only; an explicit model declared in an AgentCard still takes precedence
- Some packs install multiple agents, so you may also want `--agent <name>`

### `--noenv` mode

Use `--noenv` when you want to run without implicit environment side effects.

- Session persistence is disabled.
- Session resume is disabled.
- Default environment card auto-loading is disabled.

Conflicts (fail fast):

- `--noenv` + `--env`
- `--noenv` + `--resume`
- `--noenv` + `--pack`

### URL Connection Details

The `--url` parameter allows you to connect directly to HTTP or SSE servers using URLs.

- URLs must have http or https scheme
- The transport type is determined by the URL path:
  - URLs ending with `/sse` are treated as SSE transport
  - URLs ending with `/mcp` or automatically appended with `/mcp` are treated as HTTP transport
- Server names are generated automatically based on the hostname, port, and path
- The URL-based servers are added to the agent's configuration and enabled

### Authentication

The `--auth` parameter provides authentication for URL-based servers:

- When provided, it creates an `Authorization: Bearer TOKEN` header for all URL-based servers
- This is commonly used with API endpoints that require authentication
- Example: `fast-agent go --url=https://api.example.com/mcp --auth=12345abcde`
