# Fast-Agent CLI Commands

This directory contains the command implementations for the fast-agent CLI.

## Go Command

The `go` command allows you to run an interactive agent directly from the command line without
creating a dedicated agent.py file.

### Usage

```bash
fast-agent go [OPTIONS]
```

### Options

- `--name TEXT`: Name for the agent (default: "FastAgent CLI")
- `--instruction`, `-i TEXT`: Instruction for the agent (default: "You are a helpful AI Agent.")
- `--config-path`, `-c TEXT`: Path to config file
- `--servers TEXT`: Comma-separated list of server names to enable from config
- `--url TEXT`: Comma-separated list of HTTP/SSE URLs to connect to directly
- `--auth TEXT`: Bearer token for authorization with URL-based servers
- `--model TEXT`: Override the default model (e.g., haiku, sonnet, gpt-4)
- `--message`, `-m TEXT`: Message to send to the agent (skips interactive mode)
- `--prompt-file`, `-p TEXT`: Path to a prompt file to use (either text or JSON)
- `--quiet`: Disable progress display and logging

### Examples

```bash
# Basic usage with interactive mode
fast-agent go --model=haiku

# Specifying servers from configuration
fast-agent go --servers=fetch,filesystem --model=haiku

# Directly connecting to HTTP/SSE servers via URLs
fast-agent go --url=http://localhost:8001/mcp,http://api.example.com/sse

# Connecting to an authenticated API endpoint
fast-agent go --url=https://api.example.com/mcp --auth=YOUR_API_TOKEN

# Non-interactive mode with a single message
fast-agent go --message="What is the weather today?" --model=haiku

# Using a prompt file
fast-agent go --prompt-file=my-prompt.txt --model=haiku
```

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

## Serve Command

The `serve` command starts FastAgent as an MCP server so it can be consumed by MCP clients.

### Usage

```bash
fast-agent serve [OPTIONS]
```

### Options

- `--name TEXT`: Name for the MCP server (default: "fast-agent")
- `--instruction`, `-i TEXT`: Instruction for the agent (defaults to the standard FastAgent instruction)
- `--config-path`, `-c TEXT`: Path to config file
- `--servers TEXT`: Comma-separated list of server names to enable from config
- `--card`, `--agent-cards TEXT`: Path or URL to an AgentCard file or directory (repeatable)
- `--url TEXT`: Comma-separated list of HTTP/SSE URLs to connect to
- `--auth TEXT`: Bearer token for authorization with URL-based servers
- `--model TEXT`: Override the default model (e.g., haiku, sonnet, gpt-4)
- `--skills-dir`, `--skills PATH`: Override the default skills directory
- `--npx TEXT`: NPX package and args to run as an MCP server (quoted)
- `--uvx TEXT`: UVX package and args to run as an MCP server (quoted)
- `--stdio TEXT`: Command to run as STDIO MCP server (quoted)
- `--transport [http|sse|stdio|acp]`: Transport protocol to expose (default: http)
- `--host TEXT`: Host address when using HTTP or SSE transport (default: 0.0.0.0)
- `--port INTEGER`: Port when using HTTP or SSE transport (default: 8000)
- `--shell`, `-x`: Enable a local shell runtime and expose the execute tool
- `--description`, `-d TEXT`: Description used for each send tool (supports `{agent}` placeholder)
- `--tool-name-template TEXT`: Template for exposed agent tool names (supports `{agent}` placeholder)
- `--instance-scope [shared|connection|request]`: Control how MCP clients receive isolated agent instances (default: shared)
- `--reload`: Enable manual AgentCard reloads (ACP: `/reload`, MCP: `reload_agent_cards`)
- `--watch`: Watch AgentCard paths and reload

### Skills behavior

When configuring agents in code, `skills=None` explicitly disables skills for that agent. If `skills` is omitted, the default skills registry is used.

### Examples

```bash
# HTTP transport on default port
fast-agent serve --model=haiku --transport=http

# SSE transport on a custom port
fast-agent serve --transport=sse --port=8723

# Expose an MCP stdio server alongside the agent
fast-agent serve --stdio "python my_server.py --debug"

# Combine URL-based servers and an NPX helper
fast-agent serve --url=https://api.example.com/mcp --npx "@modelcontextprotocol/server-filesystem /data"

# Custom tool description (the {agent} placeholder is replaced with the agent name)
fast-agent serve --description "Interact with the {agent} workflow via MCP"

# Load AgentCards from a file or directory
fast-agent serve --card ./agents --transport=http

# Watch AgentCard directory for changes
fast-agent serve --card ./agents --watch --transport=http

# Use per-connection instances to isolate history between clients
fast-agent serve --instance-scope=connection --transport=http
```

### Environment toggles

- uvloop is enabled by default when installed (non-Windows); set `FAST_AGENT_DISABLE_UV_LOOP=1` to opt out.
