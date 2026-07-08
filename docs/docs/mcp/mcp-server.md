---
social:
  title: Run an MCP Server
  tagline: Expose fast-agent capabilities through an MCP server.
  description: Expose fast-agent capabilities through an MCP server.
  alt: fast-agent social card — Run an MCP Server
---

### Running as an MCP Server

**`fast-agent`** Can deploy any configured agents over MCP, letting external MCP clients connect via STDIO or Streamable HTTP.

Additionally, there is a convenient `serve` command enabling rapid, command line deployment of MCP enabled agents in a variety of instancing modes.

This feature also works with [Agent Skills](../guides/skills/), enabling powerful adaptable behaviours.

This page describes **managed MCP server mode**: fast-agent owns the FastMCP
server process. Use [Custom MCP Servers](harness-adapter.md) when you want to
own the FastMCP server and call fast-agent through the harness adapter.

#### Ways to build an MCP server

Choose the smallest surface that fits what you are building:

| Goal | Use |
| --- | --- |
| Expose an existing agent quickly | `fast-agent serve` |
| Package an `agent.py` as a server | `uv run agent.py --transport http` |
| Design your own FastMCP tools | [Custom MCP Servers](harness-adapter.md) |
| Deploy with Hugging Face OAuth | [Host on Hugging Face Spaces](huggingface-spaces.md) |
| Return interactive UI | [FastMCP Apps](fastmcp-apps.md) |

The default server and custom FastMCP integrations use the same harness-backed
application boundary:

```text
Agent definitions
├─ Python decorators: @fast.agent() in agent.py
├─ AgentCards/config: fast-agent serve --agent-cards ./agents
└─ Harness app: harness_app.entrypoint for custom orchestration

FastAgent runtime
└─ HarnessSessions
   └─ HarnessApp
      └─ MCP adapter
         ├─ AgentCard/agent-named tools
         ├─ custom FastMCP tools
         └─ FastMCP Apps
```

Use the Python form when you want a packaged application and the `serve` form
when you want a declarative/card-based server. A configured
`harness_app.entrypoint` wraps the harness application used by managed
AgentCard tools, so authors can intercept sessions/invocations while keeping
the standard MCP tool surface. Use a custom FastMCP server only when you want to
own the MCP-facing interface or expose non-AgentCard tools.

For hosted MCP, prefer request-scoped serving. Each tool call opens a transient
harness session and durable state lives in storage you control. For stateful MCP
clients, connection-scoped serving can use the current `Mcp-Session-Id` as the
default harness session key. Server authors can also expose their own handle
arguments on custom MCP tools and pass any chosen key to the Harness API;
fast-agent does not require a universal state-handle convention.

#### Using the CLI (fast-agent serve)

```bash
fast-agent serve [OPTIONS]
```

Key options:

- `--transport [http|stdio|acp|a2a]` (default http). Note: `acp` exposes Agent Client Protocol instead of MCP (see [ACP](../acp/)); `a2a` exposes A2A instead of MCP.
- `--port / --host` (for HTTP; host defaults to `127.0.0.1`)
- `--instance-scope [shared|connection|request] `– choose how agent state is isolated
    - `shared` (default) reuses a single agent for all clients
    - `connection` (sessions) Create one Agent per MCP session (separate history per client)
    - `request` (stateless) - create a new Agent for every tool call and disable MCP Sessions
- `--shell`, `-x` – Enable local shell tool access (bash or pwsh)
- `--no-shell` – Disable local shell/filesystem tools even when skills or config request them
- `--workspace` – Override the workspace root; when `--home` is omitted, the home defaults to `<workspace>/.fast-agent`
- `--no-home` – Run without implicit home side effects (no implicit card discovery, no session persistence/resume, and no ACP permission-store writes)
- `--no-permissions` – Disable ACP tool permission requests
- `--prefer-local-shell` – In ACP shell mode, use fast-agent's local shell runtime instead of the ACP client's terminal capability
- `--missing-shell-cwd [ask|create|warn|error]` – Override the shell missing-cwd policy
- `--reload` – Enable manual AgentCard reloads. `--watch` is not supported for MCP serving because clients discover a fixed tool surface at startup.

Standard CLI flags also apply (e.g. `--config-path`, `--model`, `--servers`, `--url`, `--auth`, `--client-metadata-url`, `--agent-cards`, `--card-tool`, `--stdio`, `--npx`, `--uvx`, and global `-q/--quiet`).
This allows **`fast-agent`** to serve any existing MCP Server in "Agent Mode", use custom system prompts and so on.

`--no-home` conflicts with `--home` (they cannot be used together).

HTTP serving binds to loopback by default. Use `--host 0.0.0.0` or another
non-loopback address only when remote clients should connect; `fast-agent serve`
prints a warning for remote HTTP/A2A binds. It also prints a warning whenever
`--shell` is enabled, with stronger wording when shell access is exposed to
remote callers.

For public or multi-user hosted servers, prefer `--instance-scope request`.
Use `shared` only for trusted deployments or application-level shared state you
intend all callers to see.

Managed MCP serving publishes one MCP tool per served AgentCard/agent. The
tool name, description, and optional structured input schema come from the
AgentCard:

```yaml
name: weather
description: Answer questions about current weather.
tool_input_schema:
  type: object
  properties:
    location:
      type: string
  required: [location]
```

If no `tool_input_schema` is set, the tool uses a simple `message` string
schema. For a different MCP interface, write a custom FastMCP harness server
and register the tools you want.

Examples:

```bash
fast-agent serve --agent-cards ./agents/weather.md --transport http
```

This publishes the card as an MCP tool named `weather`.

```bash
fast-agent serve \
  --url https://huggingface.co/mcp \
  --instance-scope connection \
  --model haiku
```

This starts a Streamable HTTP MCP Server on port 8000, providing access to an Agent connected to the Hugging Face MCP Server using Anthropic Haiku.



```bash
fast-agent serve \
  --npx @modelcontextprotocol/server-everything \
  --instance-scope request \
  -i system_prompt.md \
  --model kimi
```

This starts a Streamable HTTP MCP Server on port 8000, providing agent access to  the STDIO version of the "Everything Server" with a custom system prompt.  

#### Running an agent

If you already have an agent module or workflow (e.g. the generated agent.py), you can start it as a server directly:

```bash
uv run agent.py --transport http [OPTIONS]
```

The embedded CLI parser supports the same server flags as the serve command:

- `--transport`, `--host`, `--port`
- `--instance-scope [shared|connection|request]`
- `--quiet`, `--model`, and other agent startup options

Example:

```bash
uv run agent.py \
--transport http \
--port 8723 \
--instance-scope request
```

`--transport` enables server mode automatically.

Both approaches initialise FastAgent with the same config and skill loading pipeline;
choose whichever fits your workflow (one-off CLI invocation vs. packaging an agent as
a reusable script).
