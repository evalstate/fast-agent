### Running as an MCP Server

**`fast-agent`** Can deploy any configured agents over MCP, letting external MCP clients connect via STDIO, SSE, or HTTP. 

Additionally, there is a convenient `serve` command enabling rapid, command line deployment of MCP enabled agents in a variety of instancing modes.

This feature also works with [Agent Skills](../agents/skills.md), enabling powerful adaptable behaviours.

#### Using the CLI (fast-agent serve)

```bash
fast-agent serve [OPTIONS]
```

Key options:

- `--transport [http|sse|stdio|acp]` (default http). Note: `acp` exposes Agent Client Protocol instead of MCP (see [ACP](../acp/index.md)).
- `--port / --host` (for HTTP/SSE)
- `--instance-scope [shared|connection|request] `– choose how agent state is isolated
    - `shared` (default) reuses a single agent for all clients
    - `connection` (sessions) Create one Agent per MCP session (separate history per client)
    - `request` (stateless) - create a new Agent for every tool call and disable MCP Sessions
- `--description` – Customise the MCP tool description (supports {agent} placeholder)
- `--shell`, `-x` – Enable local shell tool access (bash or pwsh)
- `--noenv`, `--no-env` – Run without implicit environment side effects (no implicit card discovery, no session persistence/resume, and no ACP permission-store writes)

Standard CLI flags also apply (e.g. `--config-path`, `--model`, `--servers`, `--stdio`, and global `-q/--quiet`).
This allows **`fast-agent`** to serve any existing MCP Server in "Agent Mode", use custom system prompts and so on.

`--noenv` conflicts with `--env` (they cannot be used together).

Examples:

```bash
fast-agent serve \
  --url https://huggingface.co/mcp \
  --instance-scope connection \
  --description "Interact with the {agent} workflow" \
  --model haiku
```

This starts a Streamable HTTP MCP Server on port 8000, providing access to an Agent connected to the Hugging Face MCP Server using Anthropic Haiku.



```bash
fast-agent serve \
  --npx @modelcontextprotocol/server-everything \
  --instance-scope request \
  --description "Ask me anything!" \
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
- `--description` (tool instructions)
- `--quiet`, `--model`, and other agent startup options

Example:

```bash
uv run agent.py \
--transport http \
--port 8723 \
--instance-scope request
```

`--transport` now enables server mode automatically. The legacy `--server` flag is still accepted as an alias but is deprecated.

Both approaches initialise FastAgent with the same config and skill loading pipeline;
choose whichever fits your workflow (one-off CLI invocation vs. packaging an agent as
a reusable script).
