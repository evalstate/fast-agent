# Dreamtune MCP example

Connect fast-agent to [Dreamtune](https://dreamtunelabs.com) for model APIs, tools, GPU compute and run inspection. The example uses fast-agent's built-in `passthrough` model, so no separate LLM provider key is needed to discover tools or invoke one explicitly.

## Setup

Install [uv](https://docs.astral.sh/uv/) and Node.js **22.20+** with `npx` on your PATH. From the repository root:

```sh
uv sync --frozen
cd examples/mcp/dreamtune
export DREAMTUNE_API_KEY="YOUR_DREAMTUNE_API_KEY"
uv run example.py
```

In PowerShell, set `$env:DREAMTUNE_API_KEY = "YOUR_DREAMTUNE_API_KEY"` instead of `export`.

The configuration reads the key from your environment using fast-agent's `${DREAMTUNE_API_KEY}` substitution. It pins the published `@dreamtune/sdk` stdio server and sets the production API origin and request timeout explicitly. Keep real keys out of configuration committed to Git.

## Discover and invoke

The default run initializes the MCP connection, prints tool names and descriptions, then closes it. It does not execute model inference, paid tool runs or compute jobs.

Use a name from that output and arguments from the [MCP tool reference](https://docs.dreamtunelabs.com/reference/mcp-tools) to make an explicit call:

```sh
uv run example.py --tool dreamtune__compute_catalog --arguments '{}'
```

The example routes the call through the documented `***CALL_TOOL` passthrough command. To use Dreamtune in your own LLM-backed agent, reuse the `mcp.servers.dreamtune` configuration and `servers=["dreamtune"]`, then choose your normal model instead of `passthrough`.

Calls use the scopes and workspace of your key. Model runs, paid tool runs and compute jobs can incur charges; compute may continue billing until stopped. This example invokes a tool only when `--tool` is supplied.

See [local MCP setup](https://docs.dreamtunelabs.com/connect/local-mcp) for optional GPU access and [hosted MCP setup](https://docs.dreamtunelabs.com/connect/mcp) for Streamable HTTP clients.
