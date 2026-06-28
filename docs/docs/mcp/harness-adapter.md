---
title: Build MCP Servers with the Harness API
social:
  title: Build MCP Servers with the Harness API
  tagline: Use fast-agent agents from custom FastMCP servers.
  description: Use fast-agent agents from custom FastMCP servers.
  alt: fast-agent social card — Build MCP Servers with the Harness API
---

# Build MCP Servers with the Harness API

Use the MCP harness adapter when you want fast-agent to manage agents,
models, tools, skills, and session lifecycle, while you keep control of the
FastMCP server surface.

This page describes **custom tool adapter mode**: you own the `FastMCP` server,
register ordinary `@mcp.tool()` handlers, and call fast-agent through
`HarnessMCPAdapter`. The default `fast-agent serve` path is **managed MCP
server mode**. For interactive UI surfaces, use [MCP Apps adapter mode](fastmcp-apps.md).

If you just want to expose an agent quickly, use `fast-agent serve`:

```bash
fast-agent serve --transport http --instance-scope request
```

Use the adapter when you want custom MCP tools, custom arguments, custom
return values, FastMCP auth checks, or MCP Apps.

## The model

```text
FastMCP tool call
└─ HarnessMCPAdapter
   └─ HarnessApp.open(...)
      └─ session.invoke(AgentRequest)
         └─ AgentResponse
```

The adapter is only for MCP/FastMCP integration. If you are embedding
fast-agent directly in your own Python application, use the Harness API and
manage sessions yourself with `HarnessApp.open(...)` or `HarnessSessions`.

## Minimal custom FastMCP server

```python
from fastmcp import FastMCP

from fast_agent import FastAgent
from fast_agent.mcp.server import HarnessMCPAdapter, HarnessMCPAdapterOptions

fast = FastAgent("repo tools")
mcp = FastMCP("repo tools")


@fast.agent(name="researcher", instruction="Research repositories and summarize findings.")
async def researcher() -> None:
    pass


async def main() -> None:
    async with fast.harness() as harness:
        adapter = HarnessMCPAdapter(
            harness.app(),
            HarnessMCPAdapterOptions(default_agent="researcher"),
        )

        adapter.register_agent_tool(
            mcp,
            name="research",
            agent="researcher",
            description="Research a topic and return a concise answer.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "depth": {"type": "string", "enum": ["quick", "deep"]},
                },
                "required": ["topic"],
            },
            render_arguments="Research {{topic}}.\nDepth: {{depth}}",
        )

        await mcp.run_http_async(host="0.0.0.0", port=8000)
```

## `message` vs `arguments`

The adapter accepts either direct prompt input or structured MCP tool
arguments.

Use `message` when the MCP tool already has prompt text:

```python
await adapter.invoke_agent(
    ctx=ctx,
    agent="support",
    message="Help this customer reset their API key.",
)
```

Use `arguments` when the MCP tool has structured inputs:

```python
await adapter.invoke_agent(
    ctx=ctx,
    agent="researcher",
    arguments={"repo": "fast-agent-ai/fast-agent", "depth": "quick"},
)
```

Exactly one of `message` or `arguments` is required. Structured arguments are
rendered using the same conventions as AgentCards and Agents-as-Tools.

For common cases, register a named agent-backed MCP tool directly:

```python
adapter.register_agent_tool(
    mcp,
    name="research",
    agent="researcher",
    description="Research a topic and return a concise answer.",
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "depth": {"type": "string", "enum": ["quick", "deep"]},
        },
        "required": ["topic"],
    },
    render_arguments="Research {{topic}}.\nDepth: {{depth}}",
)
```

`render_arguments` uses the same `{{field}}` placeholder style as batch
generation. Non-string fields are JSON encoded. Use `{{arguments_json}}` when
you want the full argument object.

If you omit `input_schema`, the tool accepts a single `message` argument and
sends that text to the agent.

## Sessions and handles

MCP has an ambient `Mcp-Session-Id`, but it is not a universal application
state handle. Server authors choose the state model.

For request-scoped tools, each MCP tool call opens a transient harness session.
If your tool needs durable state, expose your own handle:

```python
@mcp.tool()
async def continue_review(repo: str, review_id: str, ctx: MCPContext) -> str:
    response = await adapter.invoke_agent(
        ctx=ctx,
        agent="reviewer",
        session_id=review_id,
        arguments={"repo": repo, "review_id": review_id},
    )
    return response.text_content()
```

The adapter preserves the ambient MCP session id as request metadata when it is
not the effective harness session key.

## Return values

The adapter returns an `AgentResponse`. Your FastMCP tool decides what to return
to the client.

Text:

```python
response = await adapter.invoke_agent(...)
return response.text_content()
```

Structured output:

```python
response = await adapter.invoke_agent(...)
return {
    "kind": response.kind,
    "text": response.text_content(),
    "metadata": response.metadata,
}
```

MCP Apps can project the same response into UI components.

## Auth and progress

The adapter reads auth and request context from FastMCP:

- verified bearer tokens become `AgentAuth`;
- progress reports are forwarded to the MCP client while a call is active;
- fast-agent loop/tool progress is forwarded the same way as managed MCP server
  mode;
- request-scoped bearer context is available to providers and downstream MCP
  connections that support auth forwarding.

Use FastMCP per-tool auth checks for privileged tools:

```python
@mcp.tool(auth=requires_inference_scope)
async def run_inference(prompt: str, ctx: MCPContext) -> str:
    response = await adapter.invoke_agent(ctx=ctx, message=prompt)
    return response.text_content()
```

## Default tools

The owned fast-agent MCP server uses the same adapter and registers a default
`send` tool. Custom servers can register the same default surface when useful:

```python
adapter.register_default_tools(mcp)
```

Most custom servers should prefer explicit tools with domain-specific
arguments.
