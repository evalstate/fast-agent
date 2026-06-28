---
title: MCP with fast-agent
social:
  title: MCP with fast-agent
  tagline: Connect to MCP servers, expose agents over MCP, and use MCP protocol features.
  description: Connect to MCP servers, expose agents over MCP, and use MCP protocol features.
  alt: fast-agent social card — MCP Overview
---

**`fast-agent`** provides comprehensive MCP support as  both Clients and Server:

- **Client**: connect agents to local or remote MCP servers.
- **Server**: expose fast-agent agents, AgentCards, and Harness apps as MCP servers with [FastMCP](https://gofastmcp.com/getting-started/welcome).
- **Protocol features**: work with MCP types, resources, elicitations, state transfer,
  OAuth, UI content, and MCP Apps.

## Choose your path

| I want to...                                   | Start here                                                                              |
| ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| Connect an agent to MCP tools                  | [Connect to MCP Servers](client-servers.md)                                             |
| Authenticate to remote MCP servers             | [Client OAuth](mcp-oauth.md)                                                            |
| Inspect tools, transports, and server metadata | [Inspect MCP Servers](mcp_display.md)                                                   |
| Serve a fast-agent agent over MCP              | [Run an MCP Server](mcp-server.md)                                                      |
| Build custom FastMCP tools backed by agents    | [Custom MCP Servers](harness-adapter.md)                                                |
| Host an MCP server on Hugging Face Spaces      | [Host on Hugging Face Spaces](huggingface-spaces.md)                                    |
| Build interactive UI with FastMCP Apps         | [FastMCP Apps](fastmcp-apps.md)                                                         |
| Understand fast-agent's MCP content handling   | [Integration with MCP Types](#integration-with-mcp-types) and [Resources](resources.md) |

## Deployment modes

We use these names throughout the MCP docs:

| Mode                         | Use it when...                                     | Start here                                       |
| ---------------------------- | -------------------------------------------------- | ------------------------------------------------ |
| **MCP client mode**          | fast-agent should connect agents to MCP servers    | [Connect to MCP Servers](client-servers.md)      |
| **Managed MCP server mode**  | fast-agent should own the MCP server process       | [Run an MCP Server](mcp-server.md)               |
| **Custom tool adapter mode** | you want normal MCP tools backed by agents         | [Custom MCP Servers](harness-adapter.md)         |
| **MCP Apps adapter mode**    | you want interactive MCP Apps backed by agents     | [FastMCP Apps](fastmcp-apps.md)                  |
| **Direct Harness mode**      | you are embedding fast-agent in Python without MCP | [Harness API](../agents/defining/harness-api.md) |

`--transport http` and `--transport stdio` choose the wire transport for
managed MCP server mode. They are not separate deployment modes. Session scope
(`request`, `connection`, or legacy `shared`) is a separate choice.

The two adapter modes both use `HarnessMCPAdapter`; they differ only in the
FastMCP surface. Custom tool adapter mode uses ordinary `@mcp.tool()` handlers.
MCP Apps adapter mode uses `FastMCPApp.ui()` and `FastMCPApp.tool()` handlers so
FastMCP can serve UI resources, app metadata, CSP, permissions, and app-only
backend tools.

## Integration with MCP Types

fast-agent uses MCP protocol types throughout the runtime, so MCP content can
move between servers, agents, workflows, and protocol adapters without being
flattened to plain strings too early.

Conversations are based on `PromptMessageExtended`, fast-agent's extension of
the MCP `PromptMessage` type. It supports multiple content sections and is used
for:

- normal chat turns;
- MCP tool results and resource content;
- history transfer between agents;
- Harness API `AgentRequest` / `AgentResponse` messages;
- protocol adapters such as MCP, A2A, and ACP.

That means an agent can receive MCP-native text, images, embedded resources, or
other content blocks where the provider and client support them. When a target
only supports text, fast-agent projects the content to text at the adapter or
provider boundary.

Example: transfer one agent's message history to another agent:

```python title="history_transfer.py"
@fast.agent(name="haiku", model="haiku")
@fast.agent(name="openai", model="gpt-5.5")
async def main() -> None:
    async with fast.run() as agent:
        await agent.interactive(agent_name="haiku")
        await agent.openai.generate(agent.haiku.message_history)
        await agent.interactive(agent_name="openai")
```

For MCP resources and linked content, see [Resources](resources.md). For UI
content returned by MCP servers, see [mcp-ui and fast-agent](mcp-ui.md).

## Client mode

In client mode, fast-agent connects your agents to MCP servers. Configure MCP
servers in `fast-agent.yaml`, AgentCards, or with CLI flags such as `--url`,
`--stdio`, `--npx`, and `--uvx`.

Common client topics:

- [Connect to MCP Servers](client-servers.md)
- [Client OAuth](mcp-oauth.md)
- [Inspect MCP Servers](mcp_display.md)
- [MCP Resources](resources.md)
- [mcp-ui and fast-agent](mcp-ui.md)

## Server mode

In server mode, fast-agent exposes agents and workflows over MCP. Use the CLI for
simple deployments, or use the Harness API with FastMCP when you want a custom
server surface.

Common server topics:

- [Run an MCP Server](mcp-server.md)
- [Custom MCP Servers](harness-adapter.md)
- [Host on Hugging Face Spaces](huggingface-spaces.md)
- [FastMCP Apps](fastmcp-apps.md)
- [OpenAI Apps SDK](openai-apps-sdk.md)

## MCP features

fast-agent supports several MCP protocol features directly in the agent runtime:

- [Skills over MCP](skills-over-mcp.md)
- [Elicitations](elicitations.md)
- [State Transfer](state_transfer.md)
- [Resources](resources.md)
- [mcp-ui](mcp-ui.md)
