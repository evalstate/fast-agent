---
title: AgentCards and ToolCards
description: How fast-agent loads AgentCards as runnable agents vs attached tools,
  including defaults for agent-cards and tool-cards directories.
social:
  title: AgentCards and ToolCards
  tagline: Load AgentCards as runnable agents or attach ToolCards as callable tools.
  description: Load AgentCards as runnable agents or attach ToolCards as callable
    tools.
  alt: fast-agent social card — AgentCards and ToolCards
---


# AgentCards and ToolCards

## Quick answer

In fast-agent, **ToolCards are AgentCards**. There is no separate card schema.

The distinction is **how cards are loaded**:

- `--agent-cards` (or `--card`) loads cards as runnable agents.
- `--card-tool` loads cards, then attaches those loaded agents as tools to a parent agent.

## Card file format

AgentCards can be Markdown+frontmatter or YAML:

- `.md`
- `.markdown`
- `.yaml`
- `.yml`

## Default directories

By default, `fast-agent go` discovers cards from your environment directory:

- `<env>/agent-cards/`
- `<env>/tool-cards/`

`<env>` defaults to `.fast-agent/` in your current project root.
Use `--env` to point to a different environment directory.
Use `--noenv` to disable implicit default directory discovery entirely.

## Recommended usage

Use `--agent-cards` for agents you want to run directly.

Use `--card-tool` for agents you primarily want to invoke as tools from another agent.

If a card should not appear in normal interactive agent lists, set:

```yaml
tool_only: true
```

## Runtime MCP targets (`mcp_connect`)

Use `mcp_connect` when a card needs MCP servers that are **not** preconfigured
under `mcp.servers` in `fast-agent.yaml`.

```yaml
mcp_connect:
  - target: "https://demo.hf.space"
    headers:
      Authorization: "Bearer ${DEMO_TOKEN}"
    auth:
      oauth: true
  - target: "@modelcontextprotocol/server-everything"
    name: "everything"
```

- `target` (required): URL, `@pkg`, `npx ...`, `uvx ...`, or stdio command.
- `name` (optional): explicit server alias; if omitted, fast-agent infers one.
- `headers` (optional): structured HTTP headers.
- `auth` (optional): structured auth settings (for example `oauth: true`).

For provider-managed remote MCP, use:

```yaml
mcp_connect:
  - target: "https://huggingface.co/mcp"
    name: "huggingface"
    management: provider
    access_token: "${HF_TOKEN}"
    description: "Hugging Face MCP"
```

- `management: provider` delegates remote MCP execution to the LLM provider.
- `target` must be a URL-based remote server when `management: provider` is used.
- `access_token` is the bearer token for the remote MCP server.
- `description` is optional provider-facing metadata.
- `defer_loading` is an OpenAI Responses hint for lazy remote tool loading.
- Do not use `headers` or `auth` with provider-managed entries; use `access_token` instead.

Provider-managed card targets are supported only for agents using:

- `anthropic`
- `responses`

They are not supported for `codexresponses`, Codex OAuth aliases, `openresponses`,
`anthropic-vertex`, or other providers.

OpenAI Responses connectors can also be declared as structured provider-managed
card entries. Use `connector_id` instead of `target`:

```yaml
mcp_connect:
  - name: dropbox
    management: provider
    connector_id: connector_dropbox
    access_token: "${DROPBOX_OAUTH_ACCESS_TOKEN}"
    description: "Dropbox connector"
    defer_loading: true
```

Connector-backed entries are supported only by the OpenAI `responses` provider.
They require `access_token`; omit `target`, `transport`, `headers`, and `auth`.

For provider-managed servers, use exact tool names in `tools.<server_name>`.
Wildcard tool filters, prompt filters, and resource filters are not supported.

`target` is a pure target string. Do not embed fast-agent CLI flags (like
`--auth` or `--oauth`) in card targets. Use `headers`/`auth` fields instead.

When both target-derived values and explicit fields are present, explicit fields
(`headers`, `auth`, etc.) win.

If an inferred/provided name collides with another server using different settings,
startup fails with a collision error. Prefer explicit `name` values for stability.

## Remote A2A AgentCards

A remote A2A agent can be loaded as a first-class fast-agent agent with
`type: a2a`:

```yaml
type: a2a
name: hello_remote
url: http://127.0.0.1:41241
transport: JSONRPC
```

`url` is the A2A agent base URL used to resolve the remote card at
`/.well-known/agent-card.json`. If the card is served elsewhere, set
`relative_card_path`:

```yaml
type: a2a
name: remote_custom_card
url: https://agent.example.com
relative_card_path: /custom/agent-card.json
transport: HTTP+JSON
```

Supported transport names are `JSONRPC`, `HTTP+JSON`, and `GRPC`. If `transport`
is omitted, the A2A SDK chooses from the remote AgentCard's advertised
interfaces. The current client maps text, URL resources, structured data, and
raw binary parts into fast-agent messages; inbound raw bytes are represented
safely with filename/media type/byte count.

For one-off CLI connections, use `--a2a` instead of writing a card:

```bash
fast-agent go --a2a http://127.0.0.1:41241 --a2a-transport JSONRPC --message hello
```

In the TUI, use `/a2a connect http://127.0.0.1:41241 --transport HTTP+JSON --name hello_rest`.

## Examples

```bash
# Load runnable agents
fast-agent go --agent-cards ./agents

# Load cards as tools attached to the default/selected agent
fast-agent go --card-tool ./tool-cards

# Mix both
fast-agent go --agent-cards ./agents --card-tool ./tool-cards

# Ephemeral/noenv run: only explicit paths are loaded (no implicit <env>/agent-cards or <env>/tool-cards)
fast-agent go --noenv --agent-cards ./agents --card-tool ./tool-cards

# Target a specific loaded agent
fast-agent go --agent-cards ./agents --agent researcher
```

## Notes on `--agent`

- `--agent` picks the target for `--message`, `--prompt-file`, and initial interactive mode.
- `--agent` can also target explicitly loaded tool-only agents when needed for testing.
