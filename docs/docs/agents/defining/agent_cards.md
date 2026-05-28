---
title: Agent Cards
description: How fast-agent loads AgentCards as runnable agents vs attached tools,
  including defaults for agent-cards and tool-cards directories.
social:
  title: Agent Cards
  tagline: Load AgentCards as runnable agents or attach ToolCards as callable tools.
  description: Load AgentCards as runnable agents or attach ToolCards as callable
    tools.
  alt: fast-agent social card — Agent Cards
---


# Agent Cards

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
They require `name` and `access_token`; omit `target`, `transport`, `headers`, and `auth`.

For provider-managed servers, use exact tool names in `tools.<server_name>`.
Wildcard tool filters, prompt filters, and resource filters are not supported.

`target` is a pure target string. Do not embed fast-agent CLI flags (like
`--auth` or `--oauth`) in card targets. Use `headers`/`auth` fields instead.

When both target-derived values and explicit fields are present, explicit fields
(`headers`, `auth`, etc.) win.

If an inferred/provided name collides with another server using different settings,
startup fails with a collision error. Prefer explicit `name` values for stability.

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
