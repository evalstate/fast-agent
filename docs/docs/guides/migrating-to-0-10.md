---
title: Migrating to 0.10
description: Migrate configuration, MCP APIs, commands, and agents to fast-agent 0.10.
---

# Migrating to 0.10

## MCP configuration and commands

MCP servers now use the canonical `mcp.servers` map. Preview and apply the
round-trip migration with:

```bash
fast-agent config migrate-mcp fast-agent.yaml
fast-agent config migrate-mcp fast-agent.yaml --write
```

The write command preserves the exact original as `fast-agent.yaml.bak`. Use
`/mcp attach NAME` or `/connect NAME` for configured definitions. `/connect`
also accepts ad-hoc targets; `/mcp connect TARGET` is the explicit ad-hoc form.
See [Migrate MCP Configuration](../mcp/migration.md) for the before/after schema
and conflict rules.

## Breaking change: MCP SDK v2 uses snake_case Python fields

fast-agent 0.10 uses `mcp==2.0.0` and `mcp-types==2.0.0`. JSON-RPC wire fields
remain camelCase, but Python constructors, attributes, and assignments use
snake_case:

| MCP wire or v1 spelling | Python SDK v2 spelling |
|---|---|
| `inputSchema` | `input_schema` |
| `outputSchema` | `output_schema` |
| `mimeType` | `mime_type` |
| `structuredContent` | `structured_content` |
| `nextCursor` | `next_cursor` |
| `isError` | `is_error` |

Use the canonical `mcp_types` package for protocol models:

```python
from mcp_types import CallToolResult, ImageContent

image = ImageContent(type="image", mime_type="image/png", data=encoded)
result = CallToolResult(content=[], structured_content={"status": "ok"})
```

CamelCase remains correct in serialized JSON payloads. Do not use wire aliases
for Python attribute access or assignment.

## Breaking change: slash-command grammar

Several command families were consolidated:

| Before 0.10 | 0.10 replacement |
|---|---|
| `/cards ...` | `/packs ...` |
| `/models ...` | `/model ...` |
| `/card PATH --tool` | `/card load PATH --as-tool` |
| `/agent NAME --tool` | `/agent tool add NAME` |
| removing an agent tool with flags | `/agent tool remove NAME` |
| agent/card dump flags | `/agent show [NAME]` or `/card show [NAME]` |

The removed command names are not compatibility aliases; update saved
automations and editor command palettes before upgrading.

## Removed MCP and quickstart surfaces

The built-in `prompt-server` console executable was removed. Application-owned
prompt files can be loaded with `fast_agent.load_prompt`; shared remote prompts
should be exposed by an external MCP prompt/resource server.

The old terminal MCP-UI hosting settings and APIs were also removed:

```yaml
mcp_ui_mode: ...
mcp_ui_output_dir: ...
```

0.10 rejects these YAML keys with migration guidance instead of silently
ignoring them. Use [MCP Apps](../mcp/mcp-apps.md), the
[OpenAI Apps SDK](../mcp/openai-apps-sdk.md), or
[FastMCP Apps](../mcp/fastmcp-apps.md) integration metadata. fast-agent
discovers, validates, and displays app metadata; it does not restore the old
embedded terminal HTML runtime.

## FastMCP 4 beta compatibility

fast-agent 0.10 pins `fastmcp-slim[server]==4.0.0b2`, the newest published
FastMCP release compatible with MCP SDK v2 at release time. FastMCP 3.4.5 is
stable but requires MCP SDK v1 and is not compatible with this release.

Treat custom FastMCP server and FastMCP Apps integrations as beta until a stable
FastMCP 4 release is available. fast-agent uses exact dependency pins and will
evaluate later FastMCP 4 releases independently.

## Breaking change: authentication commands are domain-specific

Provider credentials now live under `auth provider`:

```bash
fast-agent auth provider list
fast-agent auth provider show codex
fast-agent auth provider login codex
fast-agent auth provider logout codex
fast-agent auth provider token codex
fast-agent auth provider export codex ./codex.auth.json
```

The former root commands (`auth login`, `auth logout`, `auth status`,
`auth token`, and `auth export`) now exit with migration guidance instead of
performing an operation.

MCP authentication uses configured server names positionally:

```bash
fast-agent auth mcp list
fast-agent auth mcp show myserver
fast-agent auth mcp login myserver
fast-agent auth mcp credentials
fast-agent auth mcp forget myserver
```

Ad-hoc URLs must be exact and explicit:

```bash
fast-agent auth mcp login --endpoint https://example.com/custom/mcp
```

fast-agent no longer treats an unknown positional name as a URL-derived
credential identity and no longer appends `/mcp` or `/sse` in auth commands.
The user-facing term **OAuth resource** replaces **identity**. To remove a
stored resource directly, use:

```bash
fast-agent auth mcp forget --resource https://example.com
```

`forget` removes local OAuth tokens and client registration. It does not alter
server configuration or runtime connections, and it lists every configured
server sharing the credential before confirmation.

All 0.10 token and client-registration writes are indexed. Historical
client-registration-only records are backfilled when their resource remains
configured. An unindexed record for a removed server can still be removed with
`auth mcp forget --resource <exact-url>` when that URL is known.

The former `auth mcp status` and `auth mcp logout` commands now exit with the
appropriate `list`, `show`, `credentials`, or `forget` replacement.

## Breaking change: no implicit default model

fast-agent no longer falls back to `gpt-5.4-mini?reasoning=low` when no model
is configured. Model resolution now uses, in order:

1. an explicit AgentCard or decorator model;
2. `--model`;
3. `default_model` in `fast-agent.yaml`;
4. `FAST_AGENT_MODEL`.

Interactive, non-resumed runs open the model picker when none of these sources
resolves. The selected model is recorded as `$system.last_used` for future
picker initialization. Picker history no longer implicitly satisfies
`$system.default`; configure `model_references.system.default` when using that
reference as an unattended default.

`--resume` does not open the picker. It uses the model saved in the session
snapshot and restores each persisted agent model during hydration.

Unattended runs cannot use the picker. CI, one-shot, `--no-home`, server,
batch, and programmatic harness runs must configure or pass a model:

```bash
fast-agent go --model sonnet --message "Summarize the release notes"
FAST_AGENT_MODEL=sonnet fast-agent serve --transport http
```

Without a model, startup now fails with `No model configured` and lists the
supported configuration sources.

MCP sampling follows the same rule. Configure
`mcp.servers.<name>.sampling.model`, an agent model, or a global model source;
an empty `sampling: {}` block no longer selects a built-in model.

## Breaking change: Smart agents are removed

fast-agent 0.10 removes Smart agents, the model-visible `smart` tool, and the
`--smart` flag. `type: smart` remains as a deprecated 0.10 compatibility alias
and will be removed in 0.11. It is treated as:

```yaml
type: agent
subagents: true
harness_tools: true
```

Loading the alias emits a warning. Explicit `subagents` or `harness_tools`
values take precedence, which supports incremental migration. The alias does
not restore the legacy `smart` tool, but its harness tools support the `/mcp`
and `/skills` command families.

The Python class was also removed. Imports such as:

```python
from fast_agent.agents import SmartAgent
```

must migrate to the normal agent APIs plus explicit `subagents` and
`harness_tools` configuration.

AgentCards and ToolCards remain the way to define configured specialists.
Use a ToolCard when a parent should have a stable specialist with its own
instructions, tools, or model.

The former Smart agent's harness-facing tools are now an independent opt-in on
basic agents:

```yaml
harness_tools: true
```

This adds the allow-listed `slash_command` and `get_resource` tools without
changing the agent type. The slash-command surface includes model-controlled
MCP connections and skill management. Combine it with `subagents: true` when
both harness management and temporary delegation are wanted.

## Replace dynamic delegation

The built-in `subagent` tool is now opt-in:

```yaml
---
name: dev
shell: true
subagents: true
---
```

Set `subagents: true` on each parent AgentCard that should delegate temporary
work. Tool-only cards and built-in child agents do not receive the tool.

To force every built-in child to use one model, add `subagent_model`:

```yaml
subagents: true
subagent_model: $system.fast
```

From the CLI, use `--subagents`. `-xx` is shorthand for shell access plus
subagents, and `--subagent-model <model>` enables subagents while forcing the
child model:

```bash
fast-agent go -xx --subagent-model '$system.fast'
```

Use `-x` alone when shell access is needed without delegation.

For repository-level activation, add an exact standalone directive to
`AGENTS.md`:

```md
<!-- fast-agent-subagents -->
```

The marker itself is removed from the model-visible instruction. It only
applies when `subagents` is unset; `--no-subagents` and `subagents: false`
take precedence.

The comment may also contain parent-only instructions:

```md
<!-- fast-agent-subagents
use terra for analysis
-->
```

The body remains in the parent system instruction, but the complete comment is
excluded from built-in subagent instructions.

## Update cards and automation

For every former Smart card:

1. Replace `type: smart` with `type: agent` before the compatibility alias is
   removed in 0.11.
2. Keep `shell: true`, function tools, MCP servers, skills, and other normal
   AgentCard fields as needed.
3. Add `subagents: true` only when the former agent needs temporary delegation.
4. Add `harness_tools: true` when the former agent used `slash_command` or
   `get_resource`.
5. Move durable specialist definitions into AgentCards or ToolCards rather than
   creating them dynamically.

Do not retain `--smart` or Smart prompt resources. Migrate deprecated
`type: smart` cards during the 0.10 release series.
