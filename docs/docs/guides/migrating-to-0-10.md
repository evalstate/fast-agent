---
title: Migrating to 0.10
description: Migrate from removed Smart agents to opt-in built-in subagents in fast-agent 0.10.
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
not restore the legacy Smart tool or its mutation commands.

AgentCards and ToolCards remain the way to define configured specialists.
Use a ToolCard when a parent should have a stable specialist with its own
instructions, tools, or model.

The former Smart agent's harness-facing tools are now an independent opt-in on
basic agents:

```yaml
harness_tools: true
```

This adds the allow-listed `slash_command` and `get_resource` tools without
changing the agent type. Combine it with `subagents: true` when both harness
inspection and temporary delegation are wanted.

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
