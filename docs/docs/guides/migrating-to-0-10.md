---
title: Migrating to 0.10
description: Migrate from removed Smart agents to opt-in built-in subagents in fast-agent 0.10.
---

# Migrating to 0.10

## Breaking change: Smart agents are removed

fast-agent 0.10 removes Smart agents, the model-visible `smart` tool, the
`--smart` flag, and `type: smart`. Remove Smart prompts and card-pack examples
instead of treating them as compatibility configuration.

AgentCards and ToolCards remain the way to define configured specialists.
Use a ToolCard when a parent should have a stable specialist with its own
instructions, tools, or model.

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

## Update cards and automation

For every former Smart card:

1. Delete `type: smart`.
2. Keep `shell: true`, function tools, MCP servers, skills, and other normal
   AgentCard fields as needed.
3. Add `subagents: true` only when the former agent needs temporary delegation.
4. Move durable specialist definitions into AgentCards or ToolCards rather than
   creating them dynamically.

This is a breaking 0.10 migration; do not retain `--smart`, `type: smart`, or
Smart prompt resources in scripts, cards, or documentation.
