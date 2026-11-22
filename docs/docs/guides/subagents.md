---
title: Subagents
description: Use model-controlled temporary subagents or define reusable agents with Agent Cards.
social:
  title: Subagents
  tagline: Model-controlled delegation or user-defined agents.
  description: Use model-controlled temporary subagents or define reusable agents with Agent Cards.
  alt: fast-agent social card — Subagents
---

# Subagents

fast-agent supports two simple approaches to subagents.

## 1. Model controlled

Model-controlled subagents are temporary, one-shot agents. The parent model
decides when to create them and what task to delegate.

Enable them from the command line:

```bash
fast-agent go --subagents
```

Or use `-xx` to enable both subagents and shell access:

```bash
fast-agent go -xx
```

You can also opt in from a system prompt. The default agent includes the
project's `AGENTS.md`, so the directive can be placed there:

```md
<!-- fast-agent-subagents -->
```

`--no-subagents` or `subagents: false` always overrides the directive.

For a persistent configuration, use Agent Card flags:

```yaml
subagents: true
subagent_model: $system.fast  # optional
harness_tools: true
```

- `subagents` gives a normal parent agent the built-in `subagent` tool.
- `subagent_model` optionally fixes the model used by every temporary child.
- `harness_tools` is the model-facing self-management surface for fast-agent.
  It provides allow-listed access to harness information and resources, and
  lets the model use `/mcp` and `/skills` to connect servers or manage installed
  skills. Ad-hoc local MCP commands also require shell access, as provided by
  `-xx`. Enable these options only when those model-controlled changes are
  appropriate.

A child using a different model keeps harness settings such as the tool-loop
limit and stream timeout, but uses the selected model's request defaults
instead of inheriting sampling settings, output-token limits, or service
tiers. Explicit settings on the child's model string are still validated.
When the provider or model overlay changes, the child uses the destination's
credentials and connection settings. `/model switch` validates the replacement
before starting a new conversation in the running harness. A run-level
`--base-url` stays in effect only while the provider and overlay are unchanged.

## 2. User defined

Use [Agent Cards](../agents/defining/agent_cards.md) when you want named,
reusable agents rather than temporary model-created children.

An Agent Card can define:

- the agent's instructions and description;
- its model or model reference;
- tools, skills, and MCP servers; and
- the description shown to a parent model when the card is loaded as a tool.

This is the better option for stable roles such as a reviewer, researcher, or
project-specific specialist.
