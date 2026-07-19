---
title: Code with OpenAI Codex
description: An introduction to using `fast-agent` with the `codex` pack
social:
  title: Code with OpenAI Codex
  tagline: Use the Codex pack to bring coding agents into fast-agent workflows.
  description: Use the Codex pack to bring coding agents into fast-agent workflows.
  alt: fast-agent social card — Code with OpenAI Codex
---


# Code with Codex

Use the `codex` pack to start **fast-agent** with a coding agent, a
Codex-optimised filesystem search sub-agent, Responses-family transport, and an
`apply_patch` tool that matches the Codex CLI patch format.

```bash
uvx fast-agent-mcp@latest --pack codex
```

This starts **fast-agent** pre-configured for a Codex-style coding workflow.

## What the `codex` pack gives you

- A `dev` coding agent for interactive software work
- A bounded rg-first search helper backed by `codexspark`
- WebSocket-capable transport for modern Codex/OpenAI models
- An `apply_patch` tool with a familiar Codex CLI-style patch signature
- Preconfigured MCP targets available from `/connect`

The coding agent has a minimal system prompt plus tools for the shell,
filesystem and **fast-agent** services. `AGENTS.md` is included automatically if
present. Customise the agent by editing `.fast-agent/agent-cards/dev.md`.

When `apply_patch` tool calls are previewed in the console, large patches are
collapsed with a `(+N more lines)` tail. You can tune or disable that limit via
`logger.apply_patch_preview_max_lines` in `fast-agent.yaml`, or interactively
with `fast-agent config display`.

Use `/skills` to discover, add, remove and update skills. Use `/connect` to
connect to MCP Servers.

## Codex authentication

If you want to use the Codex OAuth models directly, authenticate once first:

```bash
fast-agent auth login codex
```

This stores the token in your OS keyring. After that you can use Codex OAuth
model aliases such as:

- `codexplan` — GPT-5.5

If you prefer, you can also run model setup explicitly:

```bash
uvx fast-agent-mcp@latest model setup
```

## Installation

**fast-agent** requires Python 3.12 or above. Install with:

```bash
uv tool install -U fast-agent-mcp
```

Or a specific version of Python:

```bash
uv tool install --python 3.12 -U fast-agent-mcp
```

This installs the `fast-agent` executable.

## Next steps

From the `fast-agent` prompt:

- Use `/skills` to view and manage skills
- Use `/connect` to connect to the preconfigured MCP servers
- Ask the agent to create additional cards in `.fast-agent/agent-cards/`
- Switch agents with `@`
- Configure compaction, hooks or automation with the available skills
