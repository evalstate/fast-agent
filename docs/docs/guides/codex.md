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

Codex Responses includes [stable prompt-cache keys and cache-preserving Astra effort changes](responses-caching.md).

Use the `codex` pack to start **fast-agent** with a coding agent, a
Codex-optimised filesystem search sub-agent, Responses-family transport, and
model-selected filesystem editing tools.

```bash
uvx fast-agent-mcp@latest --pack codex
```

This starts **fast-agent** pre-configured for a Codex-style coding workflow.

## What the `codex` pack gives you

- A `dev` coding agent for interactive software work
- A bounded rg-first search helper backed by `codexspark`
- WebSocket-capable transport for modern Codex/OpenAI models
- Filesystem editing tools selected for the model: Astra defaults to
  `write_text_file` plus `edit_file`; patch-oriented models use `apply_patch`
- Preconfigured MCP targets available from `/mcp attach`

The coding agent has a minimal system prompt plus tools for the shell,
filesystem and **fast-agent** services. `AGENTS.md` is included automatically if
present. Customise the agent by editing `.fast-agent/agent-cards/dev.md`.

Astra's writer/editor default applies to both Responses and Codex Responses,
including the `astra` and `codexplan` aliases. To explicitly select the
Codex-style patch interface instead:

```yaml
shell_execution:
  write_text_file_mode: apply_patch
```

When `apply_patch` tool calls are previewed in the console, large patches are
collapsed with a `(+N more lines)` tail. You can tune or disable that limit via
`logger.apply_patch_preview_max_lines` in `fast-agent.yaml`, or interactively
with `fast-agent config display`.

Use `/skills` to discover, add, remove and update skills. Use `/connect` to
connect to MCP Servers.

## Codex authentication

If you want to use the Codex OAuth models directly, authenticate once first:

```bash
fast-agent auth provider login codex
```

Login uses device auth by default: open the displayed URL in a browser and enter
the one-time code. No local callback server is required. Only enter a code from
a login you started yourself.

If device auth is unavailable for your account, use the existing browser callback
flow instead:

```bash
fast-agent auth provider login codex --method browser
```

Device login failures do not automatically switch to browser login. Credentials
are stored in your OS keyring, with a secure file fallback. After that you can use
Codex OAuth model aliases such as:

- `codexplan` — GPT-6-Astra with medium reasoning

If you prefer, you can also run model setup explicitly:

```bash
uvx fast-agent-mcp@latest model setup
```

## Web search

Use `fast-agent go --model 'astra?web_search=true'` to enable automatic `web_run`
on Codex Lite without shell access; use `web_search=false` to disable it.
In a running conversation, `/model web_search on` and `/model web_search off`
control the same feature. Sol and public Responses keep their hosted search route.
See [standalone web search](../models/providers/openai.md#standalone-web-search-codex-lite)
for configuration, the internal endpoint caveat, and a runnable library example
with caller-supplied authentication.

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
- Use `/mcp attach <name>` to attach preconfigured MCP servers; use `/connect`
  for ad-hoc targets
- Ask the agent to create additional cards in `.fast-agent/agent-cards/`
- Switch agents with `@`
- Configure compaction, hooks or automation with the available skills
