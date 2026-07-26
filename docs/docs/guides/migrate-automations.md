---
title: Migrate Your Automations
description: Convert common Claude Code, Codex, and OpenCode one-shot commands to fast-agent.
social:
  title: Migrate Your Automations
  tagline: Move existing agent scripts to fast-agent.
  description: A practical migration path for scheduled, CI, and application-driven agent work.
  alt: fast-agent social card — Migrate Your Automations
---

# Migrate Your Automations

Replace a one-shot coding-agent command with:

```bash
uvx fast-agent-mcp@latest go --no-home --shell ...
```

`--no-home` keeps the run isolated. `--shell` provides local shell and
filesystem tools.

## Convert a command

The converter supports one-shot commands from:

- Claude Code
- Codex
- OpenCode

It maps common prompt, model, reasoning variant, attachment, workspace, agent,
and JSON Schema options. It runs in your browser, does not use an LLM, and does
not send the command to a server.

<div class="fa-command-converter" data-fa-command-converter>
  <label for="fa-command-input">Claude Code, Codex, or OpenCode command</label>
  <textarea id="fa-command-input" data-fa-command-input rows="5" spellcheck="false" placeholder='claude -p "Review the current changes" --model sonnet'></textarea>
  <div class="fa-command-converter__actions">
    <button type="button" class="fa-btn fa-btn--primary fa-btn--sm" data-fa-command-convert>Convert</button>
    <button type="button" class="fa-btn fa-btn--sm" data-fa-command-copy disabled>Copy</button>
    <span data-fa-command-status aria-live="polite"></span>
  </div>
  <pre><code data-fa-command-output>Paste a command above.</code></pre>
</div>

Single-line commands produce a single line. Commands using shell `\` line
continuations produce a multiline fast-agent command.

Review converted commands that contain custom permission, session, or server
options.

## Claude Code

[Claude Code CLI reference](https://docs.anthropic.com/en/docs/claude-code/cli-usage)

```bash title="Claude Code"
claude -p "Review the current changes" --model sonnet
```

```bash title="fast-agent"
uvx fast-agent-mcp@latest go \
  --no-home \
  --shell \
  --model sonnet \
  --message "Review the current changes"
```

Claude Code `--output-format json` does not define the output shape. For
validated JSON from fast-agent, add a schema:

```bash
uvx fast-agent-mcp@latest go \
  --no-home \
  --shell \
  --model sonnet \
  --json-schema ./result.schema.json \
  --message "Review the current changes"
```

## Codex

[Codex non-interactive mode](https://developers.openai.com/codex/noninteractive)
uses `codex exec` for non-interactive runs.

```bash title="Codex"
codex exec --ephemeral --model gpt-5.5 "Review the current changes"
```

```bash title="fast-agent"
uvx fast-agent-mcp@latest go \
  --no-home \
  --shell \
  --model responses.gpt-5.5 \
  --message "Review the current changes"
```

Codex `--output-schema ./schema.json` maps to
`--json-schema ./schema.json`.

## OpenCode

[OpenCode CLI reference](https://dev.opencode.ai/docs/cli/)

```bash title="OpenCode"
opencode run \
  --model openai/gpt-5.5 \
  --variant high \
  --file report.pdf \
  "Review this report"
```

```bash title="fast-agent"
uvx fast-agent-mcp@latest go \
  --no-home \
  --shell \
  --model "responses.gpt-5.5?reasoning=high" \
  --attach report.pdf \
  --message "Review this report"
```

OpenCode `--dir PATH` maps to `--workspace PATH`.

## Load an AgentCard from a URI

`--card` accepts a local path, HTTP(S) URL, `file://` URI, or `hf://` URI. It is
repeatable.

```bash
uvx fast-agent-mcp@latest go \
  --no-home \
  --card https://example.com/agents/reviewer.md \
  --agent reviewer \
  --message "Review the current changes"
```

```bash
uvx fast-agent-mcp@latest go \
  --no-home \
  --card hf://buckets/your-name/agents/reviewer.md \
  --agent reviewer \
  --message "Review the current changes"
```

Prompt files, instructions, configuration files, card registries, and JSON
schemas can also be loaded from supported URIs. See
[`fast-agent go`](../ref/go_command.md) for the complete option list.
