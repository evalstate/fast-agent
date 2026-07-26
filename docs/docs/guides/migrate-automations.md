---
title: Migrate Your Automations
description: Move scripted agent work to fast-agent with stable inputs, structured outputs, reusable agents, and portable configuration.
social:
  title: Migrate Your Automations
  tagline: Move existing agent scripts to fast-agent.
  description: A practical migration path for scheduled, CI, and application-driven agent work.
  alt: fast-agent social card — Migrate Your Automations
---

# Migrate Your Automations

Move an existing agent script, scheduled task, or CI workflow to **fast-agent**
without rewriting everything at once. Start with the same prompt and model,
establish a machine-readable output contract, then move tools and instructions
into reusable AgentCards when the workflow is stable.

The smallest useful migration is a one-shot command:

```bash
fast-agent go \
  --no-home \
  --model "responses.gpt-5.5?reasoning=high" \
  --message "Summarize the incidents opened in the last 24 hours."
```

`--no-home` keeps automation isolated from user-level fast-agent files. It is a
good default for CI, containers, and scheduled jobs where all inputs should be
explicit.

## 1. Inventory the existing contract

Before changing runtimes, record what the current automation depends on:

- The user prompt and system instructions
- Model and reasoning settings
- Files, URLs, and environment variables used as input
- Tools or remote services the agent can call
- Expected stdout, files, or API payloads
- Timeout, retry, and scheduling behavior
- Secrets supplied by CI or the host environment

Keep those boundaries stable during the first migration. Change the harness
before changing the workflow.

## 2. Move prompts out of shell quoting

Inline `--message` input is convenient for short jobs. For longer instructions,
store the prompt in a versioned file:

```text title="prompts/daily-review.md"
Review the attached operational report.

Identify:
- customer-impacting incidents
- unresolved actions
- owners and due dates

Return only the requested structured result.
```

Run it with:

```bash
fast-agent go \
  --no-home \
  --model "responses.gpt-5.5?reasoning=high" \
  --prompt-file prompts/daily-review.md \
  --attach reports/latest.pdf
```

`--attach` accepts local files and HTTP(S) URLs and can be repeated.

## 3. Make stdout a stable API

Do not parse prose in production automation. Define a JSON Schema for the
result your next step expects:

```json title="schemas/incident-review.json"
{
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "incidents": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "owner": {"type": "string"},
          "due": {"type": ["string", "null"]}
        },
        "required": ["title", "owner", "due"],
        "additionalProperties": false
      }
    }
  },
  "required": ["summary", "incidents"],
  "additionalProperties": false
}
```

Then request machine-readable output:

```bash
fast-agent go \
  --no-home \
  --model "responses.gpt-5.5?reasoning=high" \
  --prompt-file prompts/daily-review.md \
  --attach reports/latest.pdf \
  --json-schema schemas/incident-review.json \
  > build/incident-review.json
```

In structured one-shot mode, stdout contains the validated JSON document while
diagnostics go to stderr. See [Structured Outputs](structured-outputs.md) for
provider behavior and tool policies.

## 4. Move tools into explicit configuration

If the existing automation calls MCP servers, pass them explicitly while
prototyping:

```bash
fast-agent go \
  --no-home \
  --url https://example.com/mcp \
  --auth "$EXAMPLE_MCP_TOKEN" \
  --prompt-file prompts/daily-review.md \
  --json-schema schemas/incident-review.json
```

For repeatable workflows, move server definitions, instructions, model
selection, and tool policy into an
[AgentCard](../agents/defining/agent_cards.md). Run one named agent with:

```bash
fast-agent go \
  --no-home \
  --agent-cards ./agents \
  --agent incident-review \
  --prompt-file prompts/daily-review.md \
  --json-schema schemas/incident-review.json
```

This keeps shell scripts focused on orchestration while the agent definition
remains reusable from the CLI, Python, ACP, and other harness surfaces.

## 5. Supply secrets at the boundary

Keep credentials in the scheduler, CI secret store, or deployment environment.
Do not commit tokens to prompt files, AgentCards, or `fast-agent.yaml`.

```bash
export OPENAI_API_KEY="..."
export EXAMPLE_MCP_TOKEN="..."
fast-agent go --no-home --agent-cards ./agents --agent incident-review \
  --message "Run the daily review"
```

Configuration values can reference environment variables with `${NAME}` where
supported. See the [configuration reference](../ref/config_file.md) for the
available settings.

## 6. Add CI or scheduler controls

Keep the host responsible for scheduling and process-level policy. Use
fast-agent for the agent turn, tools, structured output, and trajectory.

```bash title="scripts/daily-review.sh"
#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

fast-agent go \
  --no-home \
  --timeout 900 \
  --agent-cards ./agents \
  --agent incident-review \
  --prompt-file prompts/daily-review.md \
  --attach reports/latest.pdf \
  --json-schema schemas/incident-review.json \
  --trajectory-output build/incident-review.atif.json \
  > build/incident-review.json
```

The same script can run from cron, a systemd timer, a container job, or a CI
runner. Pin the fast-agent version in production and update it deliberately.

## 7. Migrate in stages

Use a small set of representative inputs before switching production traffic:

1. Run the old and new automation against the same inputs.
2. Compare structured fields rather than prose formatting.
3. Check tool permissions and external side effects.
4. Confirm timeout and failure behavior in the host environment.
5. Record cost, token usage, and output quality.
6. Keep the old path available for rollback during the first scheduled runs.

For row-oriented workloads, use [Batch Processing](batch-processing.md) instead
of wrapping hundreds of one-shot commands in a shell loop. For the complete
one-shot command surface, see [`fast-agent go`](../ref/go_command.md).
