---
title: A2A Client
description: Use fast-agent as a client for remote Agent2Agent (A2A) agents.
---

# A2A Client

fast-agent can connect to remote A2A agents as normal fast-agent agents. A
remote A2A agent can be used from the CLI, TUI, AgentCards, or the Python API.

## CLI

Use `--a2a` for an ad hoc remote agent:

```bash
uv run fast-agent -x \
  --a2a http://127.0.0.1:41242 \
  --a2a-transport JSONRPC \
  --message "hello"
```

`--a2a` points at the remote agent base URL. fast-agent resolves the AgentCard
from `/.well-known/agent-card.json`, selects a supported transport, and sends the
message through the A2A SDK client.

When no transport is specified, fast-agent asks the SDK to use either supported
HTTP binding: `JSONRPC` or `HTTP+JSON`. Set `--a2a-transport` only when you want
to force one binding.

Supported HTTP transports:

| Canonical | Useful aliases |
|---|---|
| `JSONRPC` | `jsonrpc`, `json-rpc`, `rpc` |
| `HTTP+JSON` | `http`, `http+json`, `rest` |

gRPC is not part of fast-agent's A2A support target.

## AgentCard

Use a checked-in AgentCard when the remote A2A agent should be reusable:

```yaml
type: a2a
name: research_remote
url: https://research.example.com
transport: JSONRPC
```

Then run:

```bash
uv run fast-agent -x --agent-cards ./agents --agent research_remote
```

A2A cards also support:

```yaml
streaming: true
polling: false
accepted_output_modes:
  - text
  - image
headers:
  Authorization: "Bearer ${A2A_TOKEN}"
relative_card_path: "/.well-known/agent-card.json"
```

## TUI

Inside the interactive prompt, connect a remote A2A agent at runtime:

```text
/a2a connect http://127.0.0.1:41242 --transport JSONRPC --name research_remote
```

Useful diagnostics:

```text
/a2a list
/a2a status [agent]
/a2a card [agent]
/a2a transport [agent]
/a2a reset [agent]
```

`/a2a status` shows the current A2A `context_id`, pending `task_id`, last task
state, and selected client transport.

## Streaming

Remote A2A `TaskArtifactUpdateEvent` updates are emitted through the normal
fast-agent stream listener path. The client assembles final text per artifact
and honors the A2A `append` flag, so replacement updates replace the artifact
content and append updates extend it.

## `INPUT_REQUIRED`

When a remote A2A task reaches `TASK_STATE_INPUT_REQUIRED`, fast-agent:

- returns a normal `PromptMessageExtended` assistant turn with
  `stop_reason=LlmStopReason.PAUSE`;
- keeps the pending A2A `task_id`;
- preserves the returned A2A `context_id`;
- sends the next user message back to the same task.

Use `/a2a reset` to clear the pending task and start a fresh remote context.

## Attachments

The A2A client maps fast-agent prompt content to A2A parts:

| fast-agent content | A2A part |
|---|---|
| `TextContent` | `Part(text=...)` |
| `ResourceLink` | `Part(url=...)` |
| `ImageContent` | `Part(raw=..., mediaType=image/...)` |
| `AudioContent` | `Part(raw=..., mediaType=audio/...)` |

Remote URL, data, raw, and text response parts are rendered into fast-agent
assistant output. See [Protocol Compliance](protocol-compliance.md) for current
partial multimodal gaps.
