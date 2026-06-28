---
title: A2A Client
description: Use fast-agent as a client for remote Agent2Agent (A2A) agents.
---

# A2A Client

fast-agent can connect to remote A2A agents as normal fast-agent agents. A
remote A2A agent can be used from the CLI, TUI, AgentCards, or the Python API.
For direct Python construction, see [A2A API](api.md#client-api).

For a repeatable local target, start the deterministic fake A2A server from the
fast-agent repository root:

```bash
--8<-- "docs/docs/a2a/snippets/start-fake-server.sh"
```

It exposes:

| Endpoint | URL |
|---|---|
| AgentCard | `http://127.0.0.1:41242/.well-known/agent-card.json` |
| JSON-RPC | `http://127.0.0.1:41242/a2a/jsonrpc` |
| HTTP+JSON | `http://127.0.0.1:41242/a2a/rest` |

## CLI

Use `--a2a` for an ad hoc remote agent:

```bash
--8<-- "docs/docs/a2a/snippets/cli-hello-command.sh"
```

Expected output:

```text
--8<-- "docs/docs/a2a/snippets/cli-hello-output.txt"
```

`--a2a` points at the remote agent base URL. fast-agent resolves the AgentCard
from `/.well-known/agent-card.json`, selects a supported transport, and sends the
message through the A2A SDK client.

When no transport is specified, fast-agent asks the SDK to use either supported
HTTP binding: `JSONRPC` or `HTTP+JSON`. Set `--a2a-transport` only when you want
to force one binding.

Use `--a2a-oauth` or `--no-a2a-oauth` to force or disable browser OAuth for an
ad hoc remote agent:

```bash
uv run fast-agent -x \
  --a2a https://research.example.com \
  --a2a-oauth \
  --message "hello"
```

Use `--auth` when the remote A2A endpoint itself expects bearer auth. This uses
the standard `Authorization` header, including for Hugging Face Space endpoints:

```bash
uv run fast-agent -x \
  --a2a https://agent-demo.hf.space \
  --auth "$HF_TOKEN" \
  --message "hello"
```

Supported HTTP transports:

| Canonical | Useful aliases |
|---|---|
| `JSONRPC` | `jsonrpc`, `json-rpc`, `rpc` |
| `HTTP+JSON` | `http`, `http+json`, `rest` |

fast-agent does not support gRPC for A2A.

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
  - text/plain
  - application/json
  - image/*
request_timeout_seconds: 120
headers:
  Authorization: "Bearer ${A2A_TOKEN}"
auth:
  oauth: true
  persist: keyring
relative_card_path: "/.well-known/agent-card.json"
```

For Hugging Face URLs (`hf.co`, `huggingface.co`, and `*.hf.space`),
fast-agent has two different auth policies:

- Ambient Hugging Face auth discovers `HF_TOKEN` or the local Hub login and adds
  it only to Hugging Face URLs. It uses `Authorization` for `hf.co` and
  `huggingface.co`, and `X-HF-Authorization` for `*.hf.space`. This is intended
  for ordinary HF MCP calls and Space apps that consume the caller's HF token
  without taking over app-level `Authorization`.
- Explicit endpoint auth uses `Authorization: Bearer ...`, including for
  `*.hf.space`. This is the policy behind `--auth`, checked-in `headers:
  Authorization: ...`, and OAuth-managed A2A/MCP servers.

For `*.hf.space` A2A URLs, fast-agent first fetches the public AgentCard. If the
card advertises HTTP bearer security and no explicit headers were configured,
the client treats the Space as a protected endpoint: a discovered local
`HF_TOKEN`/Hub login is sent as `Authorization`, not `X-HF-Authorization`. If no
local token is available and OAuth is allowed, the client uses the OAuth flow.

When a remote AgentCard advertises OAuth2 or OpenID Connect security schemes,
fast-agent can reuse the existing browser OAuth flow. If `auth` is omitted, the
A2A client enables that flow only for OAuth/OIDC cards. Set `auth.oauth: false`
to disable browser OAuth, or `auth.oauth: true` to allow OAuth challenge handling
even before the card requires it. The flow uses the same local callback,
paste-URL fallback, client metadata URL, and keyring storage behavior as MCP URL
connections.

## TUI

Inside the interactive prompt, connect a remote A2A agent at runtime:

```text
/a2a connect http://127.0.0.1:41242 --transport JSONRPC --name research_remote
```

Use `--oauth` or `--no-oauth` to force or disable browser OAuth for a runtime
connection:

```text
/a2a connect https://research.example.com --oauth --name research_remote
```

Useful diagnostics:

```text
/a2a list
/a2a status [agent]
/tasks [agent]
/a2a card [agent]
/a2a transport [agent]
/a2a reset [agent]
```

After an A2A turn starts a remote context, fast-agent shows a compact status
line above the input:

```text
(a2a) - Context ID: 7b7c...8d9e. Tasks: 4 finished, 3 pending. /tasks for info
```

`/tasks` shows the current A2A `context_id`, pending `task_id`, last event type,
last task state, finished/pending task counts, outstanding task ids, and selected
client transport. `/a2a status` remains available for the same connection
diagnostics.

When the local A2A AgentCard or request has `use_history: false`, fast-agent
starts each completed task turn with a fresh A2A context. The exceptions are
standalone A2A `Message` responses and `TASK_STATE_INPUT_REQUIRED`: fast-agent
keeps the returned `context_id` after a refinement message, and keeps both the
`task_id` and `context_id` when a task is waiting for input.

For research-style A2A agents, this means a refinement reply can return a plain
A2A `Message` first. The TUI still shows the context id, and the next user turn
is sent in that context. Once the server starts a research task, `/tasks` lists
any outstanding task ids for monitoring.

## Task Updates

Remote A2A message events are emitted through the normal fast-agent stream
listener path. Task artifact updates are assembled into the returned
`PromptMessageExtended`; they are not exposed as live assistant-message chunk
streaming. The client assembles final text per artifact and honors the A2A
`append` flag, so replacement updates replace the artifact content and append
updates extend it.

The A2A client defaults to a longer HTTP request timeout than httpx's default so
real LLM-backed servers have time to emit the first stream event. Set
`request_timeout_seconds` on an A2A AgentCard when a remote endpoint needs a
different timeout.

## `INPUT_REQUIRED`

When a remote A2A task reaches `TASK_STATE_INPUT_REQUIRED`, fast-agent:

- returns a normal `PromptMessageExtended` assistant turn with
  `stop_reason=LlmStopReason.PAUSE`;
- keeps the pending A2A `task_id`;
- preserves the returned A2A `context_id`;
- sends the next user message back to the same task.

Use `/a2a reset` to clear the pending task and start a fresh remote context.

With the fake server, type this in the TUI:

```text
need input
blue
```

The first turn receives:

```text
A2A task TASK_STATE_INPUT_REQUIRED: Please provide the missing value.
```

The second turn is sent with the pending A2A task id and completes the task:

```text
input received: blue
```

Use `/a2a status` between those turns to inspect the preserved `Context`, `Task`,
and `Last state` fields.

### Turn Continuation Recording

This recording shows the task id being retained only while the remote task is in
`TASK_STATE_INPUT_REQUIRED`; after the follow-up completes, the task id is
cleared and the context id remains available for future turns.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/a2a/a2a-client-input-required.cast"
  data-fa-asciinema-cols="96"
  data-fa-asciinema-rows="18"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1"
  data-fa-asciinema-fit="width"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

## Attachments

The A2A client maps fast-agent prompt content to A2A parts:

| fast-agent content | A2A part |
|---|---|
| `TextContent` | `Part(text=...)` |
| `ResourceLink` | `Part(url=...)` |
| `ImageContent` | `Part(raw=..., mediaType=image/...)` |
| `AudioContent` | `Part(raw=..., mediaType=audio/...)` |
| `EmbeddedResource` with JSON `TextResourceContents` | `Part(data=...)` |

Remote URL, data, raw, and text response parts are rendered into fast-agent
assistant output. See [Protocol Compliance](protocol-compliance.md) for
content-mapping details.

The fake server can return non-text parts:

```bash
--8<-- "docs/docs/a2a/snippets/cli-files-command.sh"
```

Expected output:

````text
--8<-- "docs/docs/a2a/snippets/cli-files-output.txt"
````
