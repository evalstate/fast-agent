---
title: A2A Server
description: Deploy fast-agent agents as an Agent2Agent (A2A) HTTP server.
---

# A2A Server

Use `fast-agent serve --transport a2a` to expose fast-agent agents through A2A
HTTP transports. This follows the same deployment shape as ACP/MCP serving: the
configured fast-agent app is initialized first, then the A2A server routes
incoming protocol messages into the selected agent.

## Start a Server

```bash
uv run fast-agent serve \
  --transport a2a \
  --host 127.0.0.1 \
  --port 41241 \
  --agent-cards ./agents \
  --model codexresponses.gpt-5.4-mini
```

The server exposes:

| Endpoint | URL |
|---|---|
| AgentCard | `http://127.0.0.1:41241/.well-known/agent-card.json` |
| JSON-RPC | `http://127.0.0.1:41241/a2a/jsonrpc` |
| HTTP+JSON | `http://127.0.0.1:41241/a2a/rest` |

The AgentCard advertises `JSONRPC` and `HTTP+JSON` with protocol version `1.0`.
gRPC is intentionally not advertised.

When the server binds to a wildcard host such as `0.0.0.0` or `::`, the served
AgentCard builds interface URLs from the incoming AgentCard request host. This
keeps cards fetched from another machine routable to the server instead of
advertising the bind wildcard or the server's loopback address.

## Card Recording

This recording shows the expected shape when a wildcard-bound server is fetched
through a routable hostname. The JSON-RPC and HTTP+JSON interfaces use the
request hostname in the served card.

<div class="a2a-terminal-demo">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/asciinema-player.css">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/catppuccin.css">
  <div id="a2a-server-card-player"></div>
</div>

<script src="../../assets/vendor/asciinema-player/asciinema-player.min.js"></script>
<script>
  (function () {
    function renderServerCardCast() {
      var target = document.getElementById("a2a-server-card-player");
      if (!target || !window.AsciinemaPlayer || target.dataset.loaded === "true") {
        return;
      }
      target.dataset.loaded = "true";
      window.AsciinemaPlayer.create("../../assets/a2a/a2a-server-card.cast", target, {
        cols: 104,
        rows: 20,
        preload: true,
        speed: 1,
        idleTimeLimit: 1,
        fit: "width",
        theme: "fast-agent-dark"
      });
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", renderServerCardCast);
    } else {
      renderServerCardCast();
    }
    if (window.document$ && window.document$.subscribe) {
      window.document$.subscribe(renderServerCardCast);
    }
  })();
</script>

For static deployment checks, bind with the concrete hostname or address that
remote clients should use.

## Runtime Wiring

The served agents use the normal fast-agent runtime. AgentCards, MCP servers,
tools, skills, hooks, model settings, and workflow agents are loaded through the
same path used by the CLI and TUI before the A2A server starts.

That means an A2A request can drive a full fast-agent bundle: an orchestrator,
router, tool-using agent, MCP-backed agent, or AgentCard-loaded group.

## Agent Skills in the A2A Card

A2A `AgentSkill` is the protocol's advertised capability object. It is separate
from fast-agent "skills" on disk.

fast-agent currently exposes one A2A `AgentSkill` for each loaded fast-agent
agent:

```json
{
  "id": "researcher",
  "name": "researcher",
  "description": "Research and summarize source material.",
  "tags": ["fast-agent", "basic"],
  "examples": ["Hello"],
  "inputModes": ["text", "file", "image"],
  "outputModes": ["text", "file", "image", "task-status"]
}
```

The generated skill list comes from `primary_instance.agents` at server startup.
The skill `id` and `name` are the fast-agent agent name. The description uses the
agent's configured `description` when present, otherwise fast-agent generates a
generic description. Tags include `fast-agent` and the fast-agent agent type.

By default, messages route to the fast-agent default agent. A2A clients can
target a specific loaded agent with message metadata:

```json
{
  "metadata": {
    "agent": "researcher"
  }
}
```

`fast_agent_agent` is accepted as an equivalent metadata key.

Current limitation: examples and mode lists are still generic. They do not yet
derive richer examples or per-agent modality declarations from fast-agent
AgentCard metadata or from installed fast-agent skills.

## Sessions and Resumption

A2A `contextId` is optional on inbound messages. If a client omits it, the A2A
SDK generates one. fast-agent uses the resolved `context_id` as the server-side
session key:

- same `context_id`: reuse the same fast-agent instance and message history;
- new `context_id`: create a fresh fast-agent instance;
- same interrupted `task_id` and `context_id`: continue an `INPUT_REQUIRED`
  task.

The `context_id` selects the server-side fast-agent instance. The agent's
history setting controls how much prior conversation is sent to the model; it
does not change the A2A session key.

The current server uses in-memory A2A task storage and in-memory fast-agent
context instances. Restarting the process loses A2A task state and session
continuity.

## Streaming

The server registers a fast-agent stream listener for agents that support it.
Non-reasoning text chunks are sent as A2A `TaskArtifactUpdateEvent` updates with
a stable artifact id. The first chunk replaces/creates the artifact; later chunks
use A2A append semantics.

If the final fast-agent response differs from the streamed text, the server sends
a final replacement artifact for the same artifact id before completing the task.

## `INPUT_REQUIRED`

When a fast-agent response has:

```python
stop_reason=LlmStopReason.PAUSE
```

the A2A server reports `TASK_STATE_INPUT_REQUIRED` with the response text as the
status message. The task remains resumable. Clients should send the follow-up
message with the same A2A `task_id` and `context_id`.

## Errors

The server maps common fast-agent outcomes into A2A states:

| fast-agent outcome | A2A state |
|---|---|
| normal response | `TASK_STATE_COMPLETED` |
| `LlmStopReason.PAUSE` | `TASK_STATE_INPUT_REQUIRED` |
| provider credential error | `TASK_STATE_AUTH_REQUIRED` |
| cancellation | `TASK_STATE_CANCELED` |
| unexpected exception | `TASK_STATE_FAILED` |

Transport validation errors, task lookup errors, non-cancelable tasks, and
unsupported push notification operations are handled by the A2A SDK request
handler.

## File Parts

Incoming raw image parts become `ImageContent`. Other raw file parts become
`EmbeddedResource` values with `BlobResourceContents`, preserving the base64 file
payload, MIME type, and filename-like attachment URI for the fast-agent agent.
When a fast-agent response includes a blob resource, the server emits it back to
A2A clients as a raw file part.

See [Protocol Compliance](protocol-compliance.md) for the full supported surface
and known gaps.
