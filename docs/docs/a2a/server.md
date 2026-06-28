---
title: A2A Server
description: Deploy fast-agent agents as an Agent2Agent (A2A) HTTP server.
---

# A2A Server

Use `fast-agent serve a2a` to expose a fast-agent app through A2A HTTP
transports. `fast-agent serve --transport a2a` remains supported for parity with
the generic MCP/ACP serve command. The configured fast-agent app is initialized
first, then the A2A server routes ordinary protocol messages into the
fast-agent default agent.

## Start a Server

```bash
uv run fast-agent serve a2a \
  --host 127.0.0.1 \
  --port 41241 \
  --instance-scope shared \
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

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/a2a/a2a-server-card.cast"
  data-fa-asciinema-cols="104"
  data-fa-asciinema-rows="20"
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

For static deployment checks, bind with the concrete hostname or address that
remote clients should use.

## Runtime Wiring

The served A2A agent uses the normal fast-agent runtime. AgentCards, MCP servers,
tools, skills, hooks, model settings, and workflow agents are loaded through the
same path used by the CLI and TUI before the A2A server starts.

That means an A2A request can drive a full fast-agent bundle behind one A2A
agent boundary: an orchestrator, router, tool-using agent, MCP-backed agent, or
AgentCard-loaded group.

AgentCards in the active fast-agent environment are loaded before the harness or
server runtime starts. For example, cards in `.fast-agent/agent-cards/` are
available to `fast.harness()` and to `fast-agent serve a2a` without an explicit
`fast.load_agents(...)` call. Use `fast.load_agents(path)` only when a Python
program intentionally loads cards from a non-environment location.

## Server Shapes

Choose the smallest server shape that fits the behavior you need:

| Shape | Use when |
|---|---|
| `fast-agent serve a2a` | You want to serve a normal fast-agent app over A2A. This is the default path for AgentCard-defined agents, MCP servers, workflows, and ordinary streaming. |
| Python wrapper with `fast.start_server(transport="a2a")` | You want a small script that defines agents in Python and starts the standard A2A server. See `examples/a2a/facts_server.py`. |
| Custom A2A executor plus `FastAgent.harness()` | You need protocol-specific behavior, such as returning a standalone A2A `Message` before creating a task, custom task progress, or explicit A2A SDK route wiring. See `examples/a2a/research/server.py` and `examples/a2a/server.py`. |

The custom executor shape should still use AgentCards for normal agent
definitions where possible. The Python A2A entrypoint should own protocol
routing, not duplicate agent definitions that belong in cards.

## Agent Skills in the A2A Card

A2A models the served endpoint as one remote agent or agentic system. A2A
`AgentSkill` entries are advertised capabilities for that remote agent; they are
not a standard routing table and do not make multiple directly addressable
agents at the same endpoint. A2A `AgentSkill` is also separate from fast-agent
"skills" on disk.

fast-agent advertises loaded user-facing fast-agent agents as A2A skills so
clients can understand the capabilities available behind the endpoint:

```json
{
  "id": "researcher",
  "name": "researcher",
  "description": "Research and summarize source material.",
  "tags": ["fast-agent", "basic"],
  "examples": ["Hello"],
  "inputModes": ["text/plain", "application/json", "application/octet-stream", "image/*"],
  "outputModes": ["text/plain", "application/json", "application/octet-stream", "image/*"]
}
```

The generated skill list comes from the user-facing fast-agent agents loaded at
server startup. The skill `id` and `name` are derived from the fast-agent agent
name. The description uses the agent's configured `description` when present,
otherwise fast-agent generates a generic description. Tags include `fast-agent`
and the fast-agent agent type.

Messages route to the fast-agent default agent by default. That default agent
should orchestrate, delegate, or call helper agents internally just as it would
in normal fast-agent use.

For fast-agent-to-fast-agent integrations, the server also accepts a
fast-agent-specific routing extension in message metadata:

```json
{
  "metadata": {
    "agent": "researcher"
  }
}
```

`fast_agent_agent` is accepted as an equivalent metadata key. This metadata is
not portable A2A behavior; generic A2A clients should treat skills as capability
metadata and send normal messages to the endpoint.

AgentCard examples are generic. Mode lists describe server-wide MIME-style
content support rather than deriving per-agent modality declarations from
fast-agent AgentCard metadata or installed fast-agent skills.

## Instance Scope, Sessions, and Resumption

A2A `contextId` is optional on inbound messages. If a client omits it, the A2A
SDK generates one. The server still returns and tracks the resolved A2A
`context_id`; how that maps to fast-agent runtime instances depends on
`--instance-scope`:

| Scope | Behavior |
|---|---|
| `shared` | Use the primary fast-agent instance for all A2A contexts. This is the default for `fast-agent serve a2a`, matching the generic serve default. |
| `connection` | Use the A2A `context_id` as the server-side instance/session key. The same `context_id` reuses the same fast-agent instance; a new `context_id` creates a fresh instance. |
| `request` | Create and dispose a fresh fast-agent instance for every A2A message. |

The served agent's history setting controls how much prior conversation is sent
to the model inside whichever instance scope is selected. It does not change the
A2A protocol `context_id`.

For `INPUT_REQUIRED`, clients should continue with the returned `task_id` and
`context_id`. `shared` and `connection` scopes preserve in-memory fast-agent
state for follow-up turns. `request` scope intentionally creates a fresh
fast-agent instance for each message, so it is best for stateless agents.

The server uses in-memory A2A task storage and in-memory fast-agent context
instances. Restarting the process loses A2A task state and session continuity.

Custom Harness-backed A2A servers choose their own session mapping. The research
example maps A2A `context_id` directly to harness `session_id`, so a refinement
reply and the later research task share one fast-agent session. In that example,
session files are stored under
`examples/a2a/research/.fast-agent/sessions/`; both research AgentCards use
`use_history: false`, so the session identity is stable without accumulating
model chat history.

## Streaming

The server registers a fast-agent stream listener for agents that support it.
Non-reasoning text chunks are sent as A2A `TaskArtifactUpdateEvent` updates with
a stable artifact id. The first chunk replaces/creates the artifact; later chunks
use A2A append semantics.

If the final fast-agent response differs from the streamed text, the server sends
a final replacement artifact for the same artifact id before completing the task.

## Refinement Messages

A2A-native server code can return a standalone A2A `Message` before starting a
task. This is useful for research-intake agents that first refine a vague user
message into a concrete research goal.

```python
from fast_agent.a2a.task_api import return_message, start_task

if needs_refinement:
    await return_message("Please clarify the scope and audience for the research.")
    return

await start_task("Research task accepted")
```

Tool-capable fast-agent agents served over A2A get `return_message`,
`start_task`, and `return_artifact` as local tools. For those agents, fast-agent
defers task creation until the agent starts work. Ordinary served agents keep
the default task-generating behavior.

For custom A2A executors, prefer the Harness API shape used by the MCP adapter:
translate the incoming protocol request into an `AgentRequest`, invoke an agent
through `FastAgent.harness()`, then translate the result back into A2A events.
The research example follows this pattern:

```python
async with fast.harness() as harness:
    request_handler = DefaultRequestHandler(
        agent_executor=ResearchA2AExecutor(ResearchA2AHarnessAdapter(harness)),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card(host=HOST, port=PORT),
    )
```

Inside the adapter, the A2A `context_id` is used as the harness `session_id` so
the refinement message and the eventual research task share a session:

```python
response = await harness.invoke(
    AgentRequest.text(
        prompt,
        agent="research_refiner",
        session_id=context.context_id,
        metadata={"transport": "a2a", "phase": "research_refinement"},
    )
)
```

The example lives at `examples/a2a/research/server.py`. Its refiner and worker
are normal AgentCards in
`examples/a2a/research/.fast-agent/agent-cards/`, loaded automatically when
`fast.harness()` starts. The A2A executor is responsible only for the
protocol-level decision: return a standalone `Message` for refinement guidance,
or create a `Task` and stream progress artifacts.

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

## Hugging Face Bearer Auth

Set `FAST_AGENT_SERVE_OAUTH=huggingface` before starting `fast-agent serve a2a`
to require bearer authentication on `/a2a/jsonrpc` and `/a2a/rest` while keeping
the public AgentCard discoverable.

The A2A server middleware accepts both header names when they reach the app:

```text
Authorization: Bearer <token>
X-HF-Authorization: Bearer <token>
```

The bearer token is validated against Hugging Face before any A2A task reaches
fast-agent agents or tools. A non-empty placeholder such as `Bearer test` is
rejected; it is not treated as authentication.

For Space-hosted A2A endpoints, clients should use `Authorization` through
`--auth`, explicit AgentCard headers, or OAuth. `X-HF-Authorization` is the
ambient fast-agent CLI policy for ordinary Space app calls; it is not a
substitute for endpoint bearer auth unless the deployment ingress passes that
header through to the app. The server advertises an `hf_bearer` HTTP bearer
security scheme in the AgentCard and stores the inbound token in fast-agent
request context while the agent runs, allowing Hugging Face Inference Provider
models and Hugging Face MCP/tools to use the caller credential.

If `FAST_AGENT_SERVE_OAUTH` is not set, the server does not gate inbound A2A
requests and any `HF_TOKEN` configured in the Space environment is used as the
server's own Hugging Face credential. Use that only for trusted/private
deployments or with tightly scoped service tokens.

See [Host A2A on Hugging Face](host-on-hf.md) for a Space-oriented setup.

## File Parts

Incoming raw image parts become `ImageContent`. Other raw file parts become
`EmbeddedResource` values with `BlobResourceContents`, preserving the base64 file
payload, MIME type, and filename-like attachment URI for the fast-agent agent.
When a fast-agent response includes a blob resource, the server emits it back to
A2A clients as a raw file part.

## Structured JSON

A2A supports structured JSON exchange through JSON-compatible data content and
also allows JSON to be returned as text artifacts. fast-agent does not parse
ordinary model text and guess that it should become protocol data. Instead, it
maps `TextResourceContents` with `mimeType="application/json"` to A2A data
parts. This gives API users and structured-output wrappers an explicit path to
return protocol-level JSON while preserving normal markdown/text responses.

See [Protocol Compliance](protocol-compliance.md) for the full supported surface
and limitations.
