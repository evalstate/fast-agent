---
title: A2A API
description: Use fast-agent A2A client and server support from Python and raw A2A HTTP APIs.
---

# A2A API

The fast-agent A2A integration is designed to feel like working with normal
fast-agent agents. The local API surface uses `PromptMessageExtended`, stream
listeners, and normal fast-agent history behavior.

For operational guides, start with [A2A Client](client.md) or
[A2A Server](server.md). This page focuses on direct Python APIs and raw A2A
HTTP request shapes.

## Client API

Create an `A2ARemoteAgent` directly when you want a remote A2A server behind the
fast-agent `AgentProtocol` interface:

```python
from mcp.types import TextContent

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import PromptMessageExtended

agent = A2ARemoteAgent(
    config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=True),
    a2a_config=A2AAgentConfig(
        url="http://127.0.0.1:41242",
        transport="JSONRPC",
    ),
)

await agent.initialize()
try:
    response = await agent.generate_impl(
        [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="hello")],
            )
        ]
    )
    print(response.all_text())
finally:
    await agent.shutdown()
```

`A2AAgentConfig` supports:

```python
A2AAgentConfig(
    url="https://agent.example.com",
    transport="JSONRPC",
    streaming=True,
    polling=False,
    accepted_output_modes=["text/plain", "application/json", "image/*"],
    headers={"Authorization": "Bearer ..."},
    relative_card_path="/.well-known/agent-card.json",
    request_timeout_seconds=120,
)
```

## Client Streaming API

Register a normal fast-agent stream listener before calling `generate_impl`:

```python
chunks: list[str] = []

remove_listener = agent.add_stream_listener(lambda chunk: chunks.append(chunk.text))
try:
    response = await agent.generate_impl([message])
finally:
    remove_listener()
```

For A2A streaming, `chunk.text` contains text from message events. Task artifact
updates are assembled into the returned `PromptMessageExtended` but are not
emitted as live assistant-message chunks.

## Client `INPUT_REQUIRED`

When the remote server returns `TASK_STATE_INPUT_REQUIRED`, the response has:

```python
response.stop_reason == LlmStopReason.PAUSE
```

The same `A2ARemoteAgent` instance keeps the pending remote task id. The next
`generate_impl` call sends the follow-up message to that task:

```python
first = await agent.generate_impl([Prompt.user("need input")])
assert first.stop_reason == LlmStopReason.PAUSE

second = await agent.generate_impl([Prompt.user("blue")])
```

Use `agent.reset_a2a_state()` to clear the pending task and start a new remote
context.

## Server API

Most deployments should use:

```bash
uv run fast-agent serve a2a --agent-cards ./agents
```

If you are embedding the server in Python, use `AgentA2AServer` with an existing
fast-agent `AgentInstance` factory:

```python
from fast_agent.a2a.server import AgentA2AServer

server = AgentA2AServer(
    primary_instance=bootstrap_instance,
    create_instance=create_instance,
    dispose_instance=dispose_instance,
    server_name="research agents",
    host="127.0.0.1",
    port=41241,
    instance_scope="connection",
)

app = server.asgi_app()
```

`instance_scope` accepts the same values as `fast-agent serve`:

| Scope | Server API behavior |
|---|---|
| `shared` | Reuse `primary_instance` for all A2A messages. |
| `connection` | Call `create_instance` for each new A2A `context_id` and reuse that instance for later messages in the same context. |
| `request` | Call `create_instance` and `dispose_instance` for each A2A message. |

Each served agent's `use_history` setting still controls whether prior turns are
included in model calls inside the selected instance scope.

When the standard server path is sufficient, prefer the CLI or
`fast.start_server(transport="a2a")`; they initialize the normal fast-agent
runtime and load AgentCards from the active fast-agent home before serving.

### Server Task API

Code running inside a fast-agent A2A request can return a standalone A2A
message, or publish task status and artifact updates directly:

```python
from fast_agent.a2a.task_api import return_artifact, return_message, start_task

if needs_refinement:
    await return_message("Please clarify the research goal and desired output format.")
    return

handle = await start_task("Searching source documents")
await return_artifact(
    "Found the first batch of notes.",
    name="notes",
    artifact_id=f"{handle.task_id}:notes",
    append=False,
    last_chunk=False,
)
```

The same helpers are exposed to tool-capable fast-agent agents served over A2A
as model tools named `return_message`, `start_task`, and `return_artifact`. This
lets an agent developer either call the Python API directly from server code, or
let the model choose between a refinement `Message` and a started research
`Task` through ordinary fast-agent tool calls.

For ordinary fast-agent agents, the A2A server preserves the existing behavior
of starting a task before model work begins. Tool-capable agents defer task
creation so they can first return a standalone `Message`; once they call
`start_task`, return an artifact, or stream output, fast-agent emits the standard
A2A task/status/artifact events.

If your A2A server program is already using `AgentHarness`, the harness exposes
the same request-local helpers:

```python
async with fast.harness() as harness:
    if needs_refinement:
        await harness.return_message("Please make the research question more specific.")
        return

    handle = await harness.start_task("Building report")
    await harness.return_artifact(
        "Draft ready.",
        artifact_id=f"{handle.task_id}:draft",
        name="draft",
    )
```

For A2A-native servers that need explicit protocol routing, use the Harness API
as the adapter boundary: convert the A2A request into an `AgentRequest`, call
`harness.invoke(...)`, and emit A2A `Message`, `Task`, status, and artifact
events. The research example in `examples/a2a/research/server.py` uses this
shape. Its agents are AgentCards under
`examples/a2a/research/.fast-agent/agent-cards/`, loaded automatically when
`fast.harness()` starts.

## Raw A2A JSON-RPC

External clients can call the served fast-agent endpoint directly:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendStreamingMessage",
  "params": {
    "message": {
      "role": "ROLE_USER",
      "messageId": "msg-1",
      "parts": [
        {"text": "hello"}
      ]
    }
  }
}
```

The endpoint behaves as one A2A remote agent. Requests route to the fast-agent
default agent, which can orchestrate or delegate internally. fast-agent servers
also accept a non-portable metadata routing extension, such as
`{"agent": "researcher"}`, for fast-agent-to-fast-agent integrations.

To continue a session, include the returned `contextId`. To continue an
`INPUT_REQUIRED` task, include both the returned `contextId` and `taskId`.

## Raw HTTP+JSON

The REST binding is exposed under `/a2a/rest`. For example:

```http
POST /a2a/rest/message:stream HTTP/1.1
Content-Type: application/json

{
  "message": {
    "role": "ROLE_USER",
    "messageId": "msg-1",
    "parts": [{"text": "hello"}]
  }
}
```

Responses are A2A stream response objects containing exactly one of `task`,
`message`, `statusUpdate`, or `artifactUpdate`.

## Content Mapping

Inbound A2A parts are converted to fast-agent prompt content:

| A2A part | fast-agent prompt content |
|---|---|
| `text` | `TextContent` |
| `url` | `ResourceLink` when valid, otherwise Markdown link text |
| `raw` image bytes | `ImageContent` |
| `raw` non-image bytes | `EmbeddedResource` with `BlobResourceContents` |
| `data` | formatted JSON text |

fast-agent responses are converted back to A2A artifact parts using the content
types available in `PromptMessageExtended`.

For structured JSON, A2A supports JSON-compatible `data` parts and also permits
JSON returned as text artifacts. fast-agent keeps model text as text, but maps an
`EmbeddedResource` containing `TextResourceContents` with
`mimeType="application/json"` to an A2A `data` part:

```python
from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

PromptMessageExtended(
    role="assistant",
    content=[
        EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=AnyUrl("resource:///tickets.json"),
                mimeType="application/json",
                text='{"tickets": [{"id": "REQ123", "status": "open"}]}',
            ),
        )
    ],
)
```
