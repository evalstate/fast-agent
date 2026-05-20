# A2A Protocol Compliance

fast-agent's A2A support targets the
[A2A Protocol Specification 1.0](https://a2a-protocol.org/v1.0.0/specification/)
for HTTP transports. The implementation is built on the `a2a-sdk` 1.0 server and
client stack and intentionally excludes gRPC.

## Supported

| Area | Status | Notes |
|---|---|---|
| Agent discovery | Supported | `fast-agent serve a2a` serves an AgentCard at `/.well-known/agent-card.json`. The card declares `JSONRPC` and `HTTP+JSON` interfaces with protocol version `1.0`, and advertises MIME-style input/output modes such as `text/plain`, `application/json`, `application/octet-stream`, and `image/*`. `fast-agent serve --transport a2a` remains supported. |
| JSON-RPC transport | Supported | Client and server use the SDK JSON-RPC binding. |
| HTTP+JSON transport | Supported | Client and server use the SDK REST binding. The server exposes the REST binding under `/a2a/rest`. |
| Streaming task updates | Supported | fast-agent stream listeners are bridged to A2A `TaskArtifactUpdateEvent` events. The client preserves artifact order and honors the A2A `append` flag. |
| Multi-turn contexts | Supported | Inbound `contextId` is optional. The SDK generates one when omitted. In A2A server `connection` scope, fast-agent uses the resolved `context_id` as the server-side instance key; `shared` and `request` scopes intentionally reuse or recreate instances independently of the A2A context. |
| `INPUT_REQUIRED` continuation | Supported | Server responses with `PromptMessageExtended.stop_reason == LlmStopReason.PAUSE` become `TASK_STATE_INPUT_REQUIRED`. The fast-agent A2A client preserves the pending `task_id` and returned `context_id`, and surfaces the local response with `LlmStopReason.PAUSE`. |
| Task retrieval, listing, cancellation, and subscribe | SDK-backed | These operations are provided by the SDK request handler and in-memory task store. Cancellation also cancels the running fast-agent task when still active. |
| Text parts | Supported | A2A text parts map to `TextContent`; fast-agent text output maps back to A2A text parts. |
| URL parts | Supported | A2A URL parts map to `ResourceLink`; fast-agent resource links map back to A2A URL parts. |
| Image raw parts | Supported | Raw image bytes map to `ImageContent`; image output maps back to A2A raw parts. |
| Binary non-image raw parts | Supported | Inbound raw non-image bytes map to `EmbeddedResource` with `BlobResourceContents`; blob resources map back to A2A raw file parts. |
| Structured data parts | Supported for prompt content bridges | Inbound A2A data parts are rendered into formatted JSON text for the fast-agent prompt. Outbound fast-agent `TextResourceContents` with `mimeType="application/json"` are emitted as A2A data parts. The A2A 1.0 structured data example also permits JSON returned as text artifacts. |
| Error states | Supported through SDK plus fast-agent mappings | Provider credential failures map to `TASK_STATE_AUTH_REQUIRED`; uncaught execution failures map to `TASK_STATE_FAILED`; cancellation maps to `TASK_STATE_CANCELED`. Transport and validation errors are handled by the SDK bindings. |

## Known Gaps

| Gap | Impact | Current behavior |
|---|---|---|
| gRPC transport | Not supported by design for this work. | The AgentCard does not advertise gRPC, and the CLI/API should use `JSONRPC` or `HTTP+JSON`. |
| Push notifications | Not implemented. | The AgentCard advertises `pushNotifications=false`; SDK push configuration methods return the protocol's not-supported error. Streaming and polling remain available. |
| Extended AgentCard | Not implemented. | The server publishes the public AgentCard only and does not configure `extendedAgentCard`. |
| Authentication/security schemes on served AgentCards | Partial. | Remote clients can pass headers when connecting to other A2A agents. Serving fast-agent over A2A does not yet expose configurable A2A security schemes or enforce transport-level client auth. In-task provider auth failures are reported as `AUTH_REQUIRED`. |
| Typed audio content on the server | Partial. | The client can send `AudioContent` as raw A2A parts. The server preserves inbound audio bytes as blob resources rather than mapping them to a dedicated fast-agent `AudioContent` object. |
| Structured JSON output from model text | Partial. | fast-agent JSON text responses remain text artifacts unless represented as `TextResourceContents` with `mimeType="application/json"`. This avoids guessing whether ordinary text is intended to be protocol data. |
| Persistent task/session storage | In-memory only. | The server uses the SDK `InMemoryTaskStore` and fast-agent in-memory context instances. Restarting the server loses A2A task state and context-bound fast-agent sessions. |
| Idempotent `messageId` handling | Not implemented in fast-agent layer. | The SDK validates request shape, but fast-agent does not deduplicate repeated `messageId` values. |
| AgentCard signing | Not implemented. | The public AgentCard is unsigned. |
| Extension negotiation | Not implemented. | The server does not advertise or process custom A2A extensions. |

## Verification

The deterministic A2A integration suite exercises:

- JSON-RPC and HTTP+JSON client/server calls;
- generated `context_id` continuity across turns;
- fresh A2A context/task state for `use_history=False` turns, except while
  continuing an `INPUT_REQUIRED` task;
- A2A server `shared`, `connection`, and `request` instance scopes;
- AgentCard skill advertisement and skill routing via message metadata;
- streaming artifact updates delivered to the fast-agent client stream listener;
- artifact replacement and append semantics on the client;
- `TASK_STATE_INPUT_REQUIRED` preservation and follow-up completion;
- raw non-image file preservation into fast-agent blob resources and back to
  A2A raw parts;
- raw image and audio input mapping;
- JSON `TextResourceContents` mapping to A2A data parts;
- cancellation, task retrieval/listing after cancellation, and protocol error
  paths via SDK-backed handlers.

For provider smoke testing, run a fast-agent A2A server with
`codexresponses.gpt-5.4-mini` and connect to it with the fast-agent A2A client.
The expected behavior is incremental streaming chunks delivered to client stream
listeners before the final task completion event.
