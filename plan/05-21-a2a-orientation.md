# A2A orientation guide

Date: 2026-05-21
Last updated: 2026-05-21

## Purpose

This note is a handoff/orientation guide for fast-agent's current A2A client
and server work. It points to the key files, deterministic fixtures, docs
assets, and manual commands so the next session can avoid rediscovery.

## Current feature shape

fast-agent is currently both an A2A **client** and **server** for HTTP
transports:

- remote A2A agents can be configured via `type: a2a` AgentCards;
- `/a2a connect` can create a runtime A2A agent from the TUI;
- JSON-RPC and HTTP+JSON are covered by deterministic integration tests;
- gRPC is intentionally out of scope for this A2A support pass and is rejected
  by fast-agent card/command validation;
- A2A messages map to normal fast-agent user/assistant turns and local history;
- A2A `context_id`, `task_id`, and task state are tracked on `A2ARemoteAgent`.
- `fast-agent serve a2a` and `fast-agent serve --transport a2a` expose loaded
  fast-agent agents over JSON-RPC and HTTP+JSON.
- served A2A AgentCards advertise one A2A `AgentSkill` per loaded fast-agent
  agent, plus JSON-RPC and HTTP+JSON interfaces.
- server-side `contextId` is optional in inbound A2A messages; the SDK resolves
  one when omitted, and `--instance-scope connection` uses it as the fast-agent
  instance/session key.

## Key implementation files

### Runtime/client adapter

- `src/fast_agent/a2a/remote_agent.py`
  - `A2ARemoteAgent` is the main adapter from fast-agent `AgentProtocol` to a
    remote A2A SDK client.
  - Builds `SendMessageRequest` from fast-agent messages.
  - Converts outbound content to A2A `Part`s in `_parts_from_messages(...)`.
  - Consumes A2A event streams in `_consume_events(...)`.
  - Tracks `context_id`, `current_task_id`, and `last_task_state`.
  - Emits `StreamChunk` values from direct message/artifact text updates.
  - Renders inbound URL/data/raw/text parts via `_part_text(...)`.

- `src/fast_agent/a2a/config.py`
  - `A2AAgentConfig` fields:
    - `url`
    - `transport`
    - `streaming`
    - `polling`
    - `accepted_output_modes`
    - `headers`
    - `relative_card_path`
    - `request_timeout_seconds`

- `src/fast_agent/a2a/connect.py`
  - URL and argument normalization for `/a2a connect`.
  - Transport aliases:
    - `jsonrpc`, `json-rpc`, `rpc` -> `JSONRPC`
    - `http`, `http+json`, `rest` -> `HTTP+JSON`
    - `grpc` -> rejected; gRPC is not supported by fast-agent A2A.

### Server adapter

- `src/fast_agent/a2a/server.py`
  - `AgentA2AServer` exposes fast-agent through SDK JSON-RPC and REST routes.
  - `FastAgentA2AExecutor` maps A2A messages to `PromptMessageExtended`, calls
    the selected fast-agent agent, and emits A2A task status/artifact updates.
  - `TaskArtifactUpdateEvent.append` is used for streaming chunks; the final
    response can replace the streamed artifact when needed.
  - `TASK_STATE_INPUT_REQUIRED` is returned when the fast-agent response has
    `LlmStopReason.PAUSE`.
  - `AUTH_REQUIRED`, `FAILED`, and `CANCELED` states are mapped from provider
    auth errors, unexpected execution errors, and cancellation.
  - raw image/file/data/text URL parts are bridged to and from fast-agent
    content blocks.

### AgentCard/direct factory wiring

- `src/fast_agent/core/agent_card_loader.py`
  - Parses and serializes `type: a2a` cards.

- `src/fast_agent/core/agent_card_rules.py`
  - A2A card validation/rules.

- `src/fast_agent/core/direct_factory.py`
  - `_create_a2a_agent(...)` constructs and initializes `A2ARemoteAgent`.

### UI/TUI wiring

- `src/fast_agent/ui/interactive/command_dispatch.py`
  - Dispatches `/a2a list`, `/a2a status`, `/a2a card`, `/a2a reset`,
    `/a2a transport`, and `/a2a connect`.

- `src/fast_agent/ui/prompt/parser.py`
  - Parses `/a2a ...` input into `A2ACommand`.

- `src/fast_agent/ui/prompt/input_toolbar.py`
  - A2A toolbar treatment.
  - A2A agents show `A2A`/remote card name instead of local model info.

- `src/fast_agent/ui/prompt/toolbar.py`
  - Active-agent toolbar styling shows `name[A2A]`.

### CLI wiring

- `src/fast_agent/cli/commands/go.py`
  - `--a2a` and `--a2a-transport` runtime connection shortcuts.

- `src/fast_agent/cli/commands/serve.py`
  - `fast-agent serve a2a` subcommand.
  - legacy-compatible `fast-agent serve --transport a2a` callback path.

## Deterministic A2A fixtures and tests

### Integration fixture

- `tests/integration/a2a/conftest.py`
  - Starts an in-process deterministic A2A server on a free port.
  - `EchoAgentExecutor` scenarios:
    - normal echo;
    - fake server help: `help`, `?`, `commands`, `menu`, or `what can you do`;
    - short stream: `please stream`;
    - long stream: `please long stream`;
    - file/data/raw response: `respond with files`;
    - `INPUT_REQUIRED` flow: `need input`, then any follow-up such as `blue`.

### Manual fake server

- `tests/integration/a2a/fake_server.py`
  - Standalone deterministic fake server for CLI/TUI demos.
  - Run:

    ```bash
    uv run python tests/integration/a2a/fake_server.py --port 41242
    ```

  - AgentCard:

    ```text
    http://127.0.0.1:41242/.well-known/agent-card.json
    ```

  - JSON-RPC:

    ```text
    http://127.0.0.1:41242/a2a/jsonrpc
    ```

  - HTTP+JSON:

    ```text
    http://127.0.0.1:41242/a2a/rest
    ```

  - Useful prompts:
    - `help`
    - `hello`
    - `please stream`
    - `please long stream`
    - `respond with files`
    - `need input`, followed by `blue`

### Tests

- `tests/integration/a2a/test_remote_agent_runtime.py`
  - JSON-RPC and HTTP+JSON text send.
  - Short streaming chunks.
  - Long streaming chunks.
  - Inbound URL/data/raw rendering.
  - Outbound URL/raw parts.
  - JSON `TextResourceContents` emitted as A2A data parts.
  - no-history A2A context reset except while continuing `INPUT_REQUIRED`.
  - `/a2a connect` runtime agent creation.
  - `INPUT_REQUIRED` task preservation and follow-up completion.

- `tests/integration/a2a/test_fast_agent_a2a_server.py`
  - fast-agent served as JSON-RPC and HTTP+JSON A2A server.
  - context/session continuity, request/shared/connection instance scopes, and
    served-agent `use_history` behavior.
  - generated AgentCard interfaces, wildcard host rewriting, and A2A
    `AgentSkill` advertisement/routing.
  - streaming artifact updates, final artifact replacement, and cancellation.
  - raw image/audio/file preservation and outbound raw/data/url/text mapping.
  - task list/get/cancel behavior through SDK handlers.

- Unit coverage:
  - `tests/unit/fast_agent/test_a2a_remote_agent_events.py`
  - `tests/unit/fast_agent/test_a2a_remote_agent_config.py`
  - `tests/unit/fast_agent/a2a_connect_test.py`
  - `tests/unit/fast_agent/cli/test_a2a_go_options.py`
  - `tests/unit/fast_agent/cli/test_a2a_serve_options.py`
  - `tests/unit/fast_agent/ui/test_parse_a2a_commands.py`
  - `tests/unit/fast_agent/ui/test_a2a_command_dispatch.py`
  - `tests/unit/fast_agent/core/test_a2a_error_formatting.py`

Run:

```bash
uv run pytest tests/integration/a2a -q
```

## Manual smoke commands

Start fake server:

```bash
uv run python tests/integration/a2a/fake_server.py --port 41242
```

Short stream:

```bash
uv run fast-agent -x \
  --a2a http://127.0.0.1:41242 \
  --a2a-transport JSONRPC \
  --message "please stream" \
  --quiet
```

Long stream:

```bash
uv run fast-agent -x \
  --a2a http://127.0.0.1:41242 \
  --a2a-transport JSONRPC \
  --message "please long stream" \
  --quiet
```

Files/data/raw:

```bash
uv run fast-agent -x \
  --a2a http://127.0.0.1:41242 \
  --a2a-transport HTTP+JSON \
  --message "respond with files" \
  --quiet
```

TUI:

```bash
uv run fast-agent -x --a2a http://127.0.0.1:41242 --a2a-transport JSONRPC
```

Then try:

```text
/a2a help
help
/a2a status
/a2a transport
please stream
please long stream
respond with files
need input
/a2a status
blue
```

Serve fast-agent as A2A:

```bash
uv run fast-agent serve a2a \
  --host 127.0.0.1 \
  --port 41241 \
  --instance-scope connection \
  --agent-cards ./agents
```

Fetch the served card:

```bash
curl -s http://127.0.0.1:41241/.well-known/agent-card.json | jq .
```

## Docs and recordings

- `docs/docs/a2a/getting-started.md`
  - User-facing A2A getting-started page.
  - Includes short streaming, long streaming, file/data/raw, and
    `INPUT_REQUIRED` explanation.

- `docs/docs/a2a/client.md`
  - Client CLI, AgentCard, TUI, streaming, `INPUT_REQUIRED`, content mapping,
    error handling, and resumption guidance.
  - Embeds deterministic client recordings and the provider-backed real LLM
    recording.

- `docs/docs/a2a/server.md`
  - `fast-agent serve a2a`, served AgentCard interface URLs, runtime wiring,
    A2A `AgentSkill` exposure, instance scopes, streaming, errors, file parts,
    and structured JSON.

- `docs/docs/a2a/api.md`
  - Direct `A2ARemoteAgent` usage, `AgentA2AServer` embedding, raw JSON-RPC and
    HTTP+JSON examples, content mapping, and explicit JSON data part examples.

- `docs/docs/a2a/protocol-compliance.md`
  - Current support matrix, known gaps, and verification coverage against A2A
    Protocol Specification 1.0.

- `docs/docs/a2a/snippets/`
  - Generated snippets consumed by the docs page.

- `docs/docs/assets/a2a/a2a-streaming-files.cast`
  - Embedded asciinema recording for the A2A TUI flow.

- `docs/docs/assets/a2a/a2a-real-llm-hf-streaming.cast`
  - Provider/network smoke recording.
  - Shows `fast-agent serve a2a` backed by `codexresponses.gpt-5.4-mini`,
    connected to the Hugging Face MCP server, and an interactive fast-agent A2A
    client asking for a markdown answer about trending Hugging Face models.

- `/home/ssmith/plan/records/a2a-streaming-files-input-required.cast`
  - Local copy of the latest generated recording.

- `scripts/a2a_docs_pipeline.py`
  - `generate`: refresh snippets and CLI outputs.
  - `check`: verify snippets/assets are in sync.
  - `record`: regenerate the asciinema recording using tmux.
  - `record-real-llm`: regenerate the provider-backed Hugging Face MCP/LLM
    streaming recording; requires `HF_TOKEN`, `OPENAI_API_KEY`, network access,
    `asciinema`, `tmux`, and `curl`.
  - Note: the script uses fixed port `41242`; if a stale fake server is already
    bound there it now fails early.

Commands:

```bash
uv run scripts/a2a_docs_pipeline.py generate
uv run scripts/a2a_docs_pipeline.py check
uv run scripts/a2a_docs_pipeline.py record
uv run scripts/a2a_docs_pipeline.py record-real-llm
```

## A2A SDK reference checkout

Local SDK checkout:

```text
../a2a-python/
```

Useful files there:

- `samples/hello_world_agent.py`
  - Standalone SDK sample server exposing JSON-RPC, HTTP+JSON, gRPC v1.0, and
    gRPC v0.3 compatibility.

- `tck/sut_agent.py`
  - Useful reference for `TaskStatusUpdateEvent`, `TASK_STATE_WORKING`,
    `TASK_STATE_INPUT_REQUIRED`, cancellation, and multi-transport setup.

- `tests/integration/test_end_to_end.py`
  - Good fixture/reference for direct message responses, task responses,
    status-message text, artifact updates, and `INPUT_REQUIRED`.

## Protocol/event concepts currently used

Observed current A2A stream payload oneof fields:

- `task`
- `message`
- `status_update`
- `artifact_update`

Relevant proto/model fields:

- `Task.context_id`
- `Task.id`
- `Task.status.state`
- `Task.artifacts`
- `TaskStatus.message`
- `TaskStatusUpdateEvent.task_id`
- `TaskStatusUpdateEvent.context_id`
- `TaskArtifactUpdateEvent.artifact`
- `TaskArtifactUpdateEvent.append`
- `TaskArtifactUpdateEvent.last_chunk`
- `Message.context_id`
- `Message.task_id`
- `Message.parts`
- `Part.text`
- `Part.raw`
- `Part.url`
- `Part.data`
- `Part.filename`
- `Part.media_type`

Current fast-agent mapping:

- one `A2ARemoteAgent` instance owns one active remote `context_id`;
- terminal non-input-required tasks clear `current_task_id`;
- `TASK_STATE_INPUT_REQUIRED` preserves `current_task_id`;
- the next user turn is sent with that pending `task_id`;
- `/a2a reset` creates a fresh remote context and clears task state.
- when local A2A card/request `use_history=False`, the client generates a fresh
  remote context for completed turns, but preserves context/task state while
  continuing `INPUT_REQUIRED`.
- on the server, `--instance-scope connection` maps resolved A2A `context_id`
  to a fast-agent instance; `shared` ignores per-context instance isolation and
  `request` creates a fresh instance per message.
- structured JSON is protocol-level data only when represented as A2A `data`
  parts or fast-agent `TextResourceContents(mimeType="application/json")`.
  Ordinary model text remains a text artifact.

## Known gaps / next good targets

- A2A TVD capability chip:
  - remote card `default_input_modes` and `skills[*].input_modes` can be used to
    infer text/document/vision capability;
  - toolbar currently special-cases A2A and does not render a TVD segment.

- Inbound file persistence:
  - server-side inbound raw bytes are preserved as `BlobResourceContents` for
    the fast-agent agent;
  - client-side inbound remote raw bytes are still rendered as readable text
    placeholders and are not saved to `<env>/a2a/` yet.

- gRPC integration coverage:
  - intentionally out of scope for the current A2A HTTP support target.

- Persistent task/session storage:
  - the server currently uses the SDK `InMemoryTaskStore` plus in-memory
    fast-agent instances; process restart loses A2A task/context state.

- A2A security schemes:
  - client connections support headers;
  - served A2A AgentCards do not yet advertise configurable security schemes or
    enforce A2A transport-level client authentication.

## Required validation after A2A changes

At minimum:

```bash
uv run pytest tests/integration/a2a -q
uv run pytest tests/unit/fast_agent/test_a2a_remote_agent_events.py \
  tests/unit/fast_agent/test_a2a_remote_agent_config.py \
  tests/unit/fast_agent/a2a_connect_test.py \
  tests/unit/fast_agent/cli/test_a2a_go_options.py \
  tests/unit/fast_agent/cli/test_a2a_serve_options.py \
  tests/unit/fast_agent/ui/test_parse_a2a_commands.py \
  tests/unit/fast_agent/ui/test_a2a_command_dispatch.py \
  tests/unit/fast_agent/core/test_a2a_error_formatting.py -q
uv run scripts/a2a_docs_pipeline.py check
uv run scripts/lint.py
uv run scripts/typecheck.py
```
