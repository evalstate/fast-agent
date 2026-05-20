# A2A orientation guide

Date: 2026-05-21
Last updated: 2026-05-21

## Purpose

This note is a handoff/orientation guide for fast-agent's current A2A client
work. It points to the key files, deterministic fixtures, docs assets, and
manual commands so the next session can avoid rediscovery.

## Current feature shape

fast-agent is currently an A2A **client**:

- remote A2A agents can be configured via `type: a2a` AgentCards;
- `/a2a connect` can create a runtime A2A agent from the TUI;
- JSON-RPC and HTTP+JSON are covered by deterministic integration tests;
- gRPC is accepted/configurable through the SDK path but does not yet have
  fast-agent-owned integration coverage;
- A2A messages map to normal fast-agent user/assistant turns and local history;
- A2A `context_id`, `task_id`, and task state are tracked on `A2ARemoteAgent`.

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

- `src/fast_agent/a2a/connect.py`
  - URL and argument normalization for `/a2a connect`.
  - Transport aliases:
    - `jsonrpc`, `json-rpc`, `rpc` -> `JSONRPC`
    - `http`, `http+json`, `rest` -> `HTTP+JSON`
    - `grpc` -> `GRPC`

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
  - `/a2a connect` runtime agent creation.
  - `INPUT_REQUIRED` task preservation and follow-up completion.

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

## Docs and recordings

- `docs/docs/a2a/getting-started.md`
  - User-facing A2A getting-started page.
  - Includes short streaming, long streaming, file/data/raw, and
    `INPUT_REQUIRED` explanation.

- `docs/docs/a2a/snippets/`
  - Generated snippets consumed by the docs page.

- `docs/docs/assets/a2a/a2a-streaming-files.cast`
  - Embedded asciinema recording for the A2A TUI flow.

- Requested follow-up recording:
  - Add one docs asciinema cast that shows a real LLM-backed fast-agent A2A
    server streaming to a fast-agent A2A client.
  - Preferred scenario: serve an agent using `codexresponses.gpt-5.4-mini`
    with access to the Hugging Face MCP server, then ask for a markdown
    response about currently trending models.
  - This should be labeled as a provider/network smoke recording, separate from
    the deterministic fake-server recordings used for repeatable tests.

- `/home/ssmith/plan/records/a2a-streaming-files-input-required.cast`
  - Local copy of the latest generated recording.

- `scripts/a2a_docs_pipeline.py`
  - `generate`: refresh snippets and CLI outputs.
  - `check`: verify snippets/assets are in sync.
  - `record`: regenerate the asciinema recording using tmux.
  - Note: the script uses fixed port `41242`; if a stale fake server is already
    bound there it now fails early.

Commands:

```bash
uv run scripts/a2a_docs_pipeline.py generate
uv run scripts/a2a_docs_pipeline.py check
uv run scripts/a2a_docs_pipeline.py record
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

## Known gaps / next good targets

- A2A TVD capability chip:
  - remote card `default_input_modes` and `skills[*].input_modes` can be used to
    infer text/document/vision capability;
  - toolbar currently special-cases A2A and does not render a TVD segment.

- Inbound file persistence:
  - inbound raw bytes are currently rendered as `[filename: N bytes media/type]`;
  - they are not saved to `<env>/a2a/` yet.

- Streaming fidelity:
  - artifact updates emit `StreamChunk`s;
  - status/task events update state and log progress;
  - `TaskArtifactUpdateEvent.append`/`last_chunk` is not yet used for nuanced
    artifact assembly beyond whole-text de-dupe.

- gRPC integration coverage:
  - accepted by config/SDK;
  - no fast-agent-owned deterministic gRPC test yet.

## Required validation after A2A changes

At minimum:

```bash
uv run pytest tests/integration/a2a -q
uv run scripts/a2a_docs_pipeline.py check
uv run scripts/lint.py
uv run scripts/typecheck.py
```
