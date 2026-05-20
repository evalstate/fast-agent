# A2A connect, transport, streaming, and file handling plan

Date: 2026-05-20
Owner: fast-agent A2A client work
Recording folder: `/home/ssmith/plan/records/`

## Goal

Make remote A2A agents easy to connect to, prove each supported transport works,
and support the happy path for streaming and file-capable A2A messages from both
CLI and TUI flows.

The feature should be demonstrable against both the A2A SDK sample server and a
small fast-agent-owned fake A2A server fixture that deterministically exercises
streaming, artifacts, and file/data parts.

## Working rules

- Keep diffs small and commit at natural checkpoints.
- Before each commit, run at minimum:

  ```bash
  uv run scripts/lint.py
  uv run scripts/typecheck.py
  ```

- For focused changes, also run the relevant pytest target before committing.
- Do not rewrite unrelated files or existing user changes.
- Store terminal captures and asciinema recordings under:

  ```text
  /home/ssmith/plan/records/
  ```

- Prefer SDK/server fixtures over monkeypatching A2A clients.

## Outcomes

1. A user can test a remote A2A endpoint interactively with `/a2a connect`.
2. JSON-RPC and HTTP+JSON are covered by SDK-backed integration tests.
3. gRPC is covered when optional dependencies/environment are available, and is
   skipped clearly otherwise.
4. A2A streaming updates render without duplicate/blank headers and settle into a
   normal assistant turn in the TUI.
5. A2A file/data happy paths are implemented and tested:
   - outbound fast-agent attachment or URL -> A2A `Part(raw=...)`/`Part(url=...)`;
   - inbound A2A `Part(url=...)`/`Part(data=...)`/text artifact -> readable
     fast-agent assistant content;
   - raw inbound bytes are at least detected and represented safely.
6. The same scenarios are testable through non-interactive CLI and through the TUI.

## URL and transport semantics

### Accepted connect URLs

`/a2a connect` should accept:

```text
/a2a connect http://127.0.0.1:41241
/a2a connect https://agent.example.com
/a2a connect https://agent.example.com/base --card-path /.well-known/agent-card.json
/a2a connect http://127.0.0.1:41241/.well-known/agent-card.json
```

Preferred input is the A2A agent base URL. Direct agent-card URLs may be accepted
as a convenience by normalizing to base URL plus `relative_card_path`.

Endpoint URLs such as `/a2a/jsonrpc` are not the preferred user input. If they are
provided, emit a clear diagnostic explaining that fast-agent expects the base URL
or card URL.

### Accepted transports

Canonical transport names passed to the SDK:

```text
JSONRPC
HTTP+JSON
GRPC
```

Friendly command aliases should normalize as follows:

```text
jsonrpc, json-rpc, rpc       -> JSONRPC
http, http+json, rest        -> HTTP+JSON
grpc                         -> GRPC
```

If no transport is provided, let the SDK choose from the remote AgentCard and show
what was selected.

## Step 1 — Baseline transport integration tests

### Implementation

- Add an integration test fixture that starts an A2A SDK-compatible test server.
- Cover creation through normal fast-agent card/factory/runtime paths.
- Test JSON-RPC text request/response.
- Test HTTP+JSON text request/response.
- Add optional gRPC coverage guarded by dependency/port availability.

### Test commands

```bash
uv run pytest tests/integration/a2a -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

### CLI check

```bash
uv run fast-agent -x --agent-cards /tmp/a2a-card.yaml --agent hello_remote --message hello --quiet
```

Expected: text response from the remote server.

### TUI check

Use tmux to start fast-agent, send `hello`, capture pane text, and verify the
assistant response appears under an `A2A` assistant header.

### Commit

Commit after tests/lint/typecheck pass.

## Step 2 — Fake A2A server fixture for deterministic behavior

### Implementation

Build a small local test server under tests/support or tests/integration/a2a that
uses SDK server primitives and exposes:

- text echo response;
- delayed streaming status updates;
- artifact text updates;
- outbound file/data response modes;
- JSON-RPC and HTTP+JSON routes;
- gRPC only if available without making CI brittle.

Avoid coupling tests to the external SDK sample process where possible. Keep the
external sample useful for manual smoke tests and recordings.

### Test commands

```bash
uv run pytest tests/integration/a2a -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

### Commit

Commit after the fixture and baseline tests pass.

## Step 3 — `/a2a status`, `/a2a card`, `/a2a reset`

### Implementation

Add the diagnostic commands before connect so connect has reusable reporting
helpers.

Expected surfaces:

```text
/a2a status [agent]
/a2a card [agent]
/a2a reset [agent]
```

`status` should show URL, remote card name, selected/requested transport,
streaming/polling flags, context id, current task id, last task state, and output
modes where available.

`card` should show the resolved remote AgentCard summary and supported
interfaces.

`reset` should clear `context_id`, `current_task_id`, and `last_task_state`.

### Test commands

```bash
uv run pytest tests/unit/fast_agent/ui tests/integration/a2a -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

### CLI/TUI checks

- TUI: run `/a2a status`, `/a2a card`, `/a2a reset`, then send `hello`.
- CLI if slash-command execution is available non-interactively: run equivalent
  command dispatch tests or documented command invocation.

### Commit

Commit after command behavior is tested.

## Step 4 — `/a2a connect`

### Implementation

Add:

```text
/a2a connect <base-url-or-card-url> [--transport JSONRPC|HTTP+JSON|GRPC] [--name NAME] [--card-path PATH]
```

Behavior:

1. Normalize URL/card path.
2. Resolve remote AgentCard.
3. Display remote card summary and supported interfaces.
4. Validate requested transport if supplied.
5. Create a runtime A2A agent or update/switch to a temporary connected agent.
6. Show selected transport and next action.

Persistent save/write-back is deferred unless trivial. If deferred, print the
YAML snippet the user can save.

### Test commands

```bash
uv run pytest tests/unit/fast_agent/ui tests/integration/a2a -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

### CLI check

```bash
uv run fast-agent -x --agent-cards /tmp/minimal-local-card.yaml --agent passthrough_or_default
```

Then in TUI:

```text
/a2a connect http://127.0.0.1:41241 --transport JSONRPC --name hello_remote
hello
```

Expected: active/connected A2A agent responds.

### Recording

Save an asciinema capture:

```text
/home/ssmith/plan/records/a2a-connect-jsonrpc.cast
```

### Commit

Commit after `/a2a connect` works and recordings are captured.

## Step 5 — Streaming display

### Implementation

Route A2A streaming events into the existing streaming/progress display rather
than only aggregating final text.

Requirements:

- task status updates should update progress without corrupting the transcript;
- artifact/direct-message text should appear as streaming assistant content;
- final render should match local LLM assistant turn behavior;
- full-snapshot artifact servers should not duplicate text excessively;
- progress board should pause before the final assistant message.

### Test commands

```bash
uv run pytest tests/integration/a2a -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

### CLI check

```bash
uv run fast-agent -x --agent-cards /tmp/a2a-stream-card.yaml --agent stream_remote --message "stream please" --quiet
```

Expected: final aggregated text is correct.

### TUI check

Use tmux to verify visible streaming/progress shape. Capture text to:

```text
/home/ssmith/plan/records/a2a-streaming-tui.txt
```

Record asciinema:

```text
/home/ssmith/plan/records/a2a-streaming.cast
```

### Commit

Commit after tests, lint, typecheck, and TUI capture.

## Step 6 — File/data happy path

### Implementation

Outbound:

- map user text to `Part(text=...)` as today;
- map local file attachments to `Part(raw=..., media_type=..., filename=...)`;
- map URL/resource attachments to `Part(url=..., media_type=..., filename=...)`;
- preserve plain text fallback when unsupported.

Inbound:

- render `Part(text=...)` as assistant text;
- render `Part(url=...)` as markdown links with media type/filename when present;
- render `Part(data=...)` as fenced JSON;
- represent `Part(raw=...)` safely with filename/media type/byte count, and save
  bytes only if there is an established artifact storage path.

### Test commands

```bash
uv run pytest tests/integration/a2a tests/e2e/multimodal -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

### CLI checks

Use a local sample file and the fake server:

```bash
uv run fast-agent -x --agent-cards /tmp/a2a-file-card.yaml --agent file_remote --message "summarize attached file" --quiet
```

If the current CLI has an attachment flag, include it; otherwise record the gap
and test through TUI/resource input.

### TUI checks

- Attach or reference a local file/resource.
- Send to A2A fake server.
- Verify server receives file metadata/content.
- Verify inbound URL/data response renders readably.

Save captures:

```text
/home/ssmith/plan/records/a2a-file-cli.txt
/home/ssmith/plan/records/a2a-file-tui.txt
/home/ssmith/plan/records/a2a-file.cast
```

### Commit

Commit after file/data happy path is tested and recorded.

## Step 7 — Documentation and final verification

### Implementation

- Update user docs/examples for `type: a2a` cards.
- Document `/a2a connect` URL and transport semantics.
- Document known limitations for binary inbound data and persistence.
- Add short demo recording links if the docs pipeline supports asciinema assets.

### Final validation

```bash
uv run pytest tests/integration/a2a -q
uv run pytest tests/unit/fast_agent/core/test_agent_card_loader.py -q
uv run pytest tests/unit/fast_agent/ui/test_enhanced_prompt_toolbar.py tests/unit/fast_agent/ui/test_input_toolbar.py -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

Manual smoke matrix:

| Scenario | CLI | TUI | Recording |
|---|---:|---:|---:|
| JSON-RPC text | yes | yes | optional |
| HTTP+JSON text | yes | yes | optional |
| gRPC text | optional | optional | optional |
| `/a2a status/card/reset` | command tests | yes | optional |
| `/a2a connect` | n/a or command test | yes | yes |
| Streaming artifact/direct text | yes final text | yes live display | yes |
| File/data happy path | yes if attachment path exists | yes | yes |

### Final commit

Commit docs and final verification notes.
