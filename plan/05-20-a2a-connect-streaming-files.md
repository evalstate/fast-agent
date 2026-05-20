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

---

## 2026-05-20 implementation update

The core short-term plan is now mostly implemented and committed in focused
checkpoints.

### Commit log for this feature slice

```text
a33ad0d1 test a2a jsonrpc and http transports
e8867e12 support a2a streaming and file parts
13fb9804 add a2a diagnostic commands
94e61939 add interactive a2a connect
59c603dc add a2a cli shortcut and fake server
fffa4a90 document and expose a2a transport diagnostics
98273942 handle a2a connection failures gracefully
2767f493 add a2a getting started docs pipeline
dc0e62ad embed asciinema player in a2a docs
0b91bf5c record colored a2a docs casts
6ccc47fb add a2a cast theme switch
a4fb64ba align a2a cast theme with docs styles
```

### Runtime and transport status

Implemented:

- SDK-backed integration fixture under `tests/integration/a2a/`.
- JSON-RPC runtime coverage.
- HTTP+JSON runtime coverage.
- A deterministic fake server:

  ```bash
  uv run python tests/integration/a2a/fake_server.py --port 41242
  ```

- CLI one-shot shortcut:

  ```bash
  uv run fast-agent -x \
    --a2a http://127.0.0.1:41242 \
    --a2a-transport JSONRPC \
    --message "please stream" \
    --quiet
  ```

- Direct card URL normalization for both CLI and TUI connect flows.
- Graceful connection failure handling for A2A initialization errors. Missing or
  unreachable servers now produce an `AgentConfigError` style message instead of
  a rich traceback.

Still open:

- gRPC transport coverage remains optional/deferred.
- `/a2a transport` reports requested transport and selected SDK client class, but
  deeper SDK transport introspection may still be worth improving if the SDK
  exposes a stable public surface.

### Current fake server behavior

The fake server is the source of truth for repeatable docs/tests/demos. It
exposes:

```text
AgentCard: http://127.0.0.1:41242/.well-known/agent-card.json
JSON-RPC:  http://127.0.0.1:41242/a2a/jsonrpc
REST:      http://127.0.0.1:41242/a2a/rest
```

Useful prompts:

```text
hello
please stream
respond with files
```

It deterministically exercises:

- text echo;
- delayed streaming artifact chunks;
- inbound URL/data/raw rendering;
- outbound text/url/raw part mapping.

### Streaming and file/data mapping status

Implemented outbound mapping:

| fast-agent/MCP content | A2A part |
|---|---|
| `TextContent` | `Part(text=...)` |
| `ImageContent` / `AudioContent` | `Part(raw=..., media_type=...)` |
| `ResourceLink` | `Part(url=..., media_type=..., filename=...)` |
| `EmbeddedResource` blob | `Part(raw=..., media_type=..., filename=...)` |
| `EmbeddedResource` text | `Part(text=..., media_type=..., filename=...)` |

Implemented inbound rendering:

| A2A part | fast-agent rendering |
|---|---|
| `text` | assistant text |
| `url` | Markdown link with media type when present |
| `data` | fenced JSON |
| `raw` | safe `[filename: N bytes media/type]` placeholder |

Streaming events are consumed incrementally and emitted to stream listeners while
still producing a final normal assistant turn. The CLI currently prints final
aggregated text for one-shot mode; the TUI path displays the completed assistant
turn in the usual A2A-styled transcript.

### `/a2a` command surface now implemented

```text
/a2a
/a2a list
/a2a status [agent]
/a2a card [agent]
/a2a transport [agent]
/a2a reset [agent]
/a2a connect <url> [--transport JSONRPC|HTTP+JSON|GRPC] [--name NAME] [--card-path PATH]
```

Notes:

- `/a2a` defaults to status for the current agent.
- `/a2a connect` creates a runtime A2A agent and switches to it.
- `/a2a reset` clears local A2A `context_id`, `current_task_id`, and
  `last_task_state`.
- Persistent write-back/save is still deferred.

### CLI A2A shortcut now implemented

New flags:

```text
--a2a <url>                 repeatable; base URL or direct AgentCard URL
--a2a-transport <transport> JSONRPC, HTTP+JSON, GRPC, or aliases
```

Transport aliases:

```text
jsonrpc, json-rpc, rpc -> JSONRPC
http, http+json, rest  -> HTTP+JSON
grpc                   -> GRPC
```

Generated temporary agents are named:

```text
a2a_remote
a2a_remote_2
...
```

If exactly one `--a2a` is supplied and `--agent` is omitted, fast-agent targets
that temporary A2A agent automatically.

### Documentation and recording pipeline

A new top-level docs section exists:

```text
A2A
└── Getting Started
```

Primary page:

```text
docs/docs/a2a/getting-started.md
```

Docs nav update:

```text
docs/zensical.toml
```

Generated snippets live under:

```text
docs/docs/a2a/snippets/
```

Current snippets:

```text
agent-card.yaml
cli-files-command.sh
cli-files-output.txt
cli-stream-command.sh
cli-stream-output.txt
start-fake-server.sh
tui-session.txt
```

The repeatable pipeline is:

```bash
uv run scripts/a2a_docs_pipeline.py generate
uv run scripts/a2a_docs_pipeline.py check
uv run scripts/a2a_docs_pipeline.py record
```

Convenience wrapper:

```bash
uv run scripts/docs.py a2a
```

The `record` command now captures a live tmux session with asciinema rather than
capturing a plain-text tmux pane. This preserves ANSI colors in the `.cast`.

Current committed recording:

```text
docs/docs/assets/a2a/a2a-streaming-files.cast
```

Current recording metadata:

```text
width: 104
height: 27
idle_time_limit: 1.3
contains ANSI escape sequences: yes
```

The old local working recordings under `/home/ssmith/plan/records/` are still
useful as scratch/reference artifacts, but the docs source of truth is now the
committed asset under `docs/docs/assets/a2a/`.

### Embedded asciinema player

The A2A Getting Started page embeds the cast using vendored asciinema-player
assets:

```text
docs/docs/assets/vendor/asciinema-player/asciinema-player.css
docs/docs/assets/vendor/asciinema-player/asciinema-player.min.js
docs/docs/assets/vendor/asciinema-player/catppuccin.css
```

Despite the legacy filename `catppuccin.css`, the CSS now defines fast-agent
native terminal themes aligned to `docs/docs/stylesheets/fast-agent.css` tokens:

```text
fast-agent-light
fast-agent-dark
```

The page includes a player-local switch:

```text
Auto | Light | Dark
```

Behavior:

- Auto follows the Zensical docs light/dark mode.
- Light forces `fast-agent-light`.
- Dark forces `fast-agent-dark`.
- The player recreates itself when the local player theme or docs site theme
  changes.

### Tests added/updated

A2A runtime and command tests now include:

```text
tests/integration/a2a/conftest.py
tests/integration/a2a/test_remote_agent_runtime.py
tests/integration/a2a/fake_server.py
tests/unit/fast_agent/a2a_connect_test.py
tests/unit/fast_agent/cli/test_a2a_go_options.py
tests/unit/fast_agent/core/test_a2a_error_formatting.py
tests/unit/fast_agent/ui/test_parse_a2a_commands.py
tests/unit/test_a2a_docs_pipeline.py
```

The docs pipeline tests check that:

- generated snippets are current;
- the Getting Started page includes all generated snippets;
- the cast asset is present;
- the cast metadata uses the compact 27-row size;
- the cast contains ANSI escape sequences;
- vendored asciinema assets are present;
- fast-agent light/dark terminal themes are present;
- the page includes Auto/Light/Dark controls.

### Validation commands run after latest changes

```bash
uv run pytest tests/unit/test_a2a_docs_pipeline.py -q
uv run scripts/a2a_docs_pipeline.py check
uv run scripts/lint.py
uv run scripts/typecheck.py
uv run scripts/docs.py build
```

Earlier broader validation also passed:

```bash
uv run pytest tests/integration/a2a \
  tests/unit/fast_agent/a2a_connect_test.py \
  tests/unit/fast_agent/cli/test_a2a_go_options.py \
  tests/unit/fast_agent/core/test_a2a_error_formatting.py \
  tests/unit/fast_agent/ui/test_parse_a2a_commands.py \
  tests/unit/test_a2a_docs_pipeline.py -q
```

### Remaining recommended follow-ups

1. Add optional/skipped gRPC integration coverage.
2. Rename `catppuccin.css` to a fast-agent-specific filename if we want the asset
   name to match its current purpose. This is cosmetic but would reduce future
   confusion.
3. Add docs for persistent A2A session state once `context_id` resume semantics
   are decided.
4. Consider a `/a2a save [agent] <path>` command to write a connected runtime A2A
   agent back to an AgentCard.
5. Consider richer inbound raw-byte handling if/when fast-agent has a standard
   artifact storage/display path.
