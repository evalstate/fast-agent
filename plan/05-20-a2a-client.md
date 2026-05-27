# A2A client integration status

Historical status: superseded.

This file records the initial 2026-05-20 client-only planning pass. It is kept
as source material for the later A2A implementation, but it is no longer the
authoritative plan:

- fast-agent now supports both A2A client and A2A server paths for HTTP
  transports;
- gRPC is intentionally out of scope for the current fast-agent A2A support
  target;
- current orientation, evidence, and remaining gaps live in:
  - `plan/05-21-a2a-orientation.md`;
  - `plan/05-21-a2a-goal-addendum.md`;
  - `plan/05-21-a2a-completion-audit.md`;
  - `docs/docs/a2a/protocol-compliance.md`.

The sections below should be read as implementation history, not as current
acceptance criteria.

Date: 2026-05-20
Last updated: 2026-05-20

## Goal

Allow fast-agent to treat remote A2A agents as first-class configured agents via the
existing AgentCard mechanism.

Initial scope remains client-only. Server-side fast-agent-as-A2A is deferred.

## Current status

Implemented and smoke-tested:

- `type: a2a` AgentCards load successfully.
- `AgentType.A2A` is registered and participates in direct factory creation.
- A2A cards parse the currently supported fields:
  - `url`;
  - `transport` (`JSONRPC`, `HTTP+JSON`, `GRPC` accepted by config validation);
  - `streaming`;
  - `polling`;
  - `accepted_output_modes`;
  - `headers`;
  - `relative_card_path`.
- `A2ARemoteAgent` resolves the remote A2A AgentCard and creates an SDK client.
- JSON-RPC text request/response works against the SDK sample server.
- The A2A agent is created without attaching a local LLM.
- Fast-agent local history records A2A user/assistant turns; `/history` works.
- TUI now maps A2A turns into the existing display infrastructure:
  - user messages render via `ConsoleDisplay.show_user_message(...)`;
  - assistant messages render via `ConsoleDisplay.show_assistant_message(...)`;
  - the active-agent toolbar shows `name[A2A]` in magenta;
  - the toolbar model segment shows the remote card name instead of `$system.default`.
- CLI auto tool-card attachment now skips A2A agents, because A2A agents are not
  valid agents-as-tools parents.
- Unit coverage exists for A2A card parsing, transport validation, and toolbar
  A2A identity styling.

Validated commands:

```bash
uv run pytest tests/unit/fast_agent/core/test_agent_card_loader.py -q
uv run pytest tests/unit/fast_agent/ui/test_enhanced_prompt_toolbar.py tests/unit/fast_agent/ui/test_input_toolbar.py -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

## Reference A2A SDK sample server

The local A2A SDK source is at:

```text
../a2a-python/
```

Run the sample server:

```bash
cd ../a2a-python
uv run python samples/hello_world_agent.py
```

The sample exposes:

- agent card: `http://127.0.0.1:41241/.well-known/agent-card.json`
- JSON-RPC: `http://127.0.0.1:41241/a2a/jsonrpc`
- HTTP+JSON: `http://127.0.0.1:41241/a2a/rest`
- gRPC v1.0: `127.0.0.1:50051`
- gRPC v0.3 compatibility: `127.0.0.1:50052`

Minimal fast-agent card:

```yaml
type: a2a
name: hello_remote
url: http://127.0.0.1:41241
transport: JSONRPC
```

Smoke test without the TUI:

```bash
uv run fast-agent -x --agent-cards /tmp/a2a-card.yaml --agent hello_remote --message hello --quiet
```

Expected output:

```text
Hello World! Nice to meet you!
```

## Current runtime mapping

For each `agent.send(...)` / `agent.generate(...)` call:

1. fast-agent input is normalized by `LlmDecorator.generate(...)`.
2. `A2ARemoteAgent.generate_impl(...)` displays trailing user messages through
   the same console display path used by local LLM agents.
3. The latest user text is mapped to an A2A `Message`:

   ```python
   Message(
       role=Role.ROLE_USER,
       message_id=str(uuid.uuid4()),
       context_id=current_context_id,
       task_id=current_task_id,
       parts=[Part(text=user_text)],
   )
   ```

4. The SDK client sends `SendMessageRequest(message=message)`.
5. The returned async event iterator is consumed.
6. The agent records `context_id`, `current_task_id`, and `last_task_state` where
   available.
7. Text is aggregated from direct A2A messages and artifact updates.
8. A normal `PromptMessageExtended(role="assistant", ...)` is returned and
   displayed through `ConsoleDisplay.show_assistant_message(...)`.

Terminal state behavior currently implemented:

- completed: returns aggregated text, or a no-output message;
- failed/canceled/cancelled/rejected/input-required/auth-required: returns a
  clear A2A task state message;
- terminal non-input-required tasks clear `current_task_id`;
- `input_required` preserves task state for future follow-up work.

## TUI verification with tmux

`tmux` has been useful for deterministic TUI reproduction and regression checks.
It lets us start fast-agent, send keystrokes, and capture the visible terminal
state without manually driving the UI.

Example:

```bash
# Ensure sample A2A server is running first.
cat >/tmp/a2a-card.yaml <<'YAML'
type: a2a
name: hello_remote
url: http://127.0.0.1:41241
transport: JSONRPC
YAML

cd /home/ssmith/source/fast-agent-pr

tmux kill-session -t a2atest 2>/dev/null || true
tmux new-session -d -s a2atest \
  'cd /home/ssmith/source/fast-agent-pr && FAST_AGENT_MODEL=passthrough uv run fast-agent -x --agent-cards /tmp/a2a-card.yaml --agent hello_remote'

sleep 4
tmux send-keys -t a2atest 'whhhhhhaaaattt' Enter
sleep 4
tmux send-keys -t a2atest 'hello' Enter
sleep 4
tmux capture-pane -t a2atest -p -S -3000 | tail -80
```

Expected visible shape:

```text
▎▶ hello_remote ────────────────────────────────────────────────────────────────
whhhhhhaaaattt

▎◀ hello_remote A2A
Hello World! You said: 'whhhhhhaaaattt'. Thanks for your message!

▎▶ hello_remote ────────────────────────────────────────────────────────────────
hello

▎◀ hello_remote A2A
Hello World! Nice to meet you!

❯

  hello_remote[A2A]    ▲  Sample Agent | 002 |  NRML  | fast-agent 0.7.8
```

Useful tmux commands for future automated checks/docs captures:

```bash
# Capture current pane text.
tmux capture-pane -t a2atest -p -S -3000 > /tmp/a2a-tui.txt

# Append all pane output to a log while the session runs.
tmux pipe-pane -t a2atest -o 'cat >> /tmp/a2a-tui.log'

# Send slash commands or normal input.
tmux send-keys -t a2atest '/history' Enter
tmux send-keys -t a2atest 'hello' Enter

# Stop the session.
tmux kill-session -t a2atest
```

## Asciinema capture plan

We should try asciinema for documentation-quality terminal recordings while the
feature evolves. tmux is good for testable text snapshots; asciinema is better for
replayable demos that can be embedded or converted for the docs site.

Initial local experiment:

```bash
# Install if needed. Options depend on the environment.
uv tool install asciinema
# or: pipx install asciinema
# or: sudo apt install asciinema

# Start the A2A sample server in another terminal/tmux pane first.
cd /home/ssmith/source/fast-agent-pr
asciinema rec /tmp/fast-agent-a2a.cast \
  -c 'FAST_AGENT_MODEL=passthrough uv run fast-agent -x --agent-cards /tmp/a2a-card.yaml --agent hello_remote'
```

During recording, type a short scripted flow:

```text
hello
whhhhhhaaaattt
/history
/exit
```

Replay locally:

```bash
asciinema play /tmp/fast-agent-a2a.cast
```

Potential docs pipeline options to evaluate:

- keep `.cast` files as source artifacts;
- embed asciinema player in docs pages;
- convert selected recordings to GIF/SVG/video if static assets are preferred;
- pair each asciinema capture with a tmux `capture-pane` text fixture for
  regression-oriented assertions.

## Session and conversation state

A2A does have remote conversational continuity, but it is represented by
`context_id`, not by replaying fast-agent history.

Current behavior:

- fast-agent keeps normal local `message_history` for display, `/history`, and
  saved transcript behavior;
- the remote A2A agent receives only the latest user text for each request;
- remote continuity is carried through `context_id`;
- `task_id` tracks current/outstanding A2A work and is cleared on terminal states
  except input-required style flows.

Recommendation:

- treat local fast-agent history as the transcript/UI history;
- treat A2A `context_id` as remote conversation/thread state;
- do not resend the full fast-agent transcript by default;
- persist A2A state alongside sessions in a future step:
  - `context_id`;
  - `current_task_id`;
  - `last_task_state`;
  - selected transport;
  - remote card identity/version.

Open question for resume:

- On fast-agent session resume, should we always reuse the saved A2A `context_id`,
  or should there be a freshness/remote-card-version check that starts a new
  context when the old one may no longer be meaningful?

## Remaining work

### Short-term

1. Add SDK-backed tests for factory/runtime connectivity.
   - Prefer the A2A SDK server primitives over monkeypatching.
   - Cover JSON-RPC send and text aggregation.
2. Add a tmux-driven smoke test script or documented manual check.
   - Keep it optional initially if CI terminal behavior is unreliable.
3. Try asciinema capture and decide where `.cast` files should live.
4. Add `/a2a` diagnostics commands.

### `/a2a` command surface

MVP commands still worth adding:

```text
/a2a list
/a2a card [agent]
/a2a status [agent]
/a2a reset [agent]
/a2a transport [agent]
```

Later task lifecycle commands:

```text
/a2a tasks [agent]
/a2a get [agent] <task-id>
/a2a cancel [agent] <task-id>
/a2a subscribe [agent] <task-id>
/a2a resume [agent] <task-id>
```

### Transport coverage

Current practical validation is JSON-RPC. Configuration validation accepts
`HTTP+JSON` and `GRPC`, but these need explicit integration coverage.

Next transport tests:

- HTTP+JSON against the SDK sample server;
- gRPC only when optional dependencies are available and the environment is
  suitable.

### Content mapping beyond text

Current MVP is text-only:

- fast-agent user text -> A2A text part;
- A2A text messages/artifacts -> fast-agent assistant text.

Later mapping:

- `data` parts -> JSON/fenced JSON or structured side channel;
- `url` parts -> markdown links/resource references;
- `raw` parts -> media/document attachments where fast-agent can display or
  persist them;
- preserve `media_type` where possible.

### Streaming behavior

The current UI path prioritizes stable user/assistant turn rendering. It consumes
A2A events synchronously and displays the final aggregated assistant message.

Future streaming refinement:

- route artifact/direct-message text updates into the existing streaming handle;
- avoid duplicate blank headers;
- preserve post-stream re-render behavior exactly like local LLM agents;
- dedupe servers that send full artifact snapshots rather than deltas.

### Server-side fast-agent-as-A2A

Deferred until the client mapping settles.

The SDK server layer should make this straightforward later:

- implement an `AgentExecutor` that wraps an `AgentProtocol` or `AgentApp`;
- map A2A user messages to fast-agent `send()`/`generate()`;
- stream fast-agent `StreamChunk`s as A2A artifact updates;
- expose fast-agent `agent_card()` as the A2A AgentCard.

## 2026-05-20 asciinema/progress spike update

Asciinema capture is now validated as part of the local development/testing
workflow for TUI-facing A2A work.

Artifacts produced during the spike:

```text
/tmp/fast-agent-a2a-clean.cast
/tmp/fast-agent-a2a-natural.cast
/tmp/fast-agent-a2a-progress.cast
```

The most useful current demo is:

```text
/tmp/fast-agent-a2a-progress.cast
```

Replay it with:

```bash
asciinema play /tmp/fast-agent-a2a-progress.cast
```

Fast inspection:

```bash
asciinema play /tmp/fast-agent-a2a-progress.cast --speed 100
```

Recording script:

```text
/tmp/a2a-asciinema-progress.sh
```

The progress demo uses the same tmux-driven approach:

- create a fixed-size tmux session;
- disable the tmux status bar;
- start fast-agent with the A2A card;
- type input character-by-character for a more natural feel;
- leave enough delay after Enter for the SDK sample server's wait state to show;
- record the whole interaction with asciinema.

Recording command:

```bash
asciinema rec \
  --overwrite \
  --cols 104 \
  --rows 34 \
  --idle-time-limit 1.3 \
  -t 'fast-agent A2A progress display demo' \
  -c /tmp/a2a-asciinema-progress.sh \
  /tmp/fast-agent-a2a-progress.cast
```

The `.cast` file is newline-delimited JSON:

- first line: metadata (`version`, `width`, `height`, `timestamp`, `env`, `title`,
  optional `idle_time_limit`);
- remaining lines: events shaped like `[time_offset_seconds, "o", "terminal output"]`.

This makes simple edits scriptable:

- retitle recordings by rewriting the first JSON line;
- trim beginning/end events;
- redact paths or usernames;
- compress pauses by rewriting timestamps.

For docs embedding, use `asciinema-player` and serve `.cast` files as static
assets. Keep the tmux scripts as reproducible sources and the `.cast` files as
recorded documentation artifacts.

A2A progress display was also wired into the normal progress board. While waiting
for the remote A2A response the TUI now shows the standard sending row, e.g.:

```text
▎▶ hello_remote ────────────────────────────────────────────────────────────────
hello
▎▶ Sending       ⠄ hello_remote
```

Then the progress display is paused before rendering the final assistant message,
so the progress row does not overwrite the completed A2A response:

```text
▎◀ hello_remote A2A
Hello World! Nice to meet you!
```

Recommendation for ongoing A2A UI work:

- run a tmux text capture as a quick regression check;
- record or update an asciinema cast when the visible behavior changes;
- keep casts short and focused (one feature per cast);
- use `--idle-time-limit` so docs recordings remain compact;
- prefer deterministic tmux scripts over manual recordings for repeatability.
