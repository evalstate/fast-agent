# A2A examples

## Harness adapter server

`server.py` shows the lower-level adapter shape: A2A protocol requests are
translated into `AgentRequest` calls through `FastAgent.harness()`, then returned
as A2A task artifacts.

Run it:

```bash
uv run python examples/a2a/server.py
```

The server exposes:

- AgentCard: `http://localhost:9999/.well-known/agent-card.json`
- JSON-RPC: `http://localhost:9999/a2a/jsonrpc`
- HTTP+JSON: `http://localhost:9999/a2a/rest`

## Streaming facts server

`facts_server.py` is the fast-agent equivalent of the ADK facts sample. It serves
a default `facts_agent` over A2A and streams model output to A2A clients.

Run it:

```bash
GOOGLE_API_KEY=... uv run python examples/a2a/facts_server.py
```

Defaults:

- `HOST=0.0.0.0`
- `PORT=8001`
- `MODEL`/`FAST_AGENT_MODEL=gemini25`

The server exposes:

- AgentCard: `http://localhost:8001/.well-known/agent-card.json`
- JSON-RPC: `http://localhost:8001/a2a/jsonrpc`
- HTTP+JSON: `http://localhost:8001/a2a/rest`

Test it with fast-agent as an A2A client:

```bash
uv run fast-agent go \
  --a2a http://localhost:8001 \
  --a2a-transport JSONRPC \
  --message "Tell me three surprising facts about octopuses."
```

Streaming is handled by `fast.start_server(transport="a2a")`: fast-agent stream
listeners are converted into A2A `TaskArtifactUpdateEvent` updates.

## Card-based facts server

The same agent can be served without a Python wrapper using the AgentCard in
`agent-cards/facts.md`:

```bash
GOOGLE_API_KEY=... uv run fast-agent serve a2a \
  --host 0.0.0.0 \
  --port 8001 \
  --name facts-a2a \
  --agent-cards examples/a2a/agent-cards/facts.md
```

Override the model from the CLI if desired:

```bash
uv run fast-agent serve a2a \
  --port 8001 \
  --name facts-a2a \
  --agent-cards examples/a2a/agent-cards/facts.md \
  --model gemini25
```

Here `facts-a2a` is the served A2A system name, while `facts` is the fast-agent
AgentCard/skill name. Keeping them distinct avoids the name clash/confusion with
the ADK sample's `facts_agent`.

## Research intake server

`research/server.py` demonstrates an A2A-native research workflow using the
same harness API shape as the MCP adapter. The refiner and worker are defined
as AgentCards in `research/.fast-agent/agent-cards/`; `fast.harness()` loads
them from that example-local environment at startup, and
`ResearchA2AHarnessAdapter` calls them by name. The executor then exits through
one of two explicit A2A actions:

- `request_further_refinement` returns a standalone A2A `Message`;
- `begin_research` creates a standard A2A `Task`, then streams progress through
  task artifact events.

Run it locally:

```bash
uv run python examples/a2a/research/server.py
```

Try the refinement path:

```bash
uv run fast-agent go \
  --a2a http://localhost:8002 \
  --a2a-transport JSONRPC \
  --message "research recent A2A work"
```

Try the task path:

```bash
uv run fast-agent go \
  --a2a http://localhost:8002 \
  --a2a-transport JSONRPC \
  --message "Research new multimodal models for scientists as an HTML report."
```

The intake message is plain text. `Goal:`, `Audience:`, and `Output:` labels
are accepted, but they are not required.

A2A `context_id` is passed as the harness `session_id`, so follow-up refinement
and the eventual research task share the same fast-agent session. Harness
session files are stored under `research/.fast-agent/sessions/`.

For a Hugging Face Space deployment with OAuth, set:

```bash
FAST_AGENT_SERVE_OAUTH=huggingface
FAST_AGENT_OAUTH_RESOURCE_URL=https://<space-owner>-<space-name>.hf.space
```

Then run the same server command in the Space. The served AgentCard advertises
Hugging Face bearer auth and fast-agent forwards the caller token into request
context.
