---
title: A2A Client
description: Use fast-agent as a client for remote Agent2Agent (A2A) agents.
---

# A2A Client

fast-agent can connect to remote A2A agents as normal fast-agent agents. A
remote A2A agent can be used from the CLI, TUI, AgentCards, or the Python API.

## CLI

Use `--a2a` for an ad hoc remote agent:

```bash
uv run fast-agent -x \
  --a2a http://127.0.0.1:41242 \
  --a2a-transport JSONRPC \
  --message "hello"
```

`--a2a` points at the remote agent base URL. fast-agent resolves the AgentCard
from `/.well-known/agent-card.json`, selects a supported transport, and sends the
message through the A2A SDK client.

When no transport is specified, fast-agent asks the SDK to use either supported
HTTP binding: `JSONRPC` or `HTTP+JSON`. Set `--a2a-transport` only when you want
to force one binding.

Use `--a2a-oauth` or `--no-a2a-oauth` to force or disable browser OAuth for an
ad hoc remote agent:

```bash
uv run fast-agent -x \
  --a2a https://research.example.com \
  --a2a-oauth \
  --message "hello"
```

Supported HTTP transports:

| Canonical | Useful aliases |
|---|---|
| `JSONRPC` | `jsonrpc`, `json-rpc`, `rpc` |
| `HTTP+JSON` | `http`, `http+json`, `rest` |

gRPC is not part of fast-agent's A2A support target.

### CLI Recording

This recording shows the expected shape of a streamed remote A2A response from
the deterministic fake server.

<div class="a2a-terminal-demo">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/asciinema-player.css">
  <link rel="stylesheet" href="../../assets/vendor/asciinema-player/catppuccin.css">
  <div id="a2a-client-cli-player"></div>
</div>

<script src="../../assets/vendor/asciinema-player/asciinema-player.min.js"></script>
<script>
  (function () {
    function renderClientCliCast() {
      var target = document.getElementById("a2a-client-cli-player");
      if (!target || !window.AsciinemaPlayer || target.dataset.loaded === "true") {
        return;
      }
      target.dataset.loaded = "true";
      window.AsciinemaPlayer.create("../../assets/a2a/a2a-client-cli.cast", target, {
        cols: 96,
        rows: 18,
        preload: true,
        speed: 1,
        idleTimeLimit: 1,
        fit: "width",
        theme: "fast-agent-dark"
      });
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", renderClientCliCast);
    } else {
      renderClientCliCast();
    }
    if (window.document$ && window.document$.subscribe) {
      window.document$.subscribe(renderClientCliCast);
    }
  })();
</script>

## AgentCard

Use a checked-in AgentCard when the remote A2A agent should be reusable:

```yaml
type: a2a
name: research_remote
url: https://research.example.com
transport: JSONRPC
```

Then run:

```bash
uv run fast-agent -x --agent-cards ./agents --agent research_remote
```

A2A cards also support:

```yaml
streaming: true
polling: false
accepted_output_modes:
  - text/plain
  - application/json
  - image/*
request_timeout_seconds: 120
headers:
  Authorization: "Bearer ${A2A_TOKEN}"
auth:
  oauth: true
  persist: keyring
relative_card_path: "/.well-known/agent-card.json"
```

For Hugging Face URLs (`hf.co`, `huggingface.co`, and `*.hf.space`),
fast-agent automatically applies the local Hugging Face token as A2A request
headers when no explicit auth header is configured. For Spaces this includes
both `Authorization` and `X-HF-Authorization`.

When a remote AgentCard advertises OAuth2 or OpenID Connect security schemes,
fast-agent can reuse the existing browser OAuth flow. If `auth` is omitted, the
A2A client enables that flow only for OAuth/OIDC cards. Set `auth.oauth: false`
to disable browser OAuth, or `auth.oauth: true` to allow OAuth challenge handling
even before the card requires it. The flow uses the same local callback,
paste-URL fallback, client metadata URL, and keyring storage behavior as MCP URL
connections.

## TUI

Inside the interactive prompt, connect a remote A2A agent at runtime:

```text
/a2a connect http://127.0.0.1:41242 --transport JSONRPC --name research_remote
```

Use `--oauth` or `--no-oauth` to force or disable browser OAuth for a runtime
connection:

```text
/a2a connect https://research.example.com --oauth --name research_remote
```

## Python API

Use `A2ARemoteAgent` directly when constructing agents in code:

```python
from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.config import MCPServerAuthSettings

remote_agent = A2ARemoteAgent(
    config=AgentConfig(name="research_remote", agent_type=AgentType.A2A),
    a2a_config=A2AAgentConfig(
        url="https://research.example.com",
        transport="JSONRPC",
        auth=MCPServerAuthSettings(oauth=True),
    ),
)
await remote_agent.initialize()
```

Useful diagnostics:

```text
/a2a list
/a2a status [agent]
/a2a card [agent]
/a2a transport [agent]
/a2a reset [agent]
```

`/a2a status` shows the current A2A `context_id`, pending `task_id`, last task
state, and selected client transport.

When the local A2A AgentCard or request has `use_history: false`, fast-agent
starts each completed turn with a fresh A2A context. The exception is
`TASK_STATE_INPUT_REQUIRED`: fast-agent keeps the returned `task_id` and
`context_id` so the next user message can continue the interrupted task.

## Streaming

Remote A2A `TaskArtifactUpdateEvent` updates are emitted through the normal
fast-agent stream listener path. The client assembles final text per artifact
and honors the A2A `append` flag, so replacement updates replace the artifact
content and append updates extend it.

The A2A client defaults to a longer HTTP request timeout than httpx's default so
real LLM-backed servers have time to emit the first stream event. Set
`request_timeout_seconds` on an A2A AgentCard when a remote endpoint needs a
different timeout.

### Real LLM Server Recording

This recording shows a fast-agent A2A client streaming from a fast-agent A2A
server backed by a real LLM and the Hugging Face MCP server. It is a provider
smoke recording, separate from the deterministic fake-server recordings used by
the test suite.

<div class="a2a-terminal-demo">
  <div id="a2a-real-llm-hf-player"></div>
</div>

<script>
  (function () {
    function renderRealLlmHfCast() {
      var target = document.getElementById("a2a-real-llm-hf-player");
      if (!target || !window.AsciinemaPlayer || target.dataset.loaded === "true") {
        return;
      }
      target.dataset.loaded = "true";
      window.AsciinemaPlayer.create("../../assets/a2a/a2a-real-llm-hf-streaming.cast", target, {
        cols: 120,
        rows: 32,
        preload: true,
        speed: 1,
        idleTimeLimit: 1,
        fit: "width",
        theme: "fast-agent-dark"
      });
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", renderRealLlmHfCast);
    } else {
      renderRealLlmHfCast();
    }
    if (window.document$ && window.document$.subscribe) {
      window.document$.subscribe(renderRealLlmHfCast);
    }
  })();
</script>

Regenerate this provider-backed cast with:

```bash
uv run scripts/a2a_docs_pipeline.py record-real-llm
```

## `INPUT_REQUIRED`

When a remote A2A task reaches `TASK_STATE_INPUT_REQUIRED`, fast-agent:

- returns a normal `PromptMessageExtended` assistant turn with
  `stop_reason=LlmStopReason.PAUSE`;
- keeps the pending A2A `task_id`;
- preserves the returned A2A `context_id`;
- sends the next user message back to the same task.

Use `/a2a reset` to clear the pending task and start a fresh remote context.

### Turn Continuation Recording

This recording shows the task id being retained only while the remote task is in
`TASK_STATE_INPUT_REQUIRED`; after the follow-up completes, the task id is
cleared and the context id remains available for future turns.

<div class="a2a-terminal-demo">
  <div id="a2a-client-input-required-player"></div>
</div>

<script>
  (function () {
    function renderClientInputRequiredCast() {
      var target = document.getElementById("a2a-client-input-required-player");
      if (!target || !window.AsciinemaPlayer || target.dataset.loaded === "true") {
        return;
      }
      target.dataset.loaded = "true";
      window.AsciinemaPlayer.create("../../assets/a2a/a2a-client-input-required.cast", target, {
        cols: 96,
        rows: 18,
        preload: true,
        speed: 1,
        idleTimeLimit: 1,
        fit: "width",
        theme: "fast-agent-dark"
      });
    }
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", renderClientInputRequiredCast);
    } else {
      renderClientInputRequiredCast();
    }
    if (window.document$ && window.document$.subscribe) {
      window.document$.subscribe(renderClientInputRequiredCast);
    }
  })();
</script>

## Attachments

The A2A client maps fast-agent prompt content to A2A parts:

| fast-agent content | A2A part |
|---|---|
| `TextContent` | `Part(text=...)` |
| `ResourceLink` | `Part(url=...)` |
| `ImageContent` | `Part(raw=..., mediaType=image/...)` |
| `AudioContent` | `Part(raw=..., mediaType=audio/...)` |
| `EmbeddedResource` with JSON `TextResourceContents` | `Part(data=...)` |

Remote URL, data, raw, and text response parts are rendered into fast-agent
assistant output. See [Protocol Compliance](protocol-compliance.md) for current
partial multimodal gaps.
