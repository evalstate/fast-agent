---
title: A2A Getting Started
description: Connect fast-agent to remote Agent2Agent (A2A) servers or expose fast-agent agents through A2A.
---

# A2A Getting Started

fast-agent supports [Agent2Agent (A2A)](https://a2a-protocol.org/) in two
directions:

- **Client:** connect to a remote A2A endpoint and use it like a normal
  fast-agent agent.
- **Server:** expose a fast-agent app as an A2A HTTP endpoint.

Use this page to choose the right path. The detailed guides are split by role:

- [A2A Client](client.md): `--a2a`, checked-in `type: a2a` AgentCards, TUI
  commands, remote contexts, tasks, OAuth, attachments, and `INPUT_REQUIRED`.
- [A2A Server](server.md): `fast-agent serve a2a`, served AgentCards, skills,
  instance scope, streaming, refinement messages, Harness API, examples, and
  Hugging Face bearer auth.
- [A2A API](api.md): direct Python APIs and raw A2A HTTP request shapes.
- [Protocol Compliance](protocol-compliance.md): supported A2A 1.0 surface and
  limitations.

!!! warning

    A2A support is evolving. Pin fast-agent versions for production deployments
    and review the [Protocol Compliance](protocol-compliance.md) page when
    upgrading.

## Quick Client Check

The fastest way to call a remote A2A agent is `--a2a`:

```bash
uv run fast-agent -x \
  --a2a http://127.0.0.1:41242 \
  --a2a-transport JSONRPC \
  --message "hello"
```

`--a2a` points at the remote agent base URL. fast-agent resolves
`/.well-known/agent-card.json`, selects a supported HTTP transport, and sends the
message through the A2A SDK client.

For a repeatable local demo, start the included fake A2A server:

```bash
--8<-- "docs/docs/a2a/snippets/start-fake-server.sh"
```

Then follow [A2A Client](client.md) for task updates, attachments, TUI commands,
AgentCards, and task continuation.

## Quick Server Check

The fastest way to expose fast-agent over A2A is `fast-agent serve a2a`:

```bash
uv run fast-agent serve a2a \
  --host 127.0.0.1 \
  --port 41241 \
  --agent-cards ./agents \
  --model codexresponses.gpt-5.4-mini
```

The server exposes:

| Endpoint | URL |
|---|---|
| AgentCard | `http://127.0.0.1:41241/.well-known/agent-card.json` |
| JSON-RPC | `http://127.0.0.1:41241/a2a/jsonrpc` |
| HTTP+JSON | `http://127.0.0.1:41241/a2a/rest` |

The served endpoint is one A2A remote agent or agentic system. The fast-agent
default agent handles inbound A2A messages and can orchestrate other loaded
agents internally.

For server behavior, instance scope, Harness API usage, and the research-intake
example, see [A2A Server](server.md).

## Examples

The repository contains A2A examples under `examples/a2a/`:

| Example | Purpose |
|---|---|
| `examples/a2a/facts_server.py` | Minimal Python wrapper that serves a normal fast-agent facts agent over A2A. |
| `examples/a2a/agent-cards/facts.md` | Card-based facts agent for `uv run fast-agent serve a2a --agent-cards ...`. |
| `examples/a2a/server.py` | Low-level Harness adapter example that manually wires A2A SDK routes to `FastAgent.harness()`. |
| `examples/a2a/research/server.py` | A2A-native research intake server with standalone refinement `Message` and started research `Task` paths. |

The research example uses an example-local fast-agent environment at
`examples/a2a/research/.fast-agent/`. Its refiner and worker are AgentCards in
`.fast-agent/agent-cards/`, loaded automatically by `fast.harness()` at server
startup.
