---
title: Harness API
description: Run fast-agent headlessly with typed, session-oriented Python APIs.
social:
  title: Harness API
  tagline: Run fast-agent headlessly with explicit sessions.
  description: Run fast-agent headlessly with typed, session-oriented Python APIs.
  alt: fast-agent social card — Harness API
---

# Harness API

The Harness API runs **fast-agent** from Python without entering the TUI or
starting an MCP/ACP transport server.

Use it when you want to embed fast-agent in another Python application, such as
a web service, batch worker, test harness, CLI automation layer, or adapter.

```python
async with fast.harness() as harness:
    session = await harness.session("support-123", agent_name="support")
    response = await session.generate("Help this customer")
```

The harness is session-oriented:

- each harness session owns one stable `AgentInstance` for that session's lifetime;
- the same session ID returns the same `HarnessSession` object;
- different session IDs get isolated `AgentInstance` objects;
- deleting a session disposes its instance;
- exiting the harness context disposes all remaining session instances.

A fast-agent session is an affinity key for a full `AgentInstance`, not a wrapper
around one agent. The instance contains the active agent map for the app:
regular agents, routers, evaluators, orchestrators, agents-as-tools,
MCP-backed agents, and tool-only helper agents.

## Quick start

```python
import asyncio

from fast_agent import FastAgent


fast = FastAgent("Support Bot", parse_cli_args=False)


@fast.agent(
    "support",
    instruction="You are a concise customer support assistant.",
    model="sonnet",
)
async def main() -> None:
    async with fast.harness() as harness:
        session = await harness.session("customer-123", agent_name="support")
        response = await session.generate("Help this customer reset their password.")
        print(response.last_text())


if __name__ == "__main__":
    asyncio.run(main())
```

Use `parse_cli_args=False` when embedding fast-agent in an application that owns
command-line parsing.

## Creating a harness

```python
async with fast.harness(model="sonnet") as harness:
    ...
```

| Parameter | Default | Meaning |
|---|---:|---|
| `model` | `None` | Optional global model override, similar to the CLI `--model` override. |

The harness uses the same initialization path as `fast.run()`:

- app initialization;
- config and model loading;
- AgentCard loading from the active environment's `agent-cards/` directory;
- Agent Skill discovery and prompt injection;
- MCP server configuration;
- shell/filesystem runtime setup;
- global prompt context;
- provider-key validation.

It does not enter:

- TUI mode;
- CLI `--message` mode;
- CLI `--prompt-file` mode;
- MCP server mode;
- ACP server mode.

If the active environment contains AgentCards, the harness loads them before
validating that agents exist:

```text
.fast-agent/
  agent-cards/
    support.md
```

```python
fast = FastAgent("Support Bot", parse_cli_args=False, environment_dir=".fast-agent")

async with fast.harness() as harness:
    session = await harness.session("customer-123", agent_name="support")
```

Use `fast.load_agents(path)` when you want to load AgentCards from an additional
non-environment path.

## Typing and IDE autocomplete

The public API uses concrete, typed classes:

```python
from fast_agent import AgentHarness, FastAgent, HarnessSession, HarnessSessions


async with fast.harness() as harness:
    typed_harness: AgentHarness = harness
    sessions: HarnessSessions = harness.sessions
    session: HarnessSession = await sessions.get_or_create("demo")
    message = await session.generate("hello")
```

These classes are exported from `fast_agent` for imports such as:

```python
from fast_agent import AgentHarness, FastAgent, HarnessSession
```

## Sessions

Get or create a session with `harness.session()`:

```python
session = await harness.session("demo")
```

If no ID is supplied, the default session ID is `"default"`:

```python
session = await harness.session()
assert session.id == "default"
```

Session ID behavior:

- `None` means `"default"`;
- strings are stripped;
- empty strings raise `ValueError`;
- use 1-128 character slug-style IDs made from letters, numbers, dashes, and
  underscores, starting and ending with a letter or number;
- reserved namespaces such as `task:` or `branch:` are future work.

!!! tip "Session naming"

    Treat harness session IDs as stable, human-readable keys. Prefer IDs such
    as `customer-123`, `ticket_456`, or `repo-review`. When `session_history`
    is enabled, the ID is also the persisted folder name under
    `environment_dir/sessions/`, so avoid path-like names, spaces, punctuation,
    or values that need escaping.

### Explicit session management

The harness exposes a session manager:

```python
session = await harness.sessions.get("demo")
session = await harness.sessions.create("demo")
session = await harness.sessions.get_or_create("demo")
await harness.sessions.delete("demo")
```

| Method | Behavior |
|---|---|
| `get(name)` | Return an existing session. Raise if missing. |
| `create(name)` | Create a new session. Raise if it already exists. |
| `get_or_create(name)` | Return an existing session or create it. |
| `delete(name)` | Delete a session if present; no-op if missing. |

Session map operations are protected by a harness-level lock.

## Calling agents from a session

Harness sessions reuse the existing fast-agent protocol methods and return
types. There is no new result wrapper.

```python
text = await session.send("hello")
message = await session.generate("hello")
data, raw = await session.structured("classify this", MyModel)
data, raw = await session.structured_schema("classify this", schema)
```

### `send()`

`send()` returns plain text:

```python
text = await session.send("Summarize this ticket.")
print(text)
```

### `generate()`

`generate()` returns a `PromptMessageExtended`:

```python
message = await session.generate("Summarize this ticket.")

print(message.last_text())
print(message.stop_reason)
print(message.channels)
```

Use `generate()` when an adapter or application needs the richer assistant
message rather than only text.

### `structured()`

Use `structured()` with a Pydantic model:

```python
from pydantic import BaseModel


class TicketTriage(BaseModel):
    priority: str
    category: str
    needs_human: bool


data, raw = await session.structured(
    "Classify this support ticket.",
    TicketTriage,
)

if data is not None:
    print(data.priority)
else:
    print(raw.last_text())
```

### `structured_schema()`

Use `structured_schema()` with a JSON Schema dictionary:

```python
schema = {
    "type": "object",
    "properties": {
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "reason": {"type": "string"},
    },
    "required": ["risk", "reason"],
    "additionalProperties": False,
}

data, raw = await session.structured_schema(
    "Return deployment risk metadata.",
    schema,
)
```

## Agent selection

Each call resolves the target agent in this order:

1. explicit per-call `agent_name`;
2. the session `default_agent_name`;
3. the app default agent.

```python
session = await harness.session("pr-123", agent_name="reviewer")

# Uses the session default: reviewer.
await session.generate("Review this PR")

# Overrides the session default for this call only.
await session.generate(
    "Write release notes",
    agent_name="writer",
)
```

Tool-only agents are not selected as defaults, but explicit targeting is allowed
when the existing `AgentApp` rules allow that target.

## Session lifecycle

A session owns one stable `AgentInstance` until it is deleted or the harness
context exits.

```python
async with fast.harness() as harness:
    a = await harness.session("customer-a")
    b = await harness.session("customer-b")

    await a.send("Remember that my name is Alice.")
    await b.send("Remember that my name is Bob.")
```

Behavior:

- the same session ID returns the same session object;
- different session IDs get different `AgentInstance` objects;
- histories, MCP aggregators, and tool runtime objects are isolated between sessions;
- when `session_history` is enabled, the harness creates or loads
  `environment_dir/sessions/<session_id>/`;
- persisted history for the same session ID is hydrated when a new harness
  process starts;
- deleting a session disposes its `AgentInstance`;
- deleting a session removes its persisted session folder when persistence is enabled;
- the harness context disposes any remaining session instances on exit.

For example:

```python
fast = FastAgent("Support Bot", parse_cli_args=False, environment_dir=".fast-agent")

async with fast.harness() as harness:
    session = await harness.session("customer-123", agent_name="support")
    await session.send("Remember this customer prefers email.")
```

creates:

```text
.fast-agent/
  sessions/
    customer-123/
      session.json
      history_support.json
```

Running the program again with the same `session_id` loads that persisted
history before the next turn. Set `session_history: false` in config or use
`noenv=True` to disable persistence.

Delete a session explicitly when you are done with it:

```python
session = await harness.session("demo")
await session.delete()
```

After deletion, the `HarnessSession` object is closed. Get or create the same ID
again to start a fresh session.

## Concurrency

The harness rejects concurrent operations on the same `HarnessSession`.

```python
import asyncio

session = await harness.session("customer-123")

first = asyncio.create_task(session.generate("first"))

try:
    await session.generate("second")
except RuntimeError as exc:
    print(exc)

await first
```

Only one operation may be active for a session at a time. This protects mutable
message histories, MCP aggregators, and tool runtime state while surfacing
accidental misuse immediately.

Expected error shape:

```text
RuntimeError: Session 'customer-123' is already running generate; start another session for parallel conversation branches.
```

For independent parallel branches, create separate sessions:

```python
a = await harness.session("customer-a")
b = await harness.session("customer-b")

await asyncio.gather(
    a.send("Help customer A"),
    b.send("Help customer B"),
)
```

Deleting an active session also raises:

```text
RuntimeError: Session 'support-123' is running generate; wait before deleting it.
```

Deletion does not wait behind a long-running operation.

## Clearing history

`clear()` clears the resolved target agent's history:

```python
await session.clear()
```

To clear a specific agent in the session:

```python
await session.clear(agent_name="writer")
```

To also clear applied prompts:

```python
await session.clear(clear_prompts=True)
```

`clear(agent_name=None)` clears only the resolved default target. It does not
clear every agent in the session.

## Skills, MCP, and agents-as-tools

Agent Skills work under the harness the same way they work under `fast.run()`.
When the harness starts, default skills are discovered, agent-specific skill
configuration is resolved, and skill manifests are injected into prompts through
`{{agentSkills}}`.

```python
fast = FastAgent(
    "Developer Assistant",
    parse_cli_args=False,
    skills_directory=".fast-agent/skills",
)


@fast.agent(
    "dev",
    instruction="""
You help with repository maintenance.

Available skills:
{{agentSkills}}
""",
    model="sonnet",
)
async def main() -> None:
    async with fast.harness() as harness:
        session = await harness.session("issue-492", agent_name="dev")
        response = await session.generate(
            "Use the relevant repository skills to investigate this failure."
        )
        print(response.last_text())
```

Because a session owns a full `AgentInstance`, multi-agent workflows continue to
work inside the session:

```python
session = await harness.session("analysis-123", agent_name="manager")

response = await session.generate(
    "Analyze this issue. Use your helper agents and MCP tools if needed."
)
```

The selected agent can use configured child agents as tools, MCP servers, and
workflow dependencies in the same session-owned instance.

## Request parameters

Pass `RequestParams` to any call:

```python
from fast_agent import RequestParams


response = await session.generate(
    "Give me a concise answer.",
    request_params=RequestParams(maxTokens=1024),
)
```

Per-call request parameters are passed to the selected agent method.

## Shell and filesystem tools

The harness does not add first-class `harness.shell()` or `session.shell()`
methods. Shell and filesystem access remains tool-mediated through configured
agents.

If an agent is configured with shell access, a harness call can use that tool
through the normal model/tool loop:

```python
async with fast.harness() as harness:
    session = await harness.session("repo-review", agent_name="reviewer")
    response = await session.generate(
        "Inspect the current git diff and summarize risky changes."
    )
```

The shell/tool activity belongs to the selected agent in that session's
`AgentInstance`, so it is part of the normal conversation and tool execution
flow.

Session IDs are conversation/runtime affinity keys, not security boundaries. A
session does not automatically create a per-session filesystem sandbox. For
multi-user applications that expose shell or filesystem tools, use separate
harnesses, environment roots, process-level sandboxes, or another explicit
isolation layer appropriate for your deployment.

## Deployment concerns and future slices

The Python Harness API does not expose server-style instance scoping. Shared
runtimes, per-request runtimes, connection affinity, tenant isolation, sandbox
policy, direct HTTP handlers, and A2A adapters are deployment/adapter concerns
for later work.

Current non-goals include:

- provider-facing affinity keys;
- background runs;
- cancellable call handles;
- event streams;
- first-class shell/filesystem helper methods;
- task or branch sessions;
- direct HTTP or A2A adapters;
- sandbox or tenant-isolation policy;
- a normalized result object.

For a single shared in-process application runtime without harness sessions,
continue to use `fast.run()`.
