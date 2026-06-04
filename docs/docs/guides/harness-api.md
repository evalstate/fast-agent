---
title: Harness API
description: A design preview for headless, session-oriented fast-agent applications.
social:
  title: Harness API
  tagline: Run fast-agent headlessly with explicit sessions.
  description: A design preview for headless, session-oriented fast-agent applications.
  alt: fast-agent social card — Harness API
---

# Harness API

!!! warning "Design preview"

    The Harness API described here is the proposed first implementation slice for
    a Python-first headless API. It is intended to guide implementation and user
    expectations before the API lands.

The Harness API is a session-oriented way to run **fast-agent** from Python
without entering the TUI or starting a transport server.

The intended shape is:

```python
async with fast.harness(instance_scope="session") as harness:
    session = await harness.session("support-123")
    response = await session.generate("Help this customer")
```

Use it when you want to embed fast-agent in another Python application, such as
a web service, batch worker, test harness, CLI automation layer or future direct
HTTP adapter.

## Why a harness?

The existing Python API is app-oriented:

```python
async with fast.run() as app:
    text = await app.send("hello")
```

That remains unchanged.

The Harness API adds an explicit session container:

```python
async with fast.harness() as harness:
    session = await harness.session("demo")
    message = await session.generate("hello")
```

The important design point is:

> A fast-agent session is an affinity key for a full `AgentInstance`, not a
> wrapper around one agent.

A fast-agent `AgentInstance` contains the active agent map for the app. That
means a single session can contain multiple loaded agents, routers, evaluators,
orchestrators, agents-as-tools, MCP-backed agents and tool-only helper agents.

```python
review = await harness.session("pr-123", agent_name="reviewer")

# Uses the session default agent: "reviewer".
review_response = await review.generate("Review this PR")

# Same session, different target agent in the same AgentInstance.
notes = await review.generate(
    "Summarize the user-visible changes",
    agent_name="writer",
)
```

This differs from systems where a session wraps one configured agent runtime.
fast-agent keeps the multi-agent runtime model and adds sessions around it.

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
    async with fast.harness(instance_scope="session") as harness:
        session = await harness.session("customer-123", agent_name="support")

        response = await session.generate("Help this customer reset their password.")

        print(response.last_text())


if __name__ == "__main__":
    asyncio.run(main())
```

Use `parse_cli_args=False` when embedding fast-agent in another application that
owns command-line parsing.

## Creating a harness

Planned signature:

```python
async with fast.harness(
    instance_scope="session",
    model="sonnet",
) as harness:
    ...
```

Parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `instance_scope` | `"session"` | Controls how `AgentInstance` objects are shared or recreated. |
| `model` | `None` | Optional global model override, similar to the CLI `--model` override. |

The harness should use the same initialization path as `fast.run()`:

- app initialization;
- config and model loading;
- AgentCard loading;
- Agent Skill discovery and prompt injection;
- MCP server configuration;
- shell/filesystem runtime setup;
- global prompt context;
- provider-key validation.

It should not enter:

- TUI mode;
- CLI `--message` mode;
- CLI `--prompt-file` mode;
- MCP server mode;
- ACP server mode.

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

Recommended slice-1 session ID behavior:

- `None` means `"default"`;
- strings are stripped;
- empty strings are rejected;
- Python accepts any non-empty string;
- reserved namespaces such as `task:` or `branch:` are deferred until task or
  branch sessions exist.

## Explicit session management

The harness also exposes a session manager:

```python
session = await harness.sessions.get("demo")
session = await harness.sessions.create("demo")
session = await harness.sessions.get_or_create("demo")
await harness.sessions.delete("demo")
```

Recommended behavior:

| Method | Behavior |
|---|---|
| `get(name)` | Return an existing session. Raise if missing. |
| `create(name)` | Create a new session. Raise if it already exists. |
| `get_or_create(name)` | Return an existing session or create it. |
| `delete(name)` | Delete a session if present; no-op if missing. |

Session map operations should be protected by a harness-level lock so
concurrent create/get/delete calls cannot corrupt the in-memory session map.

## Calling agents from a session

Slice 1 reuses the existing fast-agent protocol methods instead of inventing a
new result model:

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

Tool-only agents are not selected as the default, but can be targeted
explicitly if the current `AgentApp` rules allow that target.

## Instance scopes

`instance_scope` controls how sessions map to `AgentInstance` objects.

| Scope | Meaning | Conversation continuity | Typical use |
|---|---|---:|---|
| `"session"` | Each `session_id` gets a stable `AgentInstance`. | Yes, per session. | Recommended default. |
| `"shared"` | All sessions use one shared `AgentInstance`. | Yes, globally shared. | Advanced shared-state use. |
| `"request"` | Each call gets a fresh `AgentInstance`. | No. | Stateless jobs and tests. |

### `instance_scope="session"`

This is the recommended default.

```python
async with fast.harness(instance_scope="session") as harness:
    a = await harness.session("customer-a")
    b = await harness.session("customer-b")

    await a.send("Remember that my name is Alice.")
    await b.send("Remember that my name is Bob.")
```

Behavior:

- each session ID gets one stable `AgentInstance`;
- the same session ID returns the same session object;
- different session IDs get different `AgentInstance` objects;
- deleting a session disposes its instance;
- histories are isolated between sessions.

### `instance_scope="shared"`

All sessions use the same primary `AgentInstance`.

```python
async with fast.harness(instance_scope="shared") as harness:
    a = await harness.session("a")
    b = await harness.session("b")
```

Behavior:

- session IDs are metadata only;
- all sessions share agent histories and MCP runtime state;
- deleting a session does not dispose the shared instance;
- concurrent calls against the shared instance should be rejected to avoid
  interleaving mutable shared state.

Use this only when shared state is intentional.

### `instance_scope="request"`

Each call creates and disposes a fresh `AgentInstance`.

```python
async with fast.harness(instance_scope="request") as harness:
    session = await harness.session("job-runner")

    first = await session.send("Remember the number 42.")
    second = await session.send("What number did I ask you to remember?")
```

The second call should not rely on the first call's in-memory history.

Use this for stateless classification, batch jobs and tests.

## Concurrency

Slice 1 should reject concurrent calls on the same `AgentSession`:

```python
session = await harness.session("customer-123")

first = asyncio.create_task(session.send("first"))

try:
    await session.send("second")
except RuntimeError as exc:
    print(exc)

await first
```

Only one operation may be active for a session at a time. This protects mutable
message histories, MCP aggregators and tool runtime state while surfacing
accidental misuse immediately.

For true parallel branches, create separate sessions:

```python
a = await harness.session("customer-a")
b = await harness.session("customer-b")

await asyncio.gather(
    a.send("Help customer A"),
    b.send("Help customer B"),
)
```

Recommended error shape:

```text
RuntimeError: Session 'customer-123' is already running send; start another session for parallel conversation branches.
```

This matches Flue's session-operation model and is simpler to reason about than
quietly queueing calls.

For `instance_scope="shared"`, different session IDs still point at the same
underlying `AgentInstance`. Slice 1 should therefore also reject concurrent
calls against the shared instance, even if they come from different
`AgentSession` objects.

## Deleting active sessions

Never dispose a session-owned `AgentInstance` while a generation is active.

Recommended slice-1 behavior is Flue-like and explicit:

- if a session has an active operation, `session.delete()` raises `RuntimeError`;
- `harness.sessions.delete(name)` delegates to the open session and follows the
  same rule;
- deleting a missing session is still a no-op.

Example error shape:

```text
RuntimeError: Session 'support-123' is running generate; wait before deleting it.
```

Deletion should not wait behind a long-running operation. The caller should wait
for or cancel the active operation in a future run API, then delete the session.

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

Recommended slice-1 semantics:

- `clear(agent_name=None)` clears the resolved default target only;
- it does not clear every agent in the session;
- a future API can add `clear_all()` if needed.

## Shell handling

Slice 1 should not add first-class `harness.shell()` or `session.shell()` methods.

Instead, shell access continues to work the way it does today: through the
agent's configured tools and runtime setup.

For example, if an agent is configured with shell access, a harness call can use
that tool through the model/tool loop:

```python
async with fast.harness(instance_scope="session") as harness:
    session = await harness.session("repo-review", agent_name="reviewer")

    response = await session.generate(
        "Inspect the current git diff and summarize risky changes."
    )
```

The shell/tool activity belongs to the selected agent in that session's
`AgentInstance`, so it is part of the normal conversation/tool execution flow.

Recommended slice-1 rules:

- the harness should initialize the same shell/filesystem runtime as `fast.run()`;
- the harness should not create a separate out-of-band shell API;
- shell availability should remain controlled by existing config, AgentCards and
  CLI/programmatic options;
- tool-mediated shell use is recorded in agent history according to existing
  agent/tool behavior;
- `instance_scope="session"` isolates the agent instances, histories and
  MCP/tool runtime objects created for each session;
- `instance_scope="shared"` shares those objects across sessions;
- `instance_scope="request"` creates fresh agent/tool runtime objects per call.

This is not the same thing as a per-session filesystem sandbox. Unless the
underlying shell/filesystem runtime is configured to create a separate sandbox
or working tree per session, sessions will normally see the same workspace/cwd.
Treat session IDs as conversation/runtime affinity keys, not security
boundaries.

For multi-user applications that execute shell or filesystem tools, prefer one
of these patterns:

- run untrusted users in separate harnesses with separate environment roots or
  sandbox backends;
- add an explicit per-user/per-session working directory policy before exposing
  shell tools;
- use `instance_scope="request"` only when statelessness is desired; it creates
  fresh agent instances, but it does not by itself create a new filesystem.

This deliberately differs from Flue, which exposes both `harness.shell()` and
`session.shell()`. If fast-agent later adds first-class shell helpers, the useful
distinction is:

| Future API | Intended behavior |
|---|---|
| `harness.shell(...)` | Out-of-band command; not recorded in conversation. |
| `session.shell(...)` | Session-affine command; recorded as part of that session's conversation/tool trace. |

Those helpers are future work, not slice 1.

## Skill handling

Agent Skills should work under the harness exactly as they work under
`fast.run()`.

When the harness starts, it should load default skills and apply them to agent
configs before creating instances:

- installed skills are discovered from the active environment and configured
  skill directories;
- agent-specific skill configuration is resolved;
- skill manifests are added to the system prompt through `{{agentSkills}}`;
- skills that request shell access should trigger the same shell runtime setup as
  normal runs;
- warnings and validation should follow existing fast-agent behavior.

Example:

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
    async with fast.harness(instance_scope="session") as harness:
        session = await harness.session("issue-492", agent_name="dev")
        response = await session.generate(
            "Use the relevant repository skills to investigate this failure."
        )
        print(response.last_text())
```

Skills are not separate session objects. They are capabilities loaded into the
agents in each `AgentInstance`.

Scope implications:

| Scope | Skill behavior |
|---|---|
| `"session"` | Each session-owned `AgentInstance` gets agents initialized with the resolved skill manifests. |
| `"shared"` | The shared primary instance has the resolved skills; all sessions use it. |
| `"request"` | Each call creates a fresh instance with the resolved skills. |

Runtime skill management, such as TUI `/skills` commands or Skills over MCP
installation flows, is outside slice 1. The harness should consume the active
skill configuration; it should not introduce a new skill-management API.

## MCP and agents-as-tools

Because a session owns a full `AgentInstance`, multi-agent workflows continue to
work inside the session.

```python
session = await harness.session("analysis-123", agent_name="manager")

response = await session.generate(
    "Analyze this issue. Use your helper agents and MCP tools if needed."
)
```

The `manager` agent can use configured child agents as tools, MCP servers, and
workflow dependencies in the same instance.

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

## Lifecycle

Always use the harness as an async context manager:

```python
async with fast.harness() as harness:
    session = await harness.session("demo")
    await session.send("hello")
```

When the context exits:

- session-owned instances are disposed;
- request-scoped temporary instances should already be disposed;
- the shared primary instance is disposed by the run finalizer;
- progress display and runtime callbacks are cleaned up;
- agent shutdown hooks run.

Session deletion inside the context:

```python
session = await harness.session("demo")
await session.delete()
```

After deletion, the session object should be treated as closed.

## Relationship to Flue

The Harness API shape is inspired by Flue's harness/session model:

```ts
const harness = await ctx.init(agent)
const session = await harness.session("support-123")
const result = await session.prompt("Help this customer")
```

The closest mapping is:

| Flue | fast-agent Harness API |
|---|---|
| `ctx.init(agent)` | `fast.harness()` |
| `FlueHarness` | `AgentHarness` |
| `harness.session(name)` | `await harness.session(session_id)` |
| `FlueSessions` | `AgentSessions` |
| `FlueSession` | `AgentSession` |
| `session.prompt()` | `session.generate()` / `session.send()` |
| `prompt(..., { result })` | `structured()` / `structured_schema()` |
| `session.task()` | future branch/task/run API |
| `CallHandle.abort()` | future cancellable run API |
| persisted `SessionStore` | future durable session store |
| direct handler over harness/session | future direct HTTP adapter |

The deliberate divergence:

| Concept | Flue | fast-agent |
|---|---|---|
| Session materialization | One configured agent runtime. | One `AgentInstance` with the full active agent map. |
| Same-session concurrent calls | Reject. | Reject in slice 1. |
| Persistence | Durable session data and provider affinity key. | In-memory sessions only in slice 1. |
| Primary prompt method | `prompt()`. | Existing `send()`, `generate()`, `structured()`, `structured_schema()`. |

This lets future direct HTTP, A2A, CLI automation and webhook surfaces become
thin adapters over the Python harness:

```python
session = await harness.session(body.session_id, agent_name=body.agent)
message = await session.generate(body.message)
return serialize(message)
```

## Future slices

Slice 1 intentionally avoids:

- durable session persistence;
- provider-facing affinity keys;
- background runs;
- cancellable call handles;
- event streams;
- first-class shell/fs helpers;
- task or branch sessions;
- direct HTTP;
- A2A refactor;
- a normalized result object.

Likely future additions:

```python
result = await session.invoke("hello")
print(result.text, result.data, result.usage, result.model)
```

```python
run = await session.start("Review this PR")
await run.cancel()
result = await run.wait()
```

```python
child = await session.branch("task-1")
result = await child.generate("Do focused work", agent_name="reviewer")
```

## Future Features

### Sandbox integration

Slice 1 should treat sessions as conversation/runtime affinity, not filesystem
or security isolation. A future sandbox integration can make that boundary
explicit.

Possible API shape:

```python
async with fast.harness(
    instance_scope="session",
    sandbox="e2b",
) as harness:
    session = await harness.session("user-123")
```

or:

```python
session = await harness.session(
    "user-123",
    sandbox={"provider": "local", "cwd": "/tmp/fast-agent/user-123"},
)
```

Open design questions:

- Should sandboxes be configured per harness, per session, or per request?
- Should `instance_scope="session"` imply a stable sandbox for that session when
  a sandbox provider is configured?
- How should MCP server lifecycles attach to sandboxed filesystem and shell
  runtimes?
- How should permission prompts and audit logs identify sandbox boundaries?
- What cleanup policy should apply to sandbox files after session deletion?

This is where direct HTTP and webhook adapters become more appropriate for
multi-tenant hosting. The adapter should be able to map an authenticated user,
tenant or request to a sandbox policy before invoking:

```python
session = await harness.session(session_id, agent_name=agent_name)
message = await session.generate(body.message)
```

### Isolated environments

Flue's default harness/session model appears optimized for a relatively
single-user or trusted-workspace use case: multiple sessions in one harness
share the harness environment, cwd and sandbox unless the caller creates a
separate sandbox/environment outside the session API.

fast-agent should be explicit about this too. Session IDs alone should not be
treated as tenant isolation.

A future isolated-environment feature could let callers choose stronger
boundaries:

```python
async with fast.harness(
    instance_scope="session",
    environment_scope="session",
) as harness:
    session = await harness.session("tenant-a/user-123")
```

Possible environment scopes:

| Scope | Meaning |
|---|---|
| `shared` | All sessions use the same configured environment root and shell/filesystem runtime. |
| `session` | Each session gets a separate environment root, cwd, auth store and sandbox/runtime attachment. |
| `request` | Each call gets a disposable environment. |

This would be a larger feature than slice 1 because fast-agent environments
contain more than conversation state:

- AgentCards and ToolCards;
- config and secrets;
- MCP server configuration;
- skills;
- shell/filesystem runtime policy;
- session history;
- auth and permission records;
- model overlays;
- plugin state.

For now, the recommended guidance is:

- use one harness for trusted single-user or trusted-team workflows;
- use separate harnesses, environment roots or process-level sandboxes for
  untrusted users;
- do not expose shell/filesystem tools to multi-tenant API callers without an
  explicit isolation layer.
