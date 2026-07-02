---
title: Harness API
description: Run fast-agent headlessly with typed, session-oriented Python APIs.
social:
  title: Harness API
  tagline: Run fast-agent headlessly with explicit sessions.
  description: Run fast-agent headlessly with typed, session-oriented Python APIs.
  alt: fast-agent social card — Harness API
---

The Harness API runs **fast-agent** from Python without entering the TUI or
starting an MCP/ACP transport server.

Use it when you want to embed fast-agent in another Python application, such as
a web service, batch worker, test harness, CLI automation layer, or adapter.

```python
async with fast.harness() as harness:
    session = await harness.session("support-123", agent_name="support")
    response = await session.generate("Help this customer")
```

!!! warning

    The Harness API is under active development and should not be considered stable.


## Mental model

The Harness API has three layers:

| Layer            | What it does                                                                               | Typical caller                                                        |
| ---------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| `AgentHarness`   | Starts the fast-agent runtime without the TUI or a protocol server.                        | Your process startup code.                                            |
| `HarnessSession` | Owns one live `AgentInstance` for a stable conversation or job.                            | Batch jobs, tests, scripts, and direct Python code.                   |
| `HarnessApp`     | Converts an application or protocol request into a harness session and exposes `invoke()`. | HTTP routes, webhooks, MCP adapters, A2A adapters, and local UI code. |

Use `harness.session(...)` when your code already knows which conversation it
wants to run. Use `harness.app().open(...)` when another boundary first needs to
choose a session and agent from request data.

```python
async with fast.harness() as harness:
    # Direct Python use: choose the session in application code.
    session = await harness.session("customer-123", agent_name="support")
    text = await session.send("Help this customer reset their password.")
```

For an HTTP route, webhook, queue consumer, or protocol adapter, the entrypoint
usually receives transport-native data first. It authenticates the caller,
derives a stable application key, then opens the app boundary:

```python
from dataclasses import dataclass

from fast_agent import AgentRequest, AppOpenRequest, FastAgent, HarnessApp


@dataclass(frozen=True)
class SupportMessage:
    ticket_id: str
    text: str


fast = FastAgent("Support service", parse_cli_args=False)


async def handle_support_message(
    app: HarnessApp,
    *,
    message: SupportMessage,
    user_id: str,
) -> str:
    session_id = f"ticket-{message.ticket_id}"

    async with app.open(
        AppOpenRequest(
            session_id=session_id,
            agent="support",
            metadata={"user_id": user_id, "ticket_id": message.ticket_id},
        )
    ) as session:
        response = await session.invoke(
            AgentRequest.text(
                message.text,
                agent="support",
                session_id=session_id,
                metadata={"source": "http", "user_id": user_id},
            )
        )

    return response.text_content()


async with fast.harness() as harness:
    # In a real web server, do this once at startup and keep `app` in
    # application state for request handlers or webhook callbacks.
    app = harness.app()

    reply = await handle_support_message(
        app,
        message=SupportMessage(ticket_id="8472", text="What is the current status?"),
        user_id="user-123",
    )
```

`AppOpenRequest` is the application-side open request. It answers "which
fast-agent session should this incoming event use?" It is where an adapter puts
the session affinity it derived from HTTP, a webhook, a queue message, an A2A
`context_id`, an MCP session id, or another external request.

`AgentRequest` is the agent-turn envelope. It carries the message plus per-turn
agent selection, auth, request parameters, metadata, and progress reporting.
Most adapters use the same value for `AppOpenRequest.session_id` and
`AgentRequest.session_id`; when they differ, document the reason because they
represent two different affinity decisions.

## Session keys from application events

Choose `session_id` from the unit of continuity in your product:

| Incoming source           | Common session key                                                         |
| ------------------------- | -------------------------------------------------------------------------- |
| authenticated chat        | user id, conversation id, or thread id                                     |
| support webhook           | ticket id or customer id                                                   |
| GitHub webhook            | issue number, pull request number, or review thread id                     |
| Slack/Teams/Discord event | channel/thread id plus workspace/team id                                   |
| queue job                 | job id when isolated, or entity id when later jobs should continue context |
| A2A server                | returned A2A `context_id`                                                  |
| MCP server                | MCP client session id, or a request-scoped id for stateless mode           |

Session IDs are runtime affinity keys, not authorization tokens. Authenticate
and authorize the external request before deriving the key. If a user can
provide a ticket id, issue number, or agent name, verify that the caller is
allowed to access it before opening the harness session.

## Harness apps

A harness app is the preferred application boundary. It opens one
`HarnessSession`, exposes the live `AgentApp` for lower-level UI code, and
provides `invoke()` for protocol and service adapters.

```python
from fast_agent import AgentRequest, AppOpenRequest


async with fast.harness() as harness:
    app = harness.app()

    async with app.open(AppOpenRequest(session_id="customer-123", agent="support")) as session:
        response = await session.invoke(
            AgentRequest.text(
                "Help this customer",
                agent="support",
                session_id="customer-123",
            )
        )
        print(response.text_content())
```

This is also the boundary used by the default CLI runtime. When `fast-agent go`
starts a local interactive session, the TUI receives the existing `AgentApp`
because it needs agent switching, slash commands, reload hooks, tool display,
MCP attach/detach flows, and session command state. One-shot
`fast-agent go --message` and `fast-agent go --prompt-file` open the same
default harness app boundary and run the turn through a `HarnessSession`.

The default MCP server uses the same boundary. `fast-agent serve` exposes one
harness app tool named `send` by default, with optional `session_id` and `agent`
arguments. `fast-agent serve --transport http` and
`fast-agent serve --transport stdio` route to that default MCP harness app
service; `fast-agent serve --transport acp` and
`fast-agent serve --transport a2a` route to their protocol-specific servers.

MCP and A2A adapters use `AgentRequest` and `AgentResponse` to keep request
metadata, auth, session affinity, and progress reporting explicit. ACP keeps its
ACP-specific session lifecycle, permissions, status-line updates, and client
terminal integration, while wrapping each agent turn in an adapter that uses the
same request/response shape.

## Custom harness apps

Configure a custom harness app with `harness_app.entrypoint`:

```yaml
harness_app:
  entrypoint: "my_app:create_app"
```

The entrypoint is a `module:function` factory. It receives a
`HarnessAppContext` with the default app and session provider. Wrap the default
app when you want to add application policy around every opened session:

```python
from fast_agent import AgentRequest, AgentResponse, AppOpenRequest
from fast_agent.core.harness_app import HarnessAppContext


class MyApp:
    def __init__(self, context: HarnessAppContext) -> None:
        self.default_app = context.default_app

    def open(self, request: AppOpenRequest | None = None):
        return MyAppSession(self.default_app.open(request))


class MyAppSession:
    def __init__(self, default_session_context) -> None:
        self.default_session_context = default_session_context
        self.session = None

    async def __aenter__(self):
        self.session = await self.default_session_context.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return await self.default_session_context.__aexit__(exc_type, exc, traceback)

    @property
    def agent_app(self):
        return self.session.agent_app

    @property
    def env(self):
        return self.session.env

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        await self.env.tools.execute("git", args=["status", "--short"])
        return await self.session.invoke(request)


def create_app(context: HarnessAppContext) -> MyApp:
    return MyApp(context)
```

Application code can also use the runtime environment directly inside an opened
session:

```python
from pathlib import Path

async with app.open(AppOpenRequest(session_id="repo-review", agent="reviewer")) as session:
    session.env.skills.add(Path(".fast-agent/skills/repo-review"), agent="reviewer")
    status = await session.env.tools.execute("git", args=["status", "--short"])
    response = await session.env.agent("reviewer").invoke(
        AgentRequest.text(
            f"Review this workspace status:\n{status.stdout}",
            session_id="repo-review",
        )
    )
```

## Session orientation

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

| Parameter | Default | Meaning                                                                |
| --------- | ------: | ---------------------------------------------------------------------- |
| `model`   |  `None` | Optional global model override, similar to the CLI `--model` override. |

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
from fast_agent import (
    AgentHarness,
    AppOpenRequest,
    DefaultHarnessApp,
    FastAgent,
    HarnessSession,
    HarnessSessions,
)


async with fast.harness() as harness:
    typed_harness: AgentHarness = harness
    sessions: HarnessSessions = harness.sessions
    session: HarnessSession = await sessions.get_or_create("demo")
    message = await session.generate("hello")
```

These classes are exported from `fast_agent` for imports such as:

```python
from fast_agent import AgentHarness, AppOpenRequest, DefaultHarnessApp, FastAgent, HarnessSession
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
- valid IDs are 1-128 characters, start and end with a letter or number, and
  contain only letters, numbers, dashes, or underscores.

!!! tip "Session naming"

    Choose short descriptive IDs such as `customer-123`, `ticket_456`, or
    `repo-review`.

    The validation rule is exactly the one above: `^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,126}[A-Za-z0-9])?$`.
    Names with spaces, slashes, dots, colons, or other punctuation are rejected.
    The slug-style rule keeps persisted session folders simple when
    `session_history` is enabled.

### Explicit session management

The harness exposes a session manager:

```python
session = await harness.sessions.get("demo")
session = await harness.sessions.create("demo")
session = await harness.sessions.get_or_create("demo")
await harness.sessions.delete("demo")
```

| Method                | Behavior                                          |
| --------------------- | ------------------------------------------------- |
| `get(name)`           | Return an existing session. Raise if missing.     |
| `create(name)`        | Create a new session. Raise if it already exists. |
| `get_or_create(name)` | Return an existing session or create it.          |
| `delete(name)`        | Delete a session if present; no-op if missing.    |

Session map operations are protected by a harness-level lock.

## Calling agents from a session

Harness sessions reuse the existing fast-agent protocol methods for direct
agent calls. The app and protocol boundary uses `AgentRequest` and
`AgentResponse`.

```python
text = await session.send("hello")
message = await session.generate("hello")
data, raw = await session.structured("classify this", MyModel)
data, raw = await session.structured_schema("classify this", schema)
response = await session.invoke(AgentRequest.text("hello"))
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

### `invoke()`

`invoke()` accepts an `AgentRequest` and returns an `AgentResponse`:

```python
from fast_agent import AgentRequest


response = await session.invoke(
    AgentRequest.text("Summarize this ticket.", agent="support", session_id="ticket-123")
)

print(response.text_content())
```

Use `invoke()` at protocol boundaries where auth, request parameters, metadata,
progress reporting, and session affinity should travel together.

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

## Compacting history

Compaction is **on by default under the harness**, exactly as under `fast.run()`.
After each completed turn, if context usage crosses `compaction.threshold`
(default `0.85`), the agent's history is automatically compacted into a
checkpoint summary; recent turns are kept verbatim. Disable it with
`compaction: {auto: false}` in config (see
[History Compaction](../../ref/config_file/#history-compaction)).

You can also compact a session on demand with `session.compact()`:

```python
result = await session.compact()
print(f"{result.messages_before} → {result.messages_after} messages "
      f"(~{result.tokens_after_estimate} tokens)")
```

Steer the summary with one-off instructions, or target a specific agent:

```python
await session.compact(instructions="Preserve the order number and the SLA.")
await session.compact(agent_name="writer")
```

`compact()` returns a `CompactionResult` (`messages_before`/`messages_after`,
`tokens_before`/`tokens_after_estimate`, `context_window`, `summary_text`,
`archive_file`). It honors `compaction.keep_turns` and `compaction.prompt` from
config, persists the compacted history, and serializes with other session
operations. It raises `CompactionSkipped` when there is nothing worth compacting
and `CompactionError` if the summarization call fails — in both cases history is
left untouched.

For lower-level use (custom triggers, building your own UI), the primitives in
`fast_agent.history.compaction` are importable directly: `compact_conversation`,
`plan_compaction` (model-call-free retention preview), `should_auto_compact`,
`estimate_tokens`, `is_compaction_message`, and `resolve_compaction_prompt`.

## Skills, MCP, and agents-as-tools

Agent Skills work under the harness the same way they work under `fast.run()`.
When the harness starts, default skills are discovered, agent-specific skill
configuration is resolved, and skill manifests are injected into prompts through
`{{agentSkills}}`.

Harness app code can also add or replace skills for the opened session's target
agent through `session.env.skills`. This is intended for application-level policy
such as "review routes always include the repository-review skill" without
changing global defaults.

```python
from fast_agent import AppOpenRequest, FastAgent

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
        app = harness.app()
        async with app.open(AppOpenRequest(session_id="issue-492", agent="dev")) as session:
            session.env.skills.add(".fast-agent/skills/repo-maintenance", agent="dev")
            response = await session.env.agent("dev").generate(
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

## Eval inspection pattern

The harness can be used directly in evals without a separate eval runner. Use a
fresh session ID for each independent case, run the turn, then inspect the
resolved agent's `message_history` with `ConversationSummary`:

```python
from fast_agent import ConversationSummary, FastAgent


fast = FastAgent("Support Bot", parse_cli_args=False, environment_dir=".fast-agent")


async with fast.harness() as harness:
    session = await harness.session("eval-checkout-status", agent_name="support")
    message = await session.generate("Is checkout currently operational?")

    agent = session.agent_app.resolve_agent("support")
    summary = ConversationSummary(messages=agent.message_history)

    assert "operational" in message.last_text().lower()
    assert summary.tool_call_map.get("get_service_status", 0) >= 1
    assert summary.tool_errors == 0
```

`ConversationSummary` is a small analysis view over the actual agent history. It
reports message counts, turn splits, tool call counts, per-tool call maps,
tool errors, and timing data when timing channels are present. For assertions
that need exact tool arguments, tool results, citations, usage channels, or
provider-specific metadata, inspect `agent.message_history` directly; it contains
the same `PromptMessageExtended` objects returned by `generate()`.

For deterministic test cases, prefer one session ID per case so saved history
cannot leak between cases. Reuse a session ID only when the eval is intentionally
checking conversation memory. When `session_history` is enabled, call
`await session.delete()` after a case if you do not want the persisted eval
session kept under `environment_dir/sessions/`.

GEPA and artifact-heavy eval loops can use the same pattern inside their scorer
or candidate evaluator, while writing candidate inputs, outputs, summaries, and
scores through `fast_agent.eval` artifact helpers.

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

Use `harness.shell()` for programmatic shell commands that should not be added
to a conversation:

```python
async with fast.harness() as harness:
    result = await harness.shell("pwd")

    if result.exit_code == 0:
        print(result.stdout)
    else:
        print(result.stderr)
```

`harness.shell()` returns a structured `ShellExecutionResult` with `stdout`,
`stderr`, and `exit_code`. It runs through the harness shell environment, but it
does not create a harness session and does not update agent history.

By default, the harness uses the local shell environment. You can inject another
environment, such as Docker, with `fast.harness(environment=...)`. See
[Agent Environments](../environments.md) for Docker examples and the
`ShellEnvironment` protocol.

Use `session.shell()` when you want shell work serialized with a specific
`HarnessSession`:

```python
async with fast.harness() as harness:
    session = await harness.session("repo-review", agent_name="reviewer")
    result = await session.shell("git diff --stat")
```

`session.shell()` also returns `ShellExecutionResult` and does not add the
command or output to chat history. It is still a session operation, so it is
rejected while the same session is already running `send()`, `generate()`,
`structured()`, or another `shell()` call.

If an agent is configured with shell access, a harness call can also use that
tool through the normal model/tool loop:

```python
async with fast.harness() as harness:
    session = await harness.session("repo-review", agent_name="reviewer")
    response = await session.generate(
        "Inspect the current git diff and summarize risky changes."
    )
```

The shell/tool activity belongs to the selected agent in that session's
`AgentInstance`, so it is part of the normal conversation and tool execution
flow. Use this when the model should decide which commands to run or when the
tool interaction should be part of the agent turn.

Filesystem access remains tool-mediated through configured agents. If the
injected shell environment also implements `SessionFilesystem`, model-facing
file tools such as `read_text_file`, `write_text_file`, `edit_file`, and
`apply_patch` use that environment filesystem.

Session IDs are conversation/runtime affinity keys, not security boundaries. A
session does not automatically create a per-session filesystem sandbox. For
multi-user applications that expose shell or filesystem tools, use separate
harnesses, environment roots, process-level sandboxes, or another explicit
isolation layer appropriate for your deployment.
