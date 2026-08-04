---
title: A2A Server
description: Deploy fast-agent agents as an Agent2Agent (A2A) HTTP server.
---

# A2A Server

Use `fast-agent serve a2a` to expose a fast-agent app through A2A HTTP
transports. `fast-agent serve --transport a2a` remains supported for parity with
the generic MCP/ACP serve command. The configured fast-agent app is initialized
first, then the A2A server routes ordinary protocol messages into the
fast-agent default agent. A2A serving binds to `127.0.0.1` by default; pass
`--host 0.0.0.0` only when remote clients should connect.

## Start a Server

```bash
uv run fast-agent serve a2a \
  --host 127.0.0.1 \
  --port 41241 \
  --instance-scope shared \
  --agent-cards ./agents \
  --model codexresponses.gpt-5.4-mini
```

The server exposes:

| Endpoint | URL |
|---|---|
| AgentCard | `http://127.0.0.1:41241/.well-known/agent-card.json` |
| JSON-RPC | `http://127.0.0.1:41241/a2a/jsonrpc` |
| HTTP+JSON | `http://127.0.0.1:41241/a2a/rest` |

The AgentCard advertises `JSONRPC` and `HTTP+JSON` with protocol version `1.0`.
gRPC is intentionally not advertised.

When the server binds to a wildcard host such as `0.0.0.0` or `::`, the served
AgentCard builds interface URLs from the incoming AgentCard request host. This
keeps cards fetched from another machine routable to the server instead of
advertising the bind wildcard or the server's loopback address.

`fast-agent serve a2a` warns for remote binds. It also warns whenever `--shell`
is enabled, with stronger wording when shell access is exposed to remote
callers.

## Card Recording

This recording shows the expected shape when a wildcard-bound server is fetched
through a routable hostname. The JSON-RPC and HTTP+JSON interfaces use the
request hostname in the served card.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/a2a/a2a-server-card.cast"
  data-fa-asciinema-cols="104"
  data-fa-asciinema-rows="20"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1"
  data-fa-asciinema-fit="width"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

For static deployment checks, bind with the concrete hostname or address that
remote clients should use.

## Runtime Wiring

The served A2A agent uses the normal fast-agent runtime. AgentCards, MCP servers,
tools, skills, hooks, model settings, and workflow agents are loaded through the
same path used by the CLI and TUI before the A2A server starts.

That means an A2A request can drive a full fast-agent bundle behind one A2A
agent boundary: an orchestrator, router, tool-using agent, MCP-backed agent, or
AgentCard-loaded group.

AgentCards in the active fast-agent home are loaded before the harness or
server runtime starts. For example, cards in `.fast-agent/agent-cards/` are
available to `fast.harness()` and to `fast-agent serve a2a` without an explicit
`fast.load_agents(...)` call. Use `fast.load_agents(path)` only when a Python
program intentionally loads cards from a non-environment location.

## Server Shapes

Choose the smallest server shape that fits the behavior you need:

| Shape | Use when |
|---|---|
| `fast-agent serve a2a` | You want to serve a normal fast-agent app over A2A. This is the default path for AgentCard-defined agents, MCP servers, workflows, and ordinary streaming. |
| Python wrapper with `fast.start_server(transport="a2a")` | You want a small script that defines agents in Python and starts the standard A2A server. See `examples/a2a/facts_server.py`. |
| Custom A2A executor plus `FastAgent.harness()` | You need protocol-specific behavior, such as returning a standalone A2A `Message` before creating a task, custom task progress, or explicit A2A SDK route wiring. See `examples/a2a/research/server.py` and `examples/a2a/server.py`. |

The custom executor shape should still use AgentCards for normal agent
definitions where possible. The Python A2A entrypoint should own protocol
routing, not duplicate agent definitions that belong in cards.

## Design Notes from A2A Examples

The A2A tutorials and samples use `AgentExecutor` as the protocol boundary. An
executor reads the A2A `RequestContext`, decides whether to return a direct
`Message` or create a `Task`, and publishes protocol events to the A2A
`EventQueue`.

The useful patterns for fast-agent are:

| Pattern | A2A shape | fast-agent server shape |
|---|---|---|
| Simple one-turn answer | Return one direct `Message`. | `card+serve` should keep this lightweight when no task is started. |
| Work with progress | Return a `Task`, then status/artifact updates. | Map Harness tool/LLM/hook progress to `TaskStatusUpdateEvent`. |
| Final task output | Send a `TaskArtifactUpdateEvent`, then complete. | Once a task exists, return generated output as artifacts. |
| Ambiguous request | Return a pre-task `Message` or an `INPUT_REQUIRED` task status. | Use pre-task messages for intake/refinement; use `INPUT_REQUIRED` after task creation. |
| Multi-turn work | Continue with the same `task_id` and `context_id`. | Use A2A `context_id` as Harness session affinity and preserve resumable task state. |
| Long-running/disconnected client | Register push notifications and retrieve the task later. | Wire A2A push config stores/senders only when callback policy is explicit. |

The LangGraph tutorial is the closest reference model for a richer default
executor. It maps A2A `context_id` to the LangGraph thread id, emits semantic
status messages such as "Looking up the exchange rates..." while tools run,
returns the final answer as an artifact, and returns `INPUT_REQUIRED` when the
model decides it needs more information. A fast-agent Harness-backed A2A server
should provide the same shape without requiring every user to write a custom
executor.

The important design consequence is that A2A status message text is a product
surface. Harness developers, hook providers, MCP/function tool providers, and
AgentCard policy should be able to influence which tool-loop and model-loop
events become `TaskStatusUpdateEvent.status.message` values.

## Agent Skills in the A2A Card

A2A models the served endpoint as one remote agent or agentic system. A2A
`AgentSkill` entries are advertised capabilities for that remote agent; they are
not a standard routing table and do not make multiple directly addressable
agents at the same endpoint. A2A `AgentSkill` is also separate from fast-agent
"skills" on disk.

fast-agent advertises loaded user-facing fast-agent agents as A2A skills so
clients can understand the capabilities available behind the endpoint:

```json
{
  "id": "researcher",
  "name": "researcher",
  "description": "Research and summarize source material.",
  "tags": ["fast-agent", "basic"],
  "examples": ["Hello"],
  "inputModes": ["text/plain", "application/json", "application/octet-stream", "image/*"],
  "outputModes": ["text/plain", "application/json", "application/octet-stream", "image/*"]
}
```

The generated skill list comes from the user-facing fast-agent agents loaded at
server startup. The skill `id` and `name` are derived from the fast-agent agent
name. The description uses the agent's configured `description` when present,
otherwise fast-agent generates a generic description. Tags include `fast-agent`
and the fast-agent agent type.

Messages route to the fast-agent default agent by default. That default agent
should orchestrate, delegate, or call helper agents internally just as it would
in normal fast-agent use.

For fast-agent-to-fast-agent integrations, the server also accepts a
fast-agent-specific routing extension in message metadata:

```json
{
  "metadata": {
    "agent": "researcher"
  }
}
```

`fast_agent_agent` is accepted as an equivalent metadata key. This metadata is
not portable A2A behavior; generic A2A clients should treat skills as capability
metadata and send normal messages to the endpoint.

AgentCard examples are generic. Mode lists describe server-wide MIME-style
content support rather than deriving per-agent modality declarations from
fast-agent AgentCard metadata or installed fast-agent skills.

## Instance Scope, Sessions, and Resumption

A2A `contextId` is optional on inbound messages. If a client omits it, the A2A
SDK generates one. The server still returns and tracks the resolved A2A
`context_id`; how that maps to fast-agent runtime instances depends on
`--instance-scope`:

| Scope | Behavior |
|---|---|
| `shared` | Use the primary fast-agent instance for all A2A contexts. This is the default for `fast-agent serve a2a`, matching the generic serve default. |
| `connection` | Use the A2A `context_id` as the server-side instance/session key. The same `context_id` reuses the same fast-agent instance; a new `context_id` creates a fresh instance. |
| `request` | Create and dispose a fresh fast-agent instance for every A2A message. |

The served agent's history setting controls how much prior conversation is sent
to the model inside whichever instance scope is selected. It does not change the
A2A protocol `context_id`.

For `INPUT_REQUIRED`, clients should continue with the returned `task_id` and
`context_id`. `shared` and `connection` scopes preserve in-memory fast-agent
state for follow-up turns. `request` scope intentionally creates a fresh
fast-agent instance for each message, so it is best for stateless agents.

The server uses in-memory A2A task storage and in-memory fast-agent context
instances. Restarting the process loses A2A task state and session continuity.

Custom Harness-backed A2A servers choose their own session mapping. The research
example maps A2A `context_id` directly to harness `session_id`, so a refinement
reply and the later research task share one fast-agent session. In that example,
session files are stored under
`examples/a2a/research/.fast-agent/sessions/`; both research AgentCards use
`use_history: false`, so the session identity is stable without accumulating
model chat history.

## Streaming

A2A streaming is task lifecycle streaming, not token-by-token assistant-message
streaming. A streaming A2A response may contain a `Task`,
`TaskStatusUpdateEvent`, and `TaskArtifactUpdateEvent` values. A direct A2A
`Message` is returned as one message event.

In the standard fast-agent A2A server, ordinary fast-agent responses from
tool-capable agents can remain direct A2A messages when the agent does not call
`start_task()` or `return_artifact()`. Once a task is started, response content
is returned through A2A artifacts and the task is completed through the standard
A2A task status flow.

## Refinement Messages

A2A-native server code can return a standalone A2A `Message` before starting a
task. This is useful for research-intake agents that first refine a vague user
message into a concrete research goal.

```python
from fast_agent.a2a.task_api import return_message, start_task

if needs_refinement:
    await return_message("Please clarify the scope and audience for the research.")
    return

await start_task("Research task accepted")
```

Tool-capable fast-agent agents served over A2A get `return_message`,
`start_task`, and `return_artifact` as local tools. For those agents, fast-agent
defers task creation until the agent starts work. Ordinary served agents keep
the default task-generating behavior.

For custom A2A executors, prefer the Harness API shape used by the MCP adapter:
translate the incoming protocol request into an `AgentRequest`, invoke an agent
through `FastAgent.harness()`, then translate the result back into A2A events.
The research example follows this pattern:

```python
async with fast.harness() as harness:
    request_handler = DefaultRequestHandler(
        agent_executor=ResearchA2AExecutor(ResearchA2AHarnessAdapter(harness)),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card(host=HOST, port=PORT),
    )
```

Inside the adapter, the A2A `context_id` is used as the harness `session_id` so
the refinement message and the eventual research task share a session:

```python
response = await harness.invoke(
    AgentRequest.text(
        prompt,
        agent="research_refiner",
        session_id=context.context_id,
        metadata={"transport": "a2a", "phase": "research_refinement"},
    )
)
```

The example lives at `examples/a2a/research/server.py`. Its refiner and worker
are normal AgentCards in
`examples/a2a/research/.fast-agent/agent-cards/`, loaded automatically when
`fast.harness()` starts. The A2A executor is responsible only for the
protocol-level decision: return a standalone `Message` for refinement guidance,
or create a `Task` and stream progress artifacts.

### Research Intake Harness Entry Point

A research-intake A2A server should keep the product policy in a small
Harness-backed adapter while leaving the refiner and researcher as ordinary
AgentCards.

The desired flow is:

1. Receive an A2A message.
2. Open the Harness app with `session_id=context.context_id`.
3. Invoke a `research-goal-refinement` agent.
4. Let the refiner answer capability questions or call one of a small set of
   decision tools.
5. If the request needs more detail, return a standalone A2A `Message`.
6. If the request is well formed, create an A2A `Task`, invoke the research
   agent in the same Harness session, publish inner-loop progress as task status
   updates, and return final outputs as artifacts.

The refiner should not communicate its decision through fragile free-form JSON.
Prefer local function tools with typed arguments, for example:

```python
async def request_refinement(message: str) -> dict[str, str]:
    """Ask the client for more detail before starting a research task."""
    return {"kind": "needs_refinement", "message": message}


async def accept_research_goal(goal: str, title: str | None = None) -> dict[str, str | None]:
    """Accept a well-formed research goal and start task execution."""
    return {"kind": "accepted", "goal": goal, "title": title}
```

The Harness adapter can attach those tools to the refiner, capture the selected
decision, and translate it to A2A:

```python
class ResearchIntakeAdapter:
    def __init__(self, app: HarnessApp) -> None:
        self.app = app

    async def handle(self, context: RequestContext, event_queue: EventQueue) -> None:
        session_id = context.context_id
        async with self.app.open(AppOpenRequest(session_id=session_id)) as session:
            decision = await self.refine(session, context)
            if decision.needs_refinement:
                await send_refinement_message(event_queue, context, decision.message)
                return

            updater = await start_research_task(event_queue, context, "Research task accepted")
            await self.prepare_research_environment(session, context, decision.goal)
            response = await session.invoke(
                AgentRequest.text(
                    decision.goal,
                    agent="researcher",
                    session_id=session_id,
                    metadata={
                        "transport": "a2a",
                        "phase": "research_task",
                        "a2a_task_id": context.task_id or "",
                    },
                )
            )
            await updater.add_artifact(
                parts=[Part(text=response.text_content())],
                name="research-result",
                last_chunk=True,
            )
            await updater.complete()

    async def prepare_research_environment(
        self,
        session: HarnessAppSession,
        context: RequestContext,
        goal: str,
    ) -> None:
        """Attach task-scoped resources before the researcher runs."""
```

`prepare_research_environment()` is where server-specific setup belongs. For a
Hugging Face hosted researcher, that may mean creating or resolving a
task-scoped bucket, making it available to the researcher's sandbox execution
environment, and recording the bucket/task relationship so final artifacts can
be attached to the A2A task.

The runnable research example implements this with
`FAST_AGENT_RESEARCH_HF_BUCKET=<namespace>/<bucket>`. When set, each accepted
A2A task gets a fresh `HuggingFaceSandboxEnvironment` mounted at `/workspace`
against a task-scoped bucket prefix. Without that setting, the example keeps the
research worker in the shared harness environment so the protocol flow can be
tested locally without Hugging Face credentials.

The helper functions above are ordinary A2A SDK event publishing code. For
example, `send_refinement_message()` enqueues one agent `Message`, while
`start_research_task()` enqueues the initial `Task`, creates a `TaskUpdater`,
and sends a working status update. The standard fast-agent A2A server wraps this
kind of event publishing behind `fast_agent.a2a.task_api`; custom executors can
use either that helper layer or the SDK primitives, but should avoid mixing both
unless they deliberately install the task context.

The serving entry point should be thin:

```python
fast = FastAgent(
    "research A2A server",
    home=Path(".fast-agent"),
    parse_cli_args=False,
    quiet=True,
)


async def main() -> None:
    async with fast.harness() as harness:
        app = harness.app()
        request_handler = DefaultRequestHandler(
            agent_executor=ResearchIntakeExecutor(ResearchIntakeAdapter(app)),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card(),
        )
        # Add AgentCard, JSON-RPC, and HTTP+JSON routes as in the standard server.
```

If the resource setup should apply to every session rather than only this A2A
adapter, implement it as a custom `harness_app.entrypoint` wrapper. The A2A
executor can then use `harness.app()` normally and let the Harness app own
session preparation.

## Push Notifications and Callbacks

A2A callback-style delivery is modeled as push notifications. The server should
advertise `capabilities.pushNotifications: true` only when the A2A SDK
`DefaultRequestHandler` has both a `PushNotificationConfigStore` and a
`PushNotificationSender`.

`AgentA2AServer` exposes that SDK wiring for Python callers:

```python
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
)
import httpx

push_config_store = InMemoryPushNotificationConfigStore()
server = AgentA2AServer(
    primary_instance=primary_instance,
    create_instance=create_instance,
    dispose_instance=dispose_instance,
    push_config_store=push_config_store,
    push_sender=BasePushNotificationSender(
        httpx_client=httpx.AsyncClient(),
        config_store=push_config_store,
    ),
)
```

Production servers should apply normal outbound-webhook controls before enabling
push notifications: validate or allowlist callback URLs, authenticate outbound
requests according to the client's push config, and use a durable store when
tasks need to survive process restarts. The built-in CLI server keeps push
notifications disabled by default.

## `INPUT_REQUIRED`

When a fast-agent response has:

```python
stop_reason = LlmStopReason.PAUSE
```

the A2A server reports `TASK_STATE_INPUT_REQUIRED` with the response text as the
status message. The task remains resumable. Clients should send the follow-up
message with the same A2A `task_id` and `context_id`.

## Errors

The server maps common fast-agent outcomes into A2A states:

| fast-agent outcome | A2A state |
|---|---|
| normal response | `TASK_STATE_COMPLETED` |
| `LlmStopReason.PAUSE` | `TASK_STATE_INPUT_REQUIRED` |
| provider credential error | `TASK_STATE_AUTH_REQUIRED` |
| cancellation | `TASK_STATE_CANCELED` |
| unexpected exception | `TASK_STATE_FAILED` |

Transport validation errors, task lookup errors, non-cancelable tasks, and
unsupported push notification operations are handled by the A2A SDK request
handler.

## Hugging Face Bearer Auth

Set `FAST_AGENT_SERVE_OAUTH=huggingface` before starting `fast-agent serve a2a`
to require bearer authentication on `/a2a/jsonrpc` and `/a2a/rest` while keeping
the public AgentCard discoverable.

The A2A server middleware accepts both header names when they reach the app:

```text
Authorization: Bearer <token>
X-HF-Authorization: Bearer <token>
```

The bearer token is validated against Hugging Face before any A2A task reaches
fast-agent agents or tools. A non-empty placeholder such as `Bearer test` is
rejected; it is not treated as authentication.

For Space-hosted A2A endpoints, clients should use `Authorization` through
`--auth`, explicit AgentCard headers, or OAuth. `X-HF-Authorization` is the
ambient fast-agent CLI policy for ordinary Space app calls; it is not a
substitute for endpoint bearer auth unless the deployment ingress passes that
header through to the app. The server advertises an `hf_bearer` HTTP bearer
security scheme in the AgentCard and stores the inbound token in fast-agent
request context while the agent runs, allowing Hugging Face Inference Provider
models and Hugging Face MCP/tools to use the caller credential.

If `FAST_AGENT_SERVE_OAUTH` is not set, the server does not gate inbound A2A
requests and any `HF_TOKEN` configured in the Space environment is used as the
server's own Hugging Face credential. Use that only for trusted/private
deployments or with tightly scoped service tokens.

See [Host A2A on Hugging Face](host-on-hf.md) for a Space-oriented setup.

## File Parts

Incoming raw image parts become `ImageContent`. Other raw file parts become
`EmbeddedResource` values with `BlobResourceContents`, preserving the base64 file
payload, MIME type, and filename-like attachment URI for the fast-agent agent.
When a fast-agent response includes a blob resource, the server emits it back to
A2A clients as a raw file part.

## Structured JSON

A2A supports structured JSON exchange through JSON-compatible data content and
also allows JSON to be returned as text artifacts. fast-agent does not parse
ordinary model text and guess that it should become protocol data. Instead, it
maps `TextResourceContents` with `mime_type="application/json"` to A2A data
parts. This gives API users and structured-output wrappers an explicit path to
return protocol-level JSON while preserving normal markdown/text responses.

See [Protocol Compliance](protocol-compliance.md) for the full supported surface
and limitations.
