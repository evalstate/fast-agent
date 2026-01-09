# ACP Sessions Without ACP (In-Process Streaming)

## Summary
ACP is the protocol layer (session lifecycle, permissions, tool routing, and
client/server transport). If you only need **in-process** streaming of logs,
tool calls, and final responses, you can run fast-agent directly and consume
its event stream without ACP.

You will still need to handle **session persistence** and **routing** yourself
(per chat/topic), but the core runtime already emits structured events that you
can stream to your app or logs.

## What ACP Provides (that you lose without it)
- Session lifecycle (open/close, reconnect, permissions flow).
- A formal transport (stdin/stdout or HTTP) and capability negotiation.
- Tool permission store and UX around approvals.
- Clear separation between client and agent runtime.

If you do not need these, you can stay in-process.

## In-Process Streaming Path (no ACP)
Fast-agent already emits structured events through an **async event bus**:

- `fast_agent.core.logging.transport.AsyncEventBus`
- `fast_agent.core.logging.logger.FastAgentLogger`
- `fast_agent.core.logging.listeners.ProgressListener` (progress/streaming)
- `fast_agent.core.logging.listeners.LoggingListener` (log sink)

These events include:
- **LLM streaming updates** (progress_action = Streaming/Thinking).
- **Tool execution progress** (tool name, server name, progress/total).
- **General status** (start/finish, warnings/errors).

### Minimal integration sketch (conceptual)
1) Create/attach a listener to `AsyncEventBus`.
2) Run FastAgent / AgentApp in-process.
3) Stream events to your UI/telemetry/log pipeline.

This gives you streaming token updates + tool call progress without ACP.

## What You Must Implement Yourself
- Session registry (per Telegram chat/topic).
- History persistence (load/save message history per session).
- Any auth/permissions you want around tools.
- Routing between sessions and agent instances.

If you need long-lived sessions and reconnects, you will need your own
session store (e.g., DB/Redis or file-based history).

## Parallel Sessions (Telegram chats)
Parallel sessions are possible, but you must handle isolation explicitly.

- **Per-session state**: keep message history, active agent name, and any
  attached tools per chat/topic. Do not share one agent instance across
  concurrent chats unless you serialize access.
- **Concurrency**: each chat should have its own agent instance or task
  queue. Concurrent `send()` calls against the same agent instance can
  interleave message history.
- **Event routing**: the event bus is global. Include a `session_id` in
  event context so your listener can route logs/tool events to the right
  chat.
- **UI/progress**: the built-in TUI progress display is global; avoid
  using it for multi-chat backends and use a custom listener instead.

## Recommendation
- **Use ACP** when you need multi-client or remote transport.
- **Avoid ACP** when you want a lightweight, single-process engine with
  custom session management and direct streaming access.

If helpful, I can draft a minimal in-process session manager that:
- multiplexes multiple chat sessions,
- persists per-session history,
- fans out event-bus streaming to each client.
