# Single Runtime, One Agent per Chat (In-Process)

## Goal
Run one FastAgent runtime in-process and maintain one **long-lived agent instance per chat/topic**, each with isolated history and independent streaming to a custom UI (no TUI).

## Baseline Model
- One process, one FastAgent runtime.
- One chat/topic = one dedicated agent instance (created once, reused).
- Per-chat history is isolated and never shared across agents.
- Requests to the same agent are serialized (no concurrent send() per agent).

## Can This Work Today?
Yes, but you must provide the session layer yourself:
- Fast-agent does **not** provide per-chat session management in-process.
- There is **no built-in session_id** in the event stream by default.
- The default TUI/progress display is global, not per session.

## Session Isolation Requirements
### 1) One agent instance per chat
- Do **not** reuse a single agent across multiple chats.
- If you need 100+ chats, maintain 100+ agent instances.

### 2) Per-chat history
- Each agent holds its own `message_history`.
- Store history per chat in your own DB or file store.
- On restart: load history for the chat before first send.

### 3) Parallel calls without locks (spawn-per-request)
If you want to avoid per-chat locks, you can adopt the Agents-as-Tools pattern:

- Treat the chat history as the **source of truth** (per chat/topic).
- For each request, spawn a **fresh agent clone**, load history, run the call,
  then **merge back** any new messages on completion (similar to
  `fork_and_merge` semantics).
- This allows parallel calls per chat without reusing a shared in-memory agent.

## Event Routing (Streaming)
You want streaming logs/tool events per chat. There are two options:

### Option A (no code changes): encode session in agent name
- Create agent names like `agent__chat_12345`.
- Event stream already includes `agent_name`, so you can route by name.
- Downside: noisy names in UI/logs; tool names derived from agent name.

### Option B (small code change): add `session_id` to event context
- Add a `session_id` to event data when emitting events.
- Route in your custom listener by `session_id`.
- Requires touching `FastAgentLogger` or the call sites where events are emitted.

## Disable TUI / Progress Display
Yes, and you can still **route logger events to each chat**:
- Do not run `fast-agent go` (skip the TUI).
- Run the runtime programmatically (FastAgent / AgentApp).
- Keep `LoggerSettings.show_chat=True` and `LoggerSettings.show_tools=True` if you
  want those events available (they only gate console rendering).
- Set `LoggerSettings.progress_display=False` to avoid the global Rich UI, then
  attach your own listener to `AsyncEventBus` and render per chat.

If you need streaming token updates, either:
- implement a custom listener that mirrors `ProgressListener`/`convert_log_event`, or
- keep `progress_display=True` and accept the global progress output (not ideal
  for multi-chat backends).

## Custom Panel per Chat
Possible, but not built-in. The current progress display is global.
Two approaches:

1) **Custom listener per chat**
   - Subscribe to the global event bus.
   - Filter by `agent_name` or `session_id`.
   - Push events into your own UI (Telegram, web, etc.).

2) **Refactor display layer** (larger change)
   - Make progress display instance-scoped instead of global.
   - Requires changes in `fast_agent.ui` and `ProgressListener` wiring.

## Minimal Implementation Outline
1) Maintain a `dict[chat_id, history_store]` in your app.
2) On message:
   - Spawn a fresh agent instance from the shared AgentCard.
   - Load history for `chat_id` into the clone.
   - `agent.send(message)`.
   - Merge new messages back into the history store.
3) Subscribe to event bus and route streaming by agent_name/session_id.

## Code Sketch: Load AgentCards and Clone an Agent per Request
This example shows one runtime that loads AgentCards from a directory, then
spawns a fresh clone per chat request and merges history back.

```python
import asyncio

from fast_agent import FastAgent
from fast_agent.types import PromptMessageExtended


async def handle_message(
    app,
    base_agent_name: str,
    chat_id: str,
    history: list[PromptMessageExtended],
    message: str,
) -> str:
    base = app[base_agent_name]
    clone = await base.spawn_detached_instance(name=f"{base.name}[{chat_id}]")
    clone.load_message_history(history)
    response = await clone.send(message)
    history[:] = clone.message_history
    return response


async def main() -> None:
    fast = FastAgent(config_path="fastagent.config.yaml")
    fast.load_agents("agents")  # directory with AgentCards

    chat_histories: dict[str, list[PromptMessageExtended]] = {}
    async with fast.run() as app:
        chat_id = "chat-123"
        history = chat_histories.setdefault(chat_id, [])
        reply = await handle_message(app, "vertex-rag", chat_id, history, "ping")
        print(reply)


if __name__ == "__main__":
    asyncio.run(main())
```

## Known Risks
- Global event bus and registries are shared; isolation is logical, not physical.
- Many spawned agents can be heavy if each opens MCP connections.
- Large histories per chat can grow memory. Use compaction or replay windows.

## Summary
You can run one FastAgent runtime with **per-chat history** and
**spawn-per-request agents** (no locks) by treating the chat history as the
canonical state and cloning per call. This mirrors Agents-as-Tools behavior
and keeps concurrency simple at the cost of more instance churn.

If you want, I can sketch a minimal in-process session manager with a queue per chat and a custom event listener.
