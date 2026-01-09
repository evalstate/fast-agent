## FastAgent Multi-Instance Notes (Per-Chat Sessions)

### Goal
Run many independent FastAgent instances (100+ Telegram chats/topics) with the
same MCP config and AgentCards, inside a single process (repo `call`).

### Summary (what can go wrong)
Fast-agent is optimized for **one process-wide runtime** with shared global
state. Spawning many FastAgent instances inside the same process can cause:
- Global config/context collisions.
- Shared logging/progress/TUI state bleeding across sessions.
- Shared skill/server registries and caches.
- Shutdown issues (shared async tasks across event loops).

If you need strong isolation per chat/topic, prefer **process isolation**
or a **single FastAgent runtime** with per-session state handled outside.

---

### Shared / Singleton State (process-wide)

**1) Global settings**
- `fast_agent.config._settings` is a module-level singleton (`get_settings`,
  `update_global_settings`).
- One instance can override settings for all others.
- File: `src/fast_agent/config.py`

**2) Global context**
- `fast_agent.context._global_context` is stored when `Core.initialize()`
  uses `store_globally=True` (default).
- Context holds `server_registry`, `skill_registry`, `task_registry`,
  `executor`, and `acp` pointers.
- File: `src/fast_agent/context.py`, `src/fast_agent/core/core_app.py`

**3) Logging / progress / console**
- `LoggingConfig` and its event bus are process-wide.
- `progress_display` and `console` are global singletons.
- One instance can stop logging for all.
- Files: `src/fast_agent/core/logging/*`,
  `src/fast_agent/ui/progress_display.py`,
  `src/fast_agent/ui/console.py`

**4) TUI global state (interactive mode)**
- `enhanced_prompt` keeps globals: `available_agents`, `agent_histories`,
  `in_multiline_mode`, `help_message_shown`.
- Multiple REPLs in one process will conflict.
- File: `src/fast_agent/ui/enhanced_prompt.py`

**5) OpenTelemetry**
- Global tracer provider is configured once per process.
- Multiple instances will overwrite settings.
- File: `src/fast_agent/context.py`

**6) MCP caches and registries**
- Prompt cache, server registry, skill registry are shared in the global context.
- Changes in one instance may affect others.
- Files: `src/fast_agent/mcp/mcp_aggregator.py`,
  `src/fast_agent/context.py`

---

### What this means for 100+ Telegram chats

- **Config + skills**: per-instance overrides will leak.
- **Logging**: sessions will interleave and shutdown is fragile.
- **REPL/TUI**: not safe to run multiple REPLs in one process.
- **Async tasks**: shared event-bus tasks across loops can trigger
  "Task was destroyed but it is pending!" warnings.

---

### Recommended patterns

**A) Single FastAgent runtime + per-chat state**
- Run one FastAgent and keep per-chat message history in your app layer.
- Use ACP sessions if possible; map chat_id -> session_id.

**B) Process isolation for hard separation**
- One FastAgent process per chat/topic (or per cohort).
- Avoids global collisions at the cost of higher overhead.

**C) If you must create many instances in one process**
- Avoid REPL/TUI code paths.
- Initialize context once; do not call `Core.initialize()` repeatedly.
- Do not call `update_global_settings` per chat.
- Expect shared logging and caches.

---

### Practical next steps (for `call`)
- Decide if you can use ACP session isolation or need per-process isolation.
- If you need hard isolation, prefer a supervisor that spawns one
  FastAgent per chat/topic (or small pool).
- If you want one shared runtime, keep per-chat history in your layer
  and treat FastAgent as a stateless executor.
