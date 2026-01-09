# Fix AsyncEventBus Pending Task Warning

## Summary
`pytest` reports: `Task was destroyed but it is pending!` for
`AsyncEventBus._process_events`. Logs show the bus starts on one event loop and
stops on another, so the task is left pending when the original loop closes.

## Evidence (debug log)
From `/home/strato-space/fast-agent/fastagent.jsonl`:
- `start` uses `loop_id=127972778484144`
- `stop.begin` uses `loop_id=127972965207888`

This indicates the singleton event bus is created on loop A, but shutdown runs
on loop B (pytest creates a new loop per test).

## Root Cause
`AsyncEventBus` is a process-wide singleton. Under pytest with per-test event
loops, the bus task is bound to the loop that existed when logging was first
configured. Later tests run cleanup on a different loop, so cancellation happens
outside the original loop and the task remains pending.

## Debug Logging Added (temporary)
Enabled via `FAST_AGENT_EVENTBUS_DEBUG=1`. Output path can be overridden with
`FAST_AGENT_EVENTBUS_DEBUG_PATH` (defaults to `/home/strato-space/fast-agent/fastagent.jsonl`).

### Instrumentation points
- `AsyncEventBus.start()` logs `start` (running, loop_id, has_task, queue_ready).
- `AsyncEventBus.start()` after task creation logs `start.task` (task_id, loop_id).
- `AsyncEventBus.stop()` logs `stop.begin` and `stop.done` (running, loop_id, has_task, task_done).
- `AsyncEventBus.emit()` logs `emit.skipped` when `running=False`.
- `AsyncEventBus._process_events()` logs `process.start` (loop_id, task_id).


## Proposed Fixes (choose one)
1) **Per-loop bus instance**  
   Track the loop when `start()` is called. If `stop()` runs on a different
   loop, schedule cancellation on the original loop (or recreate a new bus for
   the new loop and reset the old one safely).

2) **Reset bus on loop change**  
   In `LoggingConfig.configure`, if the current loop differs from the bus loop,
   call `AsyncEventBus.reset()` and start a fresh bus on the current loop.

3) **Test-only workaround**  
   Use a session-scoped event loop fixture in tests so start/stop occur on the
   same loop. This removes the warning but does not fix the root cause.

## Files to modify for the fix
- `src/fast_agent/core/logging/transport.py` (store loop, ensure stop cancels on the correct loop)
- `src/fast_agent/core/logging/logger.py` (reset bus when loop changes before configure/start)
- `tests/conftest.py` or specific tests (optional: stabilize loop scope if needed)

## Tests executed
```bash
uv run pytest \
  tests/unit/fast_agent/agents/workflow/test_agents_as_tools_agent.py \
  tests/unit/fast_agent/core/test_agents_as_tools_function_tools.py
```

## Likely Files Touched
- `src/fast_agent/core/logging/transport.py` (AsyncEventBus: store loop, safe stop)
- `src/fast_agent/core/logging/logger.py` (reconfigure bus on loop change)
- `tests/conftest.py` or specific tests (optional test-only loop scope)
- `tests/unit/fast_agent/agents/workflow/test_agents_as_tools_agent.py`
- `tests/unit/fast_agent/core/test_agents_as_tools_function_tools.py`

## Notes
I added temporary debug logging behind `FAST_AGENT_EVENTBUS_DEBUG=1` to write
start/stop/loop IDs to `fastagent.jsonl`. This can be reverted after the fix.

### Debug logging diff (appendix)
```diff
diff --git a/src/fast_agent/core/logging/transport.py b/src/fast_agent/core/logging/transport.py
@@
+import os
+from datetime import datetime, timezone
@@
+_EVENTBUS_DEBUG = os.getenv("FAST_AGENT_EVENTBUS_DEBUG") == "1"
+_EVENTBUS_DEBUG_PATH = os.getenv(
+    "FAST_AGENT_EVENTBUS_DEBUG_PATH",
+    "/home/strato-space/fast-agent/fastagent.jsonl",
+)
+
+def _eventbus_debug(event: str, **data: object) -> None:
+    if not _EVENTBUS_DEBUG:
+        return
+    payload = {
+        "level": "DEBUG",
+        "timestamp": datetime.now(timezone.utc).isoformat(),
+        "namespace": "fast_agent.core.logging.eventbus",
+        "message": event,
+        "data": data,
+    }
+    try:
+        with open(_EVENTBUS_DEBUG_PATH, "a", encoding="utf-8") as handle:
+            handle.write(f"{json.dumps(payload, ensure_ascii=False)}\n")
+    except Exception:
+        return
@@
     async def start(self) -> None:
         ...
+        _eventbus_debug("start", running=..., loop_id=..., has_task=..., queue_ready=...)
@@
+        _eventbus_debug("start.task", task_id=..., loop_id=...)
@@
     async def stop(self) -> None:
+        _eventbus_debug("stop.begin", running=..., loop_id=..., has_task=..., task_done=...)
@@
+        _eventbus_debug("stop.done", running=..., loop_id=..., has_task=...)
@@
     async def emit(self, event: Event) -> None:
         if not self._running:
+            _eventbus_debug("emit.skipped", running=..., event_type=..., namespace=...)
             return
@@
     async def _process_events(self) -> None:
+        _eventbus_debug("process.start", loop_id=..., task_id=...)
```