"""Optional JSONL diagnostics for interactive prompt and streaming cancellation flows."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_TRACE_ENV_VAR = "FAST_AGENT_INTERACTIVE_DEBUG_TRACE"


def write_interactive_trace(event: str, **fields: Any) -> None:
    """Append an interactive diagnostic record when tracing is enabled.

    The trace is opt-in via ``FAST_AGENT_INTERACTIVE_DEBUG_TRACE=/path/to/file.jsonl``.
    Failures are intentionally swallowed so diagnostics never affect runtime behavior.
    """
    trace_path_raw = os.getenv(_TRACE_ENV_VAR, "").strip()
    if not trace_path_raw:
        return

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None

    payload = {
        "ts": time.time(),
        "mono": time.monotonic(),
        "event": event,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "task": task.get_name() if task is not None else None,
        "task_cancelling": task.cancelling() if task is not None else None,
        **fields,
    }

    try:
        trace_path = Path(trace_path_raw).expanduser()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    except Exception:
        pass
