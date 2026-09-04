"""Optional JSONL diagnostics for interactive prompt and streaming cancellation flows."""

from __future__ import annotations

import os
from typing import Any

from fast_agent.core.runtime_diagnostics import write_runtime_trace

_TRACE_ENV_VAR = "FAST_AGENT_INTERACTIVE_DEBUG_TRACE"


def write_interactive_trace(event: str, **fields: Any) -> None:
    """Append an interactive diagnostic record when tracing is enabled.

    The trace is opt-in via ``FAST_AGENT_INTERACTIVE_DEBUG_TRACE=/path/to/file.jsonl``.
    Failures are intentionally swallowed so diagnostics never affect runtime behavior.
    """
    trace_path_raw = os.getenv(_TRACE_ENV_VAR, "").strip()
    if not trace_path_raw:
        return

    write_runtime_trace(trace_path_raw, event, **fields)
