"""Runtime LLM tracing state."""

from __future__ import annotations

import os

_OVERRIDE: bool | None = None


def llm_trace_enabled() -> bool:
    if _OVERRIDE is not None:
        return _OVERRIDE
    return bool(os.environ.get("FAST_AGENT_LLM_TRACE"))


def set_llm_trace_enabled(enabled: bool) -> None:
    global _OVERRIDE
    _OVERRIDE = enabled
    if enabled:
        os.environ["FAST_AGENT_LLM_TRACE"] = "1"
    else:
        os.environ.pop("FAST_AGENT_LLM_TRACE", None)


def toggle_llm_trace() -> bool:
    enabled = not llm_trace_enabled()
    set_llm_trace_enabled(enabled)
    return enabled
