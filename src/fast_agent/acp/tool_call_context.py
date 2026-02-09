"""ACP tool call context helpers.

ACP's ToolCall messages include an optional `_meta` map for extensibility.
We use a ContextVar to attach per-task metadata (e.g. parent tool call id)
to all tool call notifications emitted while that task
is running.

This enables clients (like Nexus ACP) to reliably group nested tool calls under
the correct Agents-as-Tools instance without changing the ACP schema.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ACPToolCallContext:
    """Per-async-task metadata to attach to ACP tool call updates."""

    parent_tool_call_id: str | None = None

    def to_meta(self) -> dict[str, Any] | None:
        meta: dict[str, Any] = {}
        if self.parent_tool_call_id:
            meta["parentToolCallId"] = self.parent_tool_call_id
        return meta or None


_acp_tool_call_context: ContextVar[ACPToolCallContext | None] = ContextVar(
    "acp_tool_call_context", default=None
)


def get_acp_tool_call_meta() -> dict[str, Any] | None:
    """Return the current `_meta` payload to attach to ACP tool call messages."""
    ctx = _acp_tool_call_context.get()
    return ctx.to_meta() if ctx else None


@contextmanager
def acp_tool_call_context(
    *,
    parent_tool_call_id: str | None = None,
):
    """Temporarily set ACP tool call context for the current async task.

    Fields are merged over any existing context (so nested contexts can override
    only a subset of fields).
    """

    current = _acp_tool_call_context.get() or ACPToolCallContext()
    merged = ACPToolCallContext(
        parent_tool_call_id=(
            parent_tool_call_id
            if parent_tool_call_id is not None
            else current.parent_tool_call_id
        ),
    )
    token = _acp_tool_call_context.set(merged)
    try:
        yield
    finally:
        _acp_tool_call_context.reset(token)
