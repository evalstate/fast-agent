"""Context helpers for local tools invoked inside agent-as-tool calls."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


@dataclass(frozen=True, slots=True)
class AgentToolInvocationContext:
    """Metadata for a child agent invoked as a tool."""

    agent_name: str
    arguments: Mapping[str, Any]
    tool_name: str | None = None
    tool_use_id: str | None = None


_agent_tool_invocation_context: ContextVar[AgentToolInvocationContext | None] = ContextVar(
    "agent_tool_invocation_context",
    default=None,
)


@contextmanager
def agent_tool_invocation_context(
    *,
    agent_name: str,
    arguments: Mapping[str, Any],
    tool_name: str | None = None,
    tool_use_id: str | None = None,
) -> Iterator[AgentToolInvocationContext]:
    """Expose parent-supplied child-tool arguments during child execution."""

    context = AgentToolInvocationContext(
        agent_name=agent_name,
        arguments=dict(arguments),
        tool_name=tool_name,
        tool_use_id=tool_use_id,
    )
    token = _agent_tool_invocation_context.set(context)
    try:
        yield context
    finally:
        _agent_tool_invocation_context.reset(token)

