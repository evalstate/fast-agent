"""Context helpers for local tool invocations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


@dataclass(frozen=True, slots=True)
class LocalToolInvocationContext:
    """Metadata for a local Python function tool invocation."""

    tool_name: str
    arguments: Mapping[str, Any]
    tool_use_id: str | None = None


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

_local_tool_invocation_context: ContextVar[LocalToolInvocationContext | None] = ContextVar(
    "local_tool_invocation_context",
    default=None,
)


def get_local_tool_invocation_context() -> LocalToolInvocationContext | None:
    """Return metadata for the currently executing local Python function tool."""

    return _local_tool_invocation_context.get()


@contextmanager
def local_tool_invocation_context(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    tool_use_id: str | None = None,
) -> Iterator[LocalToolInvocationContext]:
    """Expose tool-call metadata without adding parameters to a tool function."""

    context = LocalToolInvocationContext(
        tool_name=tool_name,
        arguments=dict(arguments),
        tool_use_id=tool_use_id,
    )
    token = _local_tool_invocation_context.set(context)
    try:
        yield context
    finally:
        _local_tool_invocation_context.reset(token)


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
