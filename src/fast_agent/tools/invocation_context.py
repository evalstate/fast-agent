"""Context helpers for local tool invocations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping


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


# Nested PTC call ids embed the provider's execute_python call id so traces, ACP clients and
# child sessions can attribute nested work (such as subagent runs) to the call the model made.
NESTED_PTC_CALL_SEPARATOR = "::ptc-"

_programmatic_tool_call_id: ContextVar[str | None] = ContextVar(
    "programmatic_tool_call_id",
    default=None,
)


def programmatic_tool_call_id() -> str | None:
    """Return the id of the PTC nested call being dispatched, if any.

    Tools compare it with their own ``tool_use_id`` to know their result is consumed by
    sandbox code rather than the model; nested work inherits the variable, so a bare
    presence check would misattribute calls made by, say, a subagent's own tools.
    """

    return _programmatic_tool_call_id.get()


@contextmanager
def programmatic_tool_call(call_id: str) -> Generator[None]:
    """Mark ``call_id`` as a call made from programmatic tool calling sandbox code."""

    token = _programmatic_tool_call_id.set(call_id)
    try:
        yield
    finally:
        _programmatic_tool_call_id.reset(token)


def is_programmatic_tool_call(tool_use_id: str | None) -> bool:
    """Return whether ``tool_use_id`` identifies the PTC nested call being dispatched."""

    return tool_use_id is not None and tool_use_id == _programmatic_tool_call_id.get()


def outer_tool_call_id(call_id: str) -> str:
    """Return the model-issued call id ``call_id`` belongs to: itself unless PTC-nested."""

    outer, separator, _ = call_id.rpartition(NESTED_PTC_CALL_SEPARATOR)
    return outer if separator and outer else call_id


def get_local_tool_invocation_context() -> LocalToolInvocationContext | None:
    """Return metadata for the currently executing local Python function tool."""

    return _local_tool_invocation_context.get()


@contextmanager
def local_tool_invocation_context(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    tool_use_id: str | None = None,
) -> Generator[LocalToolInvocationContext]:
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
) -> Generator[AgentToolInvocationContext]:
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
