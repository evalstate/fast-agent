"""Helpers for determining which agents support local function tools."""

from __future__ import annotations

import inspect
from typing import Any

from fast_agent.agents.agent_types import AgentType


def _callable_accepts_keyword_arg(callable_obj: Any, arg_name: str) -> bool:
    """Return whether a callable accepts a named keyword argument."""
    if callable_obj is None:
        return False

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    if arg_name in signature.parameters:
        return True

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def custom_class_supports_function_tools(cls: Any) -> bool:
    """Return whether a custom agent class accepts ``tools=`` during construction.

    ``function_tools`` is the user-facing configuration name, but custom agent
    classes receive the resolved tool objects via a constructor kwarg named
    ``tools`` for compatibility with existing ``ToolAgent``/``McpAgent`` APIs.
    """
    init = getattr(cls, "__init__", None)
    return _callable_accepts_keyword_arg(init, "tools")


def decorator_supports_scoped_function_tools(
    agent_type: AgentType,
    *,
    custom_cls: Any = None,
) -> bool:
    """Return whether a decorated agent function should expose ``.tool``."""
    if agent_type in {AgentType.BASIC, AgentType.SMART}:
        return True
    if agent_type == AgentType.CUSTOM:
        return custom_class_supports_function_tools(custom_cls)
    return False
