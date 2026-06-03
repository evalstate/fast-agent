from __future__ import annotations

from fast_agent.agents.agent_types import AgentType
from fast_agent.core.function_tool_support import (
    custom_class_supports_function_tools,
    decorator_supports_scoped_function_tools,
)


class _ToolsConstructor:
    def __init__(self, *, tools: list[object] | None = None) -> None:
        self.tools = tools


class _KwargsConstructor:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _PlainConstructor:
    def __init__(self, name: str) -> None:
        self.name = name


def test_custom_class_supports_function_tools_for_tools_keyword() -> None:
    assert custom_class_supports_function_tools(_ToolsConstructor)


def test_custom_class_supports_function_tools_for_kwargs_constructor() -> None:
    assert custom_class_supports_function_tools(_KwargsConstructor)


def test_custom_class_supports_function_tools_rejects_plain_constructor() -> None:
    assert not custom_class_supports_function_tools(_PlainConstructor)


def test_decorator_supports_scoped_function_tools_uses_custom_constructor_support() -> None:
    assert decorator_supports_scoped_function_tools(
        AgentType.CUSTOM,
        custom_cls=_ToolsConstructor,
    )
    assert not decorator_supports_scoped_function_tools(
        AgentType.CUSTOM,
        custom_cls=_PlainConstructor,
    )
