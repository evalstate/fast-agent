"""Tests for the @fast.tool and @agent.tool decorators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.agents.agent_types import ScopedFunctionToolConfig
from fast_agent.core.direct_decorators import DecoratorMixin
from fast_agent.core.exceptions import AgentConfigError

if TYPE_CHECKING:
    from fastmcp.tools import FunctionTool


class _FakeFastAgent(DecoratorMixin):
    """Minimal host providing the attributes DecoratorMixin expects."""

    def __init__(self):
        self.agents: dict = {}
        self._registered_tools: list[FunctionTool] = []


# ---------------------------------------------------------------------------
# Helper to create an agent-decorated function via _FakeFastAgent
# ---------------------------------------------------------------------------
def _make_agent(fast: _FakeFastAgent, name: str = "test_agent"):
    """Return the decorated async function from ``@fast.agent(name=...)``."""

    @fast.agent(name=name, instruction="test")
    async def _agent_fn():
        pass

    return _agent_fn


def _make_custom(fast: _FakeFastAgent, cls, name: str = "custom_agent"):
    """Return the decorated async function from ``@fast.custom(...)``."""

    @fast.custom(cls, name=name, instruction="test")
    async def _agent_fn():
        pass

    return _agent_fn


class TestToolDecoratorBare:
    def test_bare_decorator_registers_tool(self):
        fast = _FakeFastAgent()

        @fast.tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        assert len(fast._registered_tools) == 1
        assert fast._registered_tools[0].name == "greet"

    def test_bare_decorator_uses_docstring_as_description(self):
        fast = _FakeFastAgent()

        @fast.tool
        def ping() -> str:
            """Check connectivity."""
            return "pong"

        assert fast._registered_tools[0].description == "Check connectivity."

    def test_bare_decorator_returns_original_function(self):
        fast = _FakeFastAgent()

        @fast.tool
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5


class TestToolDecoratorParameterized:
    def test_custom_name(self):
        fast = _FakeFastAgent()

        @fast.tool(name="sum_numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert fast._registered_tools[0].name == "sum_numbers"

    def test_custom_description(self):
        fast = _FakeFastAgent()

        @fast.tool(description="Add two integers together")
        def add(a: int, b: int) -> int:
            return a + b

        assert fast._registered_tools[0].description == "Add two integers together"

    def test_custom_name_and_description(self):
        fast = _FakeFastAgent()

        @fast.tool(name="my_add", description="Custom addition")
        def add(a: int, b: int) -> int:
            return a + b

        tool = fast._registered_tools[0]
        assert tool.name == "my_add"
        assert tool.description == "Custom addition"

    def test_parameterized_decorator_returns_original_function(self):
        fast = _FakeFastAgent()

        @fast.tool(name="multiply")
        def mul(a: int, b: int) -> int:
            return a * b

        assert mul(3, 4) == 12


class TestToolDecoratorMultiple:
    def test_multiple_tools_registered(self):
        fast = _FakeFastAgent()

        @fast.tool
        def tool_a() -> str:
            """First."""
            return "a"

        @fast.tool(name="tool_b")
        def second() -> str:
            return "b"

        assert len(fast._registered_tools) == 2
        names = [t.name for t in fast._registered_tools]
        assert names == ["tool_a", "tool_b"]


class TestToolDecoratorExecution:
    @pytest.mark.asyncio
    async def test_registered_tool_can_run(self):
        fast = _FakeFastAgent()

        @fast.tool
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        tool = fast._registered_tools[0]
        result = await tool.run({"x": 5})
        from fast_agent.mcp.helpers.content_helpers import get_text

        assert get_text(result.content[0]) == "10"

    @pytest.mark.asyncio
    async def test_async_tool_can_run(self):
        fast = _FakeFastAgent()

        @fast.tool
        async def async_double(x: int) -> int:
            """Async double."""
            return x * 2

        tool = fast._registered_tools[0]
        result = await tool.run({"x": 7})
        from fast_agent.mcp.helpers.content_helpers import get_text

        assert get_text(result.content[0]) == "14"


# ===================================================================
# @agent.tool — per-agent scoped tool decorator
# ===================================================================


class TestAgentToolBare:
    def test_bare_agent_tool_appends_to_config(self):
        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")

        @writer.tool
        def helper() -> str:
            """Help."""
            return "ok"

        config = fast.agents["writer"]["config"]
        assert config.function_tools is not None
        assert helper in config.function_tools

    def test_bare_agent_tool_returns_original_function(self):
        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")

        @writer.tool
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3

    def test_bare_agent_tool_does_not_register_globally(self):
        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")

        @writer.tool
        def local_tool() -> str:
            return "local"

        assert len(fast._registered_tools) == 0


class TestAgentToolParameterized:
    def test_custom_name_stored_on_scoped_registration(self):
        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")

        @writer.tool(name="custom_name")
        def helper() -> str:
            return "ok"

        config = fast.agents["writer"]["config"]
        assert isinstance(config.function_tools[0], ScopedFunctionToolConfig)
        assert config.function_tools[0].function is helper
        assert config.function_tools[0].name == "custom_name"

    def test_custom_description_stored_on_scoped_registration(self):
        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")

        @writer.tool(description="A custom description")
        def helper() -> str:
            return "ok"

        config = fast.agents["writer"]["config"]
        assert isinstance(config.function_tools[0], ScopedFunctionToolConfig)
        assert config.function_tools[0].function is helper
        assert config.function_tools[0].description == "A custom description"

    def test_parameterized_agent_tool_returns_original_function(self):
        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")

        @writer.tool(name="mul")
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply(3, 4) == 12


class TestAgentToolScoping:
    def test_tool_only_on_target_agent(self):
        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")
        _make_agent(fast, "analyst")

        @writer.tool
        def writer_helper() -> str:
            return "w"

        writer_config = fast.agents["writer"]["config"]
        analyst_config = fast.agents["analyst"]["config"]

        assert writer_config.function_tools is not None
        assert writer_helper in writer_config.function_tools
        assert analyst_config.function_tools is None

    def test_multiple_agent_tools_accumulate(self):
        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")

        @writer.tool
        def tool_a() -> str:
            return "a"

        @writer.tool
        def tool_b() -> str:
            return "b"

        config = fast.agents["writer"]["config"]
        assert len(config.function_tools) == 2


class TestEmptyFunctionToolsOptOut:
    """function_tools=[] should opt out of global tools."""

    def test_empty_list_keeps_config_not_none(self):
        fast = _FakeFastAgent()

        @fast.agent(name="isolated", instruction="test", function_tools=[])
        async def isolated():
            pass

        config = fast.agents["isolated"]["config"]
        assert config.function_tools is not None
        assert config.function_tools == []


class TestAgentToolMetadataPassthrough:
    """Custom name/description set via @agent.tool are picked up by load_function_tools."""

    def test_metadata_passthrough(self):
        from fast_agent.tools.function_tool_loader import load_function_tools

        def raw_fn(x: int) -> int:
            """Original doc."""
            return x

        raw_fn._fast_tool_name = "custom"  # type: ignore[attr-defined]
        raw_fn._fast_tool_description = "Custom desc"  # type: ignore[attr-defined]

        tools = load_function_tools([raw_fn])
        assert len(tools) == 1
        assert tools[0].name == "custom"
        assert tools[0].description == "Custom desc"

    def test_no_metadata_uses_defaults(self):
        from fast_agent.tools.function_tool_loader import load_function_tools

        def plain_fn(x: int) -> int:
            """Plain doc."""
            return x

        tools = load_function_tools([plain_fn])
        assert len(tools) == 1
        assert tools[0].name == "plain_fn"
        assert tools[0].description == "Plain doc."

    def test_scoped_metadata_does_not_bleed_across_shared_helpers(self):
        from fast_agent.tools.function_tool_loader import load_function_tools

        fast = _FakeFastAgent()
        writer = _make_agent(fast, "writer")
        analyst = _make_agent(fast, "analyst")

        def shared_helper() -> str:
            """Shared helper doc."""
            return "ok"

        writer.tool(name="writer_helper", description="Writer helper")(shared_helper)
        analyst.tool(name="analyst_helper", description="Analyst helper")(shared_helper)

        writer_config = fast.agents["writer"]["config"]
        analyst_config = fast.agents["analyst"]["config"]

        writer_tools = load_function_tools(writer_config.function_tools)
        analyst_tools = load_function_tools(analyst_config.function_tools)

        assert writer_tools[0].name == "writer_helper"
        assert writer_tools[0].description == "Writer helper"
        assert analyst_tools[0].name == "analyst_helper"
        assert analyst_tools[0].description == "Analyst helper"


class TestAgentToolExposure:
    def test_supported_custom_agent_exposes_tool(self):
        class SupportedCustomAgent:
            def __init__(self, config, context=None, tools=()):
                self.config = config
                self.context = context
                self.tools = tools

        fast = _FakeFastAgent()
        custom = _make_custom(fast, SupportedCustomAgent, "supported_custom")

        @custom.tool
        def helper() -> str:
            return "ok"

        config = fast.agents["supported_custom"]["config"]
        assert config.function_tools is not None
        assert helper in config.function_tools

    def test_unsupported_custom_agent_does_not_expose_tool(self):
        class UnsupportedCustomAgent:
            def __init__(self, config, context=None):
                self.config = config
                self.context = context

        fast = _FakeFastAgent()
        custom = _make_custom(fast, UnsupportedCustomAgent, "unsupported_custom")

        assert "tool" not in vars(custom)

    def test_router_does_not_expose_tool(self):
        fast = _FakeFastAgent()

        @fast.router(name="router", agents=["writer"], instruction="route")
        async def router():
            pass

        assert "tool" not in vars(router)

    def test_parallel_does_not_expose_tool(self):
        fast = _FakeFastAgent()

        @fast.parallel(name="parallel", fan_out=["writer"], fan_in="aggregator")
        async def parallel():
            pass

        assert "tool" not in vars(parallel)

    def test_unsupported_custom_agent_rejects_explicit_function_tools(self):
        class UnsupportedCustomAgent:
            def __init__(self, config, context=None):
                self.config = config
                self.context = context

        fast = _FakeFastAgent()

        def helper() -> str:
            return "ok"

        with pytest.raises(AgentConfigError, match="does not accept function tools"):

            @fast.custom(
                UnsupportedCustomAgent,
                name="bad_custom",
                instruction="test",
                function_tools=[helper],
            )
            async def bad_custom():
                pass

    def test_unsupported_custom_agent_allows_empty_function_tools_opt_out(self):
        class UnsupportedCustomAgent:
            def __init__(self, config, context=None):
                self.config = config
                self.context = context

        fast = _FakeFastAgent()

        @fast.custom(
            UnsupportedCustomAgent,
            name="isolated_custom",
            instruction="test",
            function_tools=[],
        )
        async def isolated_custom():
            pass

        config = fast.agents["isolated_custom"]["config"]
        assert config.function_tools == []
