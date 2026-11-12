import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

from mcp_types import Tool

from fast_agent.mcp.mcp_aggregator import MCPAggregator, NamespacedTool

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager


def test_get_server_instructions_does_not_implicitly_connect() -> None:
    aggregator = MCPAggregator(
        server_names=["huggingface", "optional"],
        connection_persistence=True,
        context=None,
        name="test-agent",
    )
    aggregator._namespaced_tool_map = {
        "huggingface.tool_a": NamespacedTool(
            tool=Tool(name="tool_a", input_schema={"type": "object"}),
            server_name="huggingface",
            namespaced_tool_name="huggingface.tool_a",
        )
    }

    huggingface_conn = SimpleNamespace(
        server_instructions="hf instructions",
        is_healthy=lambda: True,
    )

    fake_manager = SimpleNamespace(
        running_servers={"huggingface": huggingface_conn},
        # If get_server() is called, it means we're implicitly connecting, which this test forbids.
        get_server=AsyncMock(side_effect=AssertionError("get_server() should not be called")),
    )
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", fake_manager)

    result = asyncio.run(aggregator.get_server_instructions())
    assert result == {"huggingface": ("hf instructions", ["tool_a"])}
