from types import SimpleNamespace

from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import NoOpToolPermissionHandler


def test_mcp_aggregator_uses_acp_context_handlers_when_provided() -> None:
    progress_manager = object()
    permission_handler = object()

    ctx = SimpleNamespace(
        acp=SimpleNamespace(
            progress_manager=progress_manager,
            permission_handler=permission_handler,
        )
    )

    agg = MCPAggregator(
        server_names=[],
        connection_persistence=False,
        context=ctx,
    )

    assert agg._tool_handler is progress_manager
    assert agg._permission_handler is permission_handler


def test_mcp_aggregator_falls_back_to_noop_handlers_without_acp_context() -> None:
    agg = MCPAggregator(server_names=[], connection_persistence=False, context=None)

    assert isinstance(agg._tool_handler, NoOpToolExecutionHandler)
    assert isinstance(agg._permission_handler, NoOpToolPermissionHandler)


def test_mcp_aggregator_explicit_handlers_override_acp_context() -> None:
    progress_manager = object()
    permission_handler = object()
    explicit_tool_handler = object()
    explicit_permission_handler = object()

    ctx = SimpleNamespace(
        acp=SimpleNamespace(
            progress_manager=progress_manager,
            permission_handler=permission_handler,
        )
    )

    agg = MCPAggregator(
        server_names=[],
        connection_persistence=False,
        context=ctx,
        tool_handler=explicit_tool_handler,
        permission_handler=explicit_permission_handler,
    )

    assert agg._tool_handler is explicit_tool_handler
    assert agg._permission_handler is explicit_permission_handler

