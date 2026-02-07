from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import NoOpToolPermissionHandler

if TYPE_CHECKING:
    from fast_agent.acp.acp_context import ACPContext


def test_mcp_aggregator_uses_acp_context_handlers_when_provided() -> None:
    progress_manager = NoOpToolExecutionHandler()
    permission_handler = NoOpToolPermissionHandler()

    # Context.acp is typed as ACPContext, but these tests only need the two handler attributes.
    acp_ctx = SimpleNamespace(
        progress_manager=progress_manager,
        permission_handler=permission_handler,
    )
    ctx = Context(acp=cast("ACPContext", acp_ctx))

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
    progress_manager = NoOpToolExecutionHandler()
    permission_handler = NoOpToolPermissionHandler()
    explicit_tool_handler = NoOpToolExecutionHandler()
    explicit_permission_handler = NoOpToolPermissionHandler()

    acp_ctx = SimpleNamespace(
        progress_manager=progress_manager,
        permission_handler=permission_handler,
    )
    ctx = Context(acp=cast("ACPContext", acp_ctx))

    agg = MCPAggregator(
        server_names=[],
        connection_persistence=False,
        context=ctx,
        tool_handler=explicit_tool_handler,
        permission_handler=explicit_permission_handler,
    )

    assert agg._tool_handler is explicit_tool_handler
    assert agg._permission_handler is explicit_permission_handler
