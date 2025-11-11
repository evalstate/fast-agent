"""
Unit tests for ACP Tool Call Integration middleware.
"""

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.acp.tool_call_integration import (
    ACPToolCallMiddleware,
    create_tool_title,
    infer_tool_kind,
)
from fast_agent.acp.tool_call_permission_handler import (
    AlwaysAllowPermissionHandler,
    ToolCallPermissionResponse,
)
from fast_agent.acp.tool_call_tracker import ToolCallTracker


class MockConnection:
    """Mock ACP connection."""

    def __init__(self):
        self.notifications = []

    async def sessionUpdate(self, notification):
        self.notifications.append(notification)

    async def requestPermission(self, request):
        # Default to allow
        from acp.schema import AllowedOutcome, RequestPermissionResponse

        return RequestPermissionResponse(
            outcome=AllowedOutcome(optionId="allow_once", outcome="selected")
        )


class MockPermissionHandler:
    """Mock permission handler for testing."""

    def __init__(self, allow=True, remember=False):
        self.allow = allow
        self.remember = remember
        self.requests = []

    async def request_permission(self, request, connection):
        self.requests.append(request)
        return ToolCallPermissionResponse(
            allowed=self.allow, remember=self.remember
        )


@pytest.mark.asyncio
async def test_middleware_wraps_successful_tool_call():
    """Test that middleware properly wraps a successful tool call."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)
    middleware = ACPToolCallMiddleware(tracker=tracker)

    # Mock tool execution
    async def execute_fn():
        return CallToolResult(
            content=[TextContent(type="text", text="Success")],
            isError=False,
        )

    # Wrap the tool call
    result = await middleware.wrap_tool_call(
        tool_name="test_tool",
        arguments={"arg": "value"},
        execute_fn=execute_fn,
        tool_kind="execute",
        tool_title="Test Tool Execution",
    )

    # Verify result
    assert not result.isError
    assert len(result.content) == 1
    assert result.content[0].text == "Success"

    # Verify notifications were sent (create, in_progress, complete)
    assert len(connection.notifications) >= 3

    # Verify lifecycle
    statuses = [n.update.toolCall.status for n in connection.notifications]
    assert "pending" in statuses
    assert "in_progress" in statuses
    assert "completed" in statuses


@pytest.mark.asyncio
async def test_middleware_handles_tool_error():
    """Test that middleware properly handles tool execution errors."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)
    middleware = ACPToolCallMiddleware(tracker=tracker)

    # Mock tool that raises an exception
    async def execute_fn():
        raise RuntimeError("Tool failed!")

    # Wrap the tool call
    result = await middleware.wrap_tool_call(
        tool_name="failing_tool",
        arguments={},
        execute_fn=execute_fn,
        tool_kind="execute",
    )

    # Verify error result
    assert result.isError
    assert "Tool execution error" in result.content[0].text

    # Verify failure notification was sent
    assert len(connection.notifications) >= 3
    final_notification = connection.notifications[-1]
    assert final_notification.update.toolCall.status == "failed"


@pytest.mark.asyncio
async def test_middleware_permission_denied():
    """Test that middleware respects permission denial."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    # Create handler that denies permission
    permission_handler = MockPermissionHandler(allow=False)
    middleware = ACPToolCallMiddleware(
        tracker=tracker,
        permission_handler=permission_handler,
        enable_permissions=True,
    )

    # Mock tool execution (shouldn't be called)
    called = False

    async def execute_fn():
        nonlocal called
        called = True
        return CallToolResult(content=[], isError=False)

    # Try to execute tool
    result = await middleware.wrap_tool_call(
        tool_name="restricted_tool",
        arguments={},
        execute_fn=execute_fn,
    )

    # Verify permission was requested
    assert len(permission_handler.requests) == 1

    # Verify tool was not executed
    assert not called

    # Verify error result
    assert result.isError
    assert "permission denied" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_middleware_permission_allowed():
    """Test that middleware allows execution when permission is granted."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    # Create handler that allows permission
    permission_handler = MockPermissionHandler(allow=True)
    middleware = ACPToolCallMiddleware(
        tracker=tracker,
        permission_handler=permission_handler,
        enable_permissions=True,
    )

    # Mock tool execution
    called = False

    async def execute_fn():
        nonlocal called
        called = True
        return CallToolResult(
            content=[TextContent(type="text", text="Executed")],
            isError=False,
        )

    # Execute tool
    result = await middleware.wrap_tool_call(
        tool_name="allowed_tool",
        arguments={},
        execute_fn=execute_fn,
    )

    # Verify permission was requested
    assert len(permission_handler.requests) == 1

    # Verify tool was executed
    assert called

    # Verify success result
    assert not result.isError


@pytest.mark.asyncio
async def test_middleware_permission_caching():
    """Test that middleware caches permission decisions when requested."""
    connection = MockConnection()
    tracker = ToolCallTracker(session_id="test-session", connection=connection)

    # Create handler that remembers decisions
    permission_handler = MockPermissionHandler(allow=True, remember=True)
    middleware = ACPToolCallMiddleware(
        tracker=tracker,
        permission_handler=permission_handler,
        enable_permissions=True,
    )

    async def execute_fn():
        return CallToolResult(content=[], isError=False)

    # Execute tool twice
    await middleware.wrap_tool_call(
        tool_name="cacheable_tool",
        arguments={},
        execute_fn=execute_fn,
    )

    await middleware.wrap_tool_call(
        tool_name="cacheable_tool",
        arguments={},
        execute_fn=execute_fn,
    )

    # Verify permission was only requested once (cached for second call)
    assert len(permission_handler.requests) == 1


def test_infer_tool_kind():
    """Test tool kind inference from tool names."""
    assert infer_tool_kind("read_file") == "read"
    assert infer_tool_kind("get_data") == "read"
    assert infer_tool_kind("write_file") == "edit"
    assert infer_tool_kind("update_record") == "edit"
    assert infer_tool_kind("delete_user") == "delete"
    assert infer_tool_kind("remove_item") == "delete"
    assert infer_tool_kind("move_file") == "move"
    assert infer_tool_kind("search_documents") == "search"
    assert infer_tool_kind("find_user") == "search"
    assert infer_tool_kind("execute_command") == "execute"
    assert infer_tool_kind("run_script") == "execute"
    assert infer_tool_kind("bash_exec") == "execute"
    assert infer_tool_kind("think_about") == "think"
    assert infer_tool_kind("http_request") == "fetch"
    assert infer_tool_kind("api_call") == "fetch"
    assert infer_tool_kind("unknown_tool") == "other"


def test_create_tool_title():
    """Test tool title creation."""
    # Basic title
    assert create_tool_title("read_file") == "Read File"

    # With path argument
    title = create_tool_title("read_file", {"path": "/tmp/test.txt"})
    assert "/tmp/test.txt" in title

    # With query argument
    title = create_tool_title("search_database", {"query": "SELECT * FROM users"})
    assert "SELECT * FROM users" in title

    # Namespaced tool name
    assert create_tool_title("server__tool_name") == "Tool Name"

    # Long argument value (should be excluded)
    title = create_tool_title("read_file", {"path": "x" * 100})
    assert "x" * 100 not in title


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
