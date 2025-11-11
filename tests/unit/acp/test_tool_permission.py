"""Unit tests for tool permission handlers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fast_agent.acp.tool_permission import (
    ToolPermissionContext,
    ToolPermissionManager,
    acp_tool_permission_handler,
    allow_all_tool_permission_handler,
    deny_all_tool_permission_handler,
)


@pytest.mark.asyncio
async def test_allow_all_handler():
    """Test that allow_all handler always permits execution."""
    mock_conn = MagicMock()

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="dangerous_tool",
        server_name="test_server",
        arguments={"action": "delete_all"},
        tool_call_id="test-id",
    )

    result = await allow_all_tool_permission_handler(context, mock_conn)

    assert result.allowed is True
    assert result.remember is False
    assert result.cancelled is False


@pytest.mark.asyncio
async def test_deny_all_handler():
    """Test that deny_all handler always denies execution."""
    mock_conn = MagicMock()

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="safe_tool",
        server_name="test_server",
        arguments={},
        tool_call_id="test-id",
    )

    result = await deny_all_tool_permission_handler(context, mock_conn)

    assert result.allowed is False
    assert result.remember is False
    assert result.cancelled is False


@pytest.mark.asyncio
async def test_acp_handler_allow_once():
    """Test ACP handler with allow_once response."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "allow_once",
            }
        }
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="read_file",
        server_name="filesystem",
        arguments={"path": "/test.txt"},
        tool_call_id="test-id",
    )

    result = await acp_tool_permission_handler(context, mock_conn)

    assert result.allowed is True
    assert result.remember is False
    assert result.cancelled is False

    # Verify request was sent
    assert mock_conn._conn.send_request.called
    call_args = mock_conn._conn.send_request.call_args
    assert call_args[0][0] == "session/request_permission"


@pytest.mark.asyncio
async def test_acp_handler_allow_always():
    """Test ACP handler with allow_always response."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "allow_always",
            }
        }
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="read_file",
        server_name="filesystem",
        arguments={"path": "/test.txt"},
        tool_call_id="test-id",
    )

    result = await acp_tool_permission_handler(context, mock_conn)

    assert result.allowed is True
    assert result.remember is True
    assert result.cancelled is False


@pytest.mark.asyncio
async def test_acp_handler_reject_once():
    """Test ACP handler with reject_once response."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "reject_once",
            }
        }
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="delete_file",
        server_name="filesystem",
        arguments={"path": "/important.txt"},
        tool_call_id="test-id",
    )

    result = await acp_tool_permission_handler(context, mock_conn)

    assert result.allowed is False
    assert result.remember is False
    assert result.cancelled is False


@pytest.mark.asyncio
async def test_acp_handler_reject_always():
    """Test ACP handler with reject_always response."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "reject_always",
            }
        }
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="delete_file",
        server_name="filesystem",
        arguments={"path": "/important.txt"},
        tool_call_id="test-id",
    )

    result = await acp_tool_permission_handler(context, mock_conn)

    assert result.allowed is False
    assert result.remember is True
    assert result.cancelled is False


@pytest.mark.asyncio
async def test_acp_handler_cancelled():
    """Test ACP handler with cancelled outcome."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "cancelled",
            }
        }
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="test_tool",
        server_name="test_server",
        arguments={},
        tool_call_id="test-id",
    )

    result = await acp_tool_permission_handler(context, mock_conn)

    assert result.allowed is False
    assert result.cancelled is True


@pytest.mark.asyncio
async def test_acp_handler_network_error():
    """Test ACP handler when network request fails."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(side_effect=Exception("Network error"))

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="test_tool",
        server_name="test_server",
        arguments={},
        tool_call_id="test-id",
    )

    # Should default to denying on error
    result = await acp_tool_permission_handler(context, mock_conn)

    assert result.allowed is False


@pytest.mark.asyncio
async def test_permission_manager_caching_allow():
    """Test permission manager caches allow_always decisions."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "allow_always",
            }
        }
    )

    manager = ToolPermissionManager(
        handler=acp_tool_permission_handler,
        connection=mock_conn,
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="read_file",
        server_name="filesystem",
        arguments={"path": "/test.txt"},
        tool_call_id="test-id-1",
    )

    # First call - should query the handler
    result1 = await manager.check_permission(context)
    assert result1.allowed is True
    assert result1.remember is True
    assert mock_conn._conn.send_request.call_count == 1

    # Second call with same tool - should use cache
    context.tool_call_id = "test-id-2"
    result2 = await manager.check_permission(context)
    assert result2.allowed is True
    assert result2.remember is True
    # Should not have made another request
    assert mock_conn._conn.send_request.call_count == 1


@pytest.mark.asyncio
async def test_permission_manager_caching_deny():
    """Test permission manager caches reject_always decisions."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "reject_always",
            }
        }
    )

    manager = ToolPermissionManager(
        handler=acp_tool_permission_handler,
        connection=mock_conn,
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="delete_file",
        server_name="filesystem",
        arguments={"path": "/important.txt"},
        tool_call_id="test-id-1",
    )

    # First call
    result1 = await manager.check_permission(context)
    assert result1.allowed is False
    assert result1.remember is True

    # Second call - should use cache
    context.tool_call_id = "test-id-2"
    result2 = await manager.check_permission(context)
    assert result2.allowed is False
    assert result2.remember is True
    assert mock_conn._conn.send_request.call_count == 1


@pytest.mark.asyncio
async def test_permission_manager_no_caching_for_once():
    """Test permission manager doesn't cache allow_once/reject_once."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "allow_once",
            }
        }
    )

    manager = ToolPermissionManager(
        handler=acp_tool_permission_handler,
        connection=mock_conn,
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="read_file",
        server_name="filesystem",
        arguments={"path": "/test.txt"},
        tool_call_id="test-id-1",
    )

    # First call
    result1 = await manager.check_permission(context)
    assert result1.allowed is True
    assert result1.remember is False

    # Second call - should query again since remember=False
    context.tool_call_id = "test-id-2"
    result2 = await manager.check_permission(context)
    assert result2.allowed is True
    # Should have made two requests
    assert mock_conn._conn.send_request.call_count == 2


@pytest.mark.asyncio
async def test_permission_manager_clear_cache():
    """Test clearing permission cache."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "allow_always",
            }
        }
    )

    manager = ToolPermissionManager(
        handler=acp_tool_permission_handler,
        connection=mock_conn,
    )

    context = ToolPermissionContext(
        session_id="test-session",
        tool_name="read_file",
        server_name="filesystem",
        arguments={"path": "/test.txt"},
        tool_call_id="test-id-1",
    )

    # First call - should cache
    await manager.check_permission(context)
    assert mock_conn._conn.send_request.call_count == 1

    # Clear cache
    manager.clear_cache()

    # Second call - should query again
    context.tool_call_id = "test-id-2"
    await manager.check_permission(context)
    assert mock_conn._conn.send_request.call_count == 2


@pytest.mark.asyncio
async def test_permission_manager_different_tools():
    """Test that cache is per-tool, not global."""
    mock_conn = MagicMock()
    mock_conn._conn = AsyncMock()
    mock_conn._conn.send_request = AsyncMock(
        return_value={
            "outcome": {
                "outcome": "selected",
                "optionId": "allow_always",
            }
        }
    )

    manager = ToolPermissionManager(
        handler=acp_tool_permission_handler,
        connection=mock_conn,
    )

    # Allow read_file
    context1 = ToolPermissionContext(
        session_id="test-session",
        tool_name="read_file",
        server_name="filesystem",
        arguments={},
        tool_call_id="test-id-1",
    )
    await manager.check_permission(context1)

    # Try different tool - should query again
    context2 = ToolPermissionContext(
        session_id="test-session",
        tool_name="write_file",
        server_name="filesystem",
        arguments={},
        tool_call_id="test-id-2",
    )
    await manager.check_permission(context2)

    # Should have made two requests
    assert mock_conn._conn.send_request.call_count == 2
