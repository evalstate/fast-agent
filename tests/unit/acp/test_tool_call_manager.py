"""Unit tests for ToolCallManager."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.acp.tool_call_manager import ToolCallManager


@pytest.mark.asyncio
async def test_infer_tool_kind():
    """Test tool kind inference from tool names."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    # Test exact matches
    assert manager.infer_tool_kind("read") == "read"
    assert manager.infer_tool_kind("write") == "edit"
    assert manager.infer_tool_kind("execute") == "execute"
    assert manager.infer_tool_kind("search") == "search"

    # Test partial matches
    assert manager.infer_tool_kind("read_file") == "read"
    assert manager.infer_tool_kind("edit_file") == "edit"
    assert manager.infer_tool_kind("bash_execute") == "execute"
    assert manager.infer_tool_kind("grep_search") == "search"

    # Test case insensitivity
    assert manager.infer_tool_kind("READ_FILE") == "read"
    assert manager.infer_tool_kind("Write_File") == "edit"

    # Test unknown tool
    assert manager.infer_tool_kind("unknown_tool") == "other"


@pytest.mark.asyncio
async def test_create_tool_call():
    """Test creating a tool call sends proper notification."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    # Create a tool call
    tool_call_id = await manager.create_tool_call(
        tool_name="read_file",
        server_name="filesystem",
        arguments={"path": "/test/file.txt"},
    )

    # Verify a UUID was generated
    assert tool_call_id
    assert len(tool_call_id) == 36  # UUID format

    # Verify notification was sent
    assert mock_conn.sessionUpdate.called
    call_args = mock_conn.sessionUpdate.call_args[0][0]

    assert call_args["sessionId"] == "test-session"
    assert call_args["sessionUpdate"] == "tool_call"
    assert call_args["toolCallId"] == tool_call_id
    assert call_args["title"] == "read_file (filesystem)"
    assert call_args["status"] == "pending"
    assert call_args["kind"] == "read"
    assert call_args["rawInput"] == {"path": "/test/file.txt"}


@pytest.mark.asyncio
async def test_create_tool_call_with_explicit_kind():
    """Test creating a tool call with explicit kind override."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    tool_call_id = await manager.create_tool_call(
        tool_name="custom_tool",
        server_name="custom",
        arguments={},
        kind="fetch",  # Explicit override
    )

    call_args = mock_conn.sessionUpdate.call_args[0][0]
    assert call_args["kind"] == "fetch"


@pytest.mark.asyncio
async def test_update_tool_call_status():
    """Test updating tool call status."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    # Create a tool call first
    tool_call_id = await manager.create_tool_call(
        tool_name="test_tool",
        server_name="test_server",
        arguments={},
    )

    mock_conn.sessionUpdate.reset_mock()

    # Update to in_progress
    await manager.update_tool_call(
        tool_call_id=tool_call_id,
        status="in_progress",
    )

    assert mock_conn.sessionUpdate.called
    call_args = mock_conn.sessionUpdate.call_args[0][0]
    assert call_args["sessionUpdate"] == "tool_call_update"
    assert call_args["toolCallId"] == tool_call_id
    assert call_args["status"] == "in_progress"


@pytest.mark.asyncio
async def test_update_tool_call_with_content():
    """Test updating tool call with content."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    tool_call_id = await manager.create_tool_call(
        tool_name="test_tool",
        server_name="test_server",
        arguments={},
    )

    mock_conn.sessionUpdate.reset_mock()

    # Update with content
    await manager.update_tool_call(
        tool_call_id=tool_call_id,
        content="Processing file...",
    )

    call_args = mock_conn.sessionUpdate.call_args[0][0]
    assert "content" in call_args
    assert len(call_args["content"]) == 1
    assert call_args["content"][0]["type"] == "text"
    assert call_args["content"][0]["text"] == "Processing file..."


@pytest.mark.asyncio
async def test_update_tool_call_with_locations():
    """Test updating tool call with file locations."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    tool_call_id = await manager.create_tool_call(
        tool_name="edit_file",
        server_name="filesystem",
        arguments={},
    )

    mock_conn.sessionUpdate.reset_mock()

    # Update with locations
    await manager.update_tool_call(
        tool_call_id=tool_call_id,
        locations=["/path/file1.txt", "/path/file2.txt"],
    )

    call_args = mock_conn.sessionUpdate.call_args[0][0]
    assert "locations" in call_args
    assert len(call_args["locations"]) == 2
    assert call_args["locations"][0]["path"] == "/path/file1.txt"
    assert call_args["locations"][1]["path"] == "/path/file2.txt"


@pytest.mark.asyncio
async def test_progress_update():
    """Test progress updates."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    tool_call_id = await manager.create_tool_call(
        tool_name="download_file",
        server_name="network",
        arguments={},
    )

    mock_conn.sessionUpdate.reset_mock()

    # Send progress update
    await manager.progress_update(
        tool_call_id=tool_call_id,
        progress=50,
        total=100,
        message="Downloading...",
    )

    call_args = mock_conn.sessionUpdate.call_args[0][0]
    assert "content" in call_args
    content_text = call_args["content"][0]["text"]
    assert "50.0%" in content_text
    assert "50/100" in content_text
    assert "Downloading..." in content_text


@pytest.mark.asyncio
async def test_progress_update_without_total():
    """Test progress updates without total."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    tool_call_id = await manager.create_tool_call(
        tool_name="test_tool",
        server_name="test_server",
        arguments={},
    )

    mock_conn.sessionUpdate.reset_mock()

    # Send progress without total
    await manager.progress_update(
        tool_call_id=tool_call_id,
        progress=42,
        total=None,
        message="Processing...",
    )

    call_args = mock_conn.sessionUpdate.call_args[0][0]
    content_text = call_args["content"][0]["text"]
    assert "Progress: 42" in content_text
    assert "Processing..." in content_text


@pytest.mark.asyncio
async def test_complete_tool_call_success():
    """Test completing a tool call successfully."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    tool_call_id = await manager.create_tool_call(
        tool_name="test_tool",
        server_name="test_server",
        arguments={},
    )

    assert manager.get_active_call_count() == 1

    # Complete successfully
    result = CallToolResult(
        content=[TextContent(type="text", text="Success!")],
        isError=False,
    )

    await manager.complete_tool_call(
        tool_call_id=tool_call_id,
        output=result,
        is_error=False,
    )

    # Verify completion notification
    call_args = mock_conn.sessionUpdate.call_args[0][0]
    assert call_args["status"] == "completed"
    assert "Success!" in call_args["content"][0]["text"]

    # Verify cleanup
    assert manager.get_active_call_count() == 0


@pytest.mark.asyncio
async def test_complete_tool_call_failure():
    """Test completing a tool call with error."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    tool_call_id = await manager.create_tool_call(
        tool_name="test_tool",
        server_name="test_server",
        arguments={},
    )

    # Complete with error
    result = CallToolResult(
        content=[TextContent(type="text", text="Error: file not found")],
        isError=True,
    )

    await manager.complete_tool_call(
        tool_call_id=tool_call_id,
        output=result,
        is_error=True,
    )

    # Verify failure notification
    call_args = mock_conn.sessionUpdate.call_args[0][0]
    assert call_args["status"] == "failed"


@pytest.mark.asyncio
async def test_update_unknown_tool_call():
    """Test that updating an unknown tool call logs warning but doesn't crash."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock()

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    # Update a tool call that doesn't exist
    await manager.update_tool_call(
        tool_call_id="unknown-id",
        status="in_progress",
    )

    # Should not have called sessionUpdate
    assert not mock_conn.sessionUpdate.called


@pytest.mark.asyncio
async def test_notification_failure_handling():
    """Test that notification failures are logged but don't crash the manager."""
    mock_conn = MagicMock()
    mock_conn.sessionUpdate = AsyncMock(side_effect=Exception("Network error"))

    manager = ToolCallManager(connection=mock_conn, session_id="test-session")

    # Should not raise, even though sessionUpdate fails
    tool_call_id = await manager.create_tool_call(
        tool_name="test_tool",
        server_name="test_server",
        arguments={},
    )

    # Verify we still got an ID
    assert tool_call_id
