from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import TextContent

from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.tool_progress import ACPToolProgressManager
from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import ToolPermissionResult


class _RecordingConnection:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.reads: list[dict[str, Any]] = []
        self.writes: list[dict[str, str]] = []
        self.session_updates: list[Any] = []

    async def read_text_file(
        self,
        *,
        path: str,
        session_id: str,
        line: int | None = None,
        limit: int | None = None,
    ) -> SimpleNamespace:
        self.events.append("read")
        self.reads.append(
            {
                "path": path,
                "session_id": session_id,
                "line": line,
                "limit": limit,
            }
        )
        return SimpleNamespace(content="old")

    async def session_update(self, *, session_id: str, update: Any) -> None:
        self.events.append("diff")
        self.session_updates.append(update)

    async def write_text_file(self, *, content: str, path: str, session_id: str) -> None:
        self.events.append("write")
        self.writes.append({"path": path, "content": content})


class _RecordingPermissionHandler:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.events: list[str] | None = None
        self.calls: list[tuple[str, str, dict[str, Any] | None, str | None]] = []

    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        self.calls.append((tool_name, server_name, arguments, tool_use_id))
        if self.events is not None:
            self.events.append("permission")
        return (
            ToolPermissionResult.allow()
            if self.allowed
            else ToolPermissionResult(allowed=False, error_message="denied")
        )


class _RecordingToolHandler:
    def __init__(self) -> None:
        self.ensures: list[tuple[str, str, str, dict[Any, Any] | None]] = []
        self.starts: list[tuple[str, str, dict[Any, Any] | None, str | None]] = []
        self.completions: list[tuple[str, bool, list[Any] | None, str | None]] = []
        self.denials: list[tuple[str, str, str | None, str]] = []

    async def ensure_tool_call_exists(
        self,
        tool_use_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict[Any, Any] | None = None,
    ) -> str:
        self.ensures.append((tool_use_id, tool_name, server_name, arguments))
        return f"ensured-{tool_use_id}"

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[Any, Any] | None,
        tool_use_id: str | None = None,
    ) -> str:
        self.starts.append((tool_name, server_name, arguments, tool_use_id))
        return "tool-call-1"

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: list[Any] | None,
        error: str | None,
    ) -> None:
        self.completions.append((tool_call_id, success, content, error))

    async def on_tool_progress(
        self,
        tool_call_id: str,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        del tool_call_id, progress, total, message

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        self.denials.append((tool_name, server_name, tool_use_id, error or ""))

    async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
        return f"ensured-{tool_use_id}"


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []

    def info(self, message: str, **kwargs: Any) -> None:
        del kwargs
        self.info_messages.append(message)

    def error(self, message: str, **kwargs: Any) -> None:
        del message, kwargs


def _runtime(
    connection: _RecordingConnection,
    permission_handler: _RecordingPermissionHandler,
    tool_handler: ToolExecutionHandler,
    logger: _RecordingLogger | None = None,
    *,
    enable_read: bool = True,
    enable_write: bool = True,
) -> ACPFilesystemRuntime:
    permission_handler.events = connection.events
    return ACPFilesystemRuntime(
        connection=connection,  # type: ignore[arg-type]
        session_id="session-1",
        activation_reason="test",
        enable_read=enable_read,
        enable_write=enable_write,
        permission_handler=permission_handler,
        tool_handler=tool_handler,
        logger_instance=logger,
    )


@pytest.mark.asyncio
async def test_write_text_file_allows_empty_content() -> None:
    connection = _RecordingConnection()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(),
        _RecordingToolHandler(),
    )

    result = await runtime.write_text_file(
        {"path": "empty.txt", "content": ""},
        tool_use_id="tool-1",
    )

    assert result.isError is False
    assert connection.writes == [{"path": "empty.txt", "content": ""}]


@pytest.mark.asyncio
async def test_filesystem_tools_reject_whitespace_only_path() -> None:
    connection = _RecordingConnection()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(),
        _RecordingToolHandler(),
    )

    read_result = await runtime.read_text_file({"path": "   "}, tool_use_id="read-1")
    write_result = await runtime.write_text_file(
        {"path": "   ", "content": "new"},
        tool_use_id="write-1",
    )

    assert read_result.isError is True
    assert write_result.isError is True
    assert connection.events == []


@pytest.mark.asyncio
async def test_call_tool_rejects_disabled_write_tool_without_side_effects() -> None:
    connection = _RecordingConnection()
    tool_handler = _RecordingToolHandler()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(),
        tool_handler,
        enable_write=False,
    )

    result = await runtime.call_tool(
        "write_text_file",
        {"path": "target.txt", "content": "new"},
        tool_use_id="tool-1",
    )

    assert result.isError is True
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "Error: unsupported ACP filesystem tool 'write_text_file'."
    assert connection.events == []
    assert connection.writes == []
    assert tool_handler.ensures == []
    assert tool_handler.starts == []


@pytest.mark.asyncio
async def test_read_text_file_forwards_line_and_limit() -> None:
    connection = _RecordingConnection()
    tool_handler = _RecordingToolHandler()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(),
        tool_handler,
    )

    result = await runtime.call_tool(
        "read_text_file",
        {"path": "target.txt", "line": 2, "limit": 3},
        tool_use_id="tool-1",
    )

    assert result.isError is False
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "old"
    assert connection.events == ["permission", "read"]
    assert connection.reads == [
        {
            "path": "target.txt",
            "session_id": "session-1",
            "line": 2,
            "limit": 3,
        }
    ]
    assert tool_handler.ensures == [
        (
            "tool-1",
            "read_text_file",
            "acp_filesystem",
            {"path": "target.txt", "line": 2, "limit": 3},
        )
    ]
    assert tool_handler.completions == [("ensured-tool-1", True, result.content, None)]


@pytest.mark.asyncio
async def test_write_text_file_sends_diff_before_write_after_permission() -> None:
    connection = _RecordingConnection()
    permission_handler = _RecordingPermissionHandler()
    tool_handler = _RecordingToolHandler()
    runtime = _runtime(
        connection,
        permission_handler,
        tool_handler,
    )

    result = await runtime.write_text_file(
        {"path": "target.txt", "content": "new"},
        tool_use_id="tool-1",
    )

    assert result.isError is False
    assert connection.events[:5] == ["diff", "permission", "read", "diff", "write"]
    assert permission_handler.calls == [
        (
            "write_text_file",
            "acp_filesystem",
            {"path": "target.txt", "content_length": 3},
            "tool-1",
        )
    ]
    assert len(connection.session_updates) == 2
    assert connection.session_updates[0].tool_call_id == "ensured-tool-1"
    assert connection.session_updates[0].content[0].old_text is None
    assert connection.session_updates[0].content[0].new_text == "new"
    assert connection.session_updates[1].tool_call_id == "ensured-tool-1"
    assert connection.session_updates[1].content[0].old_text == "old"
    assert connection.session_updates[1].content[0].new_text == "new"
    assert tool_handler.ensures == [
        (
            "tool-1",
            "write_text_file",
            "acp_filesystem",
            {"path": "target.txt", "content_length": 3},
        )
    ]
    assert tool_handler.starts == []
    assert tool_handler.completions == [("ensured-tool-1", True, None, None)]


@pytest.mark.asyncio
async def test_write_text_file_in_write_only_mode_skips_old_content_read() -> None:
    connection = _RecordingConnection()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(),
        _RecordingToolHandler(),
        enable_read=False,
    )

    result = await runtime.write_text_file(
        {"path": "target.txt", "content": "new"},
        tool_use_id="tool-1",
    )

    assert result.isError is False
    assert connection.events == ["diff", "permission", "diff", "write"]
    assert connection.reads == []
    assert connection.session_updates[0].content[0].old_text is None
    assert connection.session_updates[0].content[0].new_text == "new"
    assert connection.session_updates[1].content[0].old_text is None
    assert connection.session_updates[1].content[0].new_text == "new"


@pytest.mark.asyncio
async def test_write_text_file_keeps_diff_after_progress_start_update() -> None:
    connection = _RecordingConnection()
    tool_handler = ACPToolProgressManager(
        connection=connection,  # type: ignore[arg-type]
        session_id="session-1",
    )
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(),
        tool_handler,
    )
    tool_handler.handle_tool_stream_event(
        "start",
        {
            "tool_name": "write_text_file",
            "tool_use_id": "tool-1",
        },
    )

    result = await runtime.write_text_file(
        {"path": "target.txt", "content": "new"},
        tool_use_id="tool-1",
    )

    assert result.isError is False
    content_updates = [
        (index, update)
        for index, update in enumerate(connection.session_updates)
        if getattr(update, "content", None) is not None
    ]
    diff_updates = [
        (index, update)
        for index, update in content_updates
        if update.content and getattr(update.content[0], "new_text", None) == "new"
    ]
    assert len(diff_updates) == 2
    _preview_index, preview_update = diff_updates[0]
    assert preview_update.content[0].old_text is None
    diff_index, diff_update = diff_updates[-1]
    assert diff_update.content[0].old_text == "old"
    assert diff_update.content[0].new_text == "new"
    assert all(
        getattr(update, "content", None) is None
        for update in connection.session_updates[diff_index + 1 :]
    )
    assert "write" in connection.events


@pytest.mark.asyncio
async def test_write_text_file_denial_sends_preview_without_read_or_write() -> None:
    connection = _RecordingConnection()
    tool_handler = _RecordingToolHandler()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(allowed=False),
        tool_handler,
    )

    result = await runtime.write_text_file(
        {"path": "target.txt", "content": "new"},
        tool_use_id="tool-1",
    )

    assert result.isError is True
    assert connection.events == ["diff", "permission"]
    assert len(connection.session_updates) == 1
    assert connection.session_updates[0].content[0].old_text is None
    assert connection.session_updates[0].content[0].new_text == "new"
    assert not connection.writes
    assert tool_handler.ensures == [
        (
            "tool-1",
            "write_text_file",
            "acp_filesystem",
            {"path": "target.txt", "content_length": 3},
        )
    ]
    assert tool_handler.denials == [("write_text_file", "acp_filesystem", "tool-1", "denied")]


@pytest.mark.asyncio
async def test_write_text_file_denial_without_tool_use_id_does_not_start_progress() -> None:
    connection = _RecordingConnection()
    tool_handler = _RecordingToolHandler()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(allowed=False),
        tool_handler,
    )

    result = await runtime.write_text_file(
        {"path": "target.txt", "content": "new"},
        tool_use_id=None,
    )

    assert result.isError is True
    assert connection.events == ["permission"]
    assert connection.session_updates == []
    assert connection.reads == []
    assert connection.writes == []
    assert tool_handler.starts == []
    assert tool_handler.ensures == []
    assert tool_handler.completions == []
    assert tool_handler.denials == []


@pytest.mark.asyncio
async def test_read_text_file_denial_skips_read_and_notifies_tool_handler() -> None:
    connection = _RecordingConnection()
    tool_handler = _RecordingToolHandler()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(allowed=False),
        tool_handler,
    )

    result = await runtime.read_text_file(
        {"path": "target.txt"},
        tool_use_id="tool-1",
    )

    assert result.isError is True
    assert connection.events == ["permission"]
    assert connection.reads == []
    assert tool_handler.ensures == [
        (
            "tool-1",
            "read_text_file",
            "acp_filesystem",
            {"path": "target.txt"},
        )
    ]
    assert tool_handler.denials == [("read_text_file", "acp_filesystem", "tool-1", "denied")]


@pytest.mark.asyncio
async def test_read_text_file_denial_without_tool_use_id_does_not_start_progress() -> None:
    connection = _RecordingConnection()
    tool_handler = _RecordingToolHandler()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(allowed=False),
        tool_handler,
    )

    result = await runtime.read_text_file(
        {"path": "target.txt"},
        tool_use_id=None,
    )

    assert result.isError is True
    assert connection.events == ["permission"]
    assert connection.reads == []
    assert tool_handler.starts == []
    assert tool_handler.ensures == []
    assert tool_handler.completions == []
    assert tool_handler.denials == []


@pytest.mark.asyncio
async def test_write_permission_denial_logs_write_action_label() -> None:
    connection = _RecordingConnection()
    logger = _RecordingLogger()
    runtime = _runtime(
        connection,
        _RecordingPermissionHandler(allowed=False),
        _RecordingToolHandler(),
        logger,
    )

    await runtime.write_text_file(
        {"path": "target.txt", "content": "new"},
        tool_use_id="tool-1",
    )

    assert "File write denied by permission handler" in logger.info_messages
    assert "File writ denied by permission handler" not in logger.info_messages
