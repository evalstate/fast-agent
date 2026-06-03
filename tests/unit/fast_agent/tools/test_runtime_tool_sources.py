from __future__ import annotations

import logging

import pytest
from mcp.types import TextContent

from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.mcp.tool_permission_handler import ToolPermissionResult
from fast_agent.tools.local_filesystem_runtime import LocalFilesystemRuntime
from fast_agent.tools.shell_runtime import ShellRuntime
from fast_agent.tools.skill_reader import READ_SKILL_TOOL_NAME, SkillReader
from fast_agent.tools.tool_sources import (
    ACP_FILESYSTEM_TOOL_SOURCE,
    ACP_TERMINAL_TOOL_SOURCE,
    SHELL_TOOL_SOURCE,
    SKILL_TOOL_SOURCE,
    tool_source,
)


def test_shell_runtime_stamps_execute_as_shell() -> None:
    runtime = ShellRuntime("for test", logging.getLogger(__name__))

    assert runtime.tool is not None
    assert tool_source(runtime.tool) == SHELL_TOOL_SOURCE


def test_acp_terminal_runtime_stamps_execute_as_acp_terminal() -> None:
    runtime = ACPTerminalRuntime(
        connection=object(),
        session_id="session",
        activation_reason="for test",
    )

    assert tool_source(runtime.tool) == ACP_TERMINAL_TOOL_SOURCE


def test_local_filesystem_runtime_stamps_enabled_tools_as_shell() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger(__name__),
        enable_apply_patch=True,
        enable_edit_file=True,
        enable_attach_media="on",
    )

    assert {tool.name: tool_source(tool) for tool in runtime.tools} == {
        "read_text_file": SHELL_TOOL_SOURCE,
        "write_text_file": SHELL_TOOL_SOURCE,
        "apply_patch": SHELL_TOOL_SOURCE,
        "edit_file": SHELL_TOOL_SOURCE,
        "attach_media": SHELL_TOOL_SOURCE,
    }


def test_acp_filesystem_runtime_stamps_enabled_tools_as_acp_filesystem() -> None:
    runtime = ACPFilesystemRuntime(
        connection=object(),
        session_id="session",
        activation_reason="for test",
    )

    assert {tool.name: tool_source(tool) for tool in runtime.tools} == {
        "read_text_file": ACP_FILESYSTEM_TOOL_SOURCE,
        "write_text_file": ACP_FILESYSTEM_TOOL_SOURCE,
    }


@pytest.mark.asyncio
async def test_acp_filesystem_denied_write_has_no_filesystem_or_diff_side_effects() -> None:
    class _Connection:
        read_calls = 0
        write_calls = 0
        update_calls = 0

        async def read_text_file(self, **_kwargs):
            self.read_calls += 1
            return type("ReadResult", (), {"content": "old"})()

        async def write_text_file(self, **_kwargs):
            self.write_calls += 1
            raise AssertionError("denied write must not execute")

        async def session_update(self, **_kwargs):
            self.update_calls += 1

    class _PermissionHandler:
        async def check_permission(
            self,
            tool_name: str,
            server_name: str,
            arguments: dict[str, object] | None = None,
            tool_use_id: str | None = None,
        ) -> ToolPermissionResult:
            del tool_name, server_name, arguments, tool_use_id
            return ToolPermissionResult.deny("no")

    class _ToolHandler:
        def __init__(self) -> None:
            self.ensured: list[tuple[str, str, str, dict[str, object] | None]] = []
            self.denied: list[tuple[str, str, str | None, str | None]] = []

        async def ensure_tool_call_exists(
            self,
            tool_use_id: str,
            tool_name: str,
            server_name: str,
            arguments: dict[str, object] | None = None,
        ) -> str:
            self.ensured.append((tool_use_id, tool_name, server_name, arguments))
            return "tool-call-1"

        async def on_tool_permission_denied(
            self,
            tool_name: str,
            server_name: str,
            tool_use_id: str | None,
            error: str | None = None,
        ) -> None:
            self.denied.append((tool_name, server_name, tool_use_id, error))

        async def on_tool_start(
            self,
            tool_name: str,
            server_name: str,
            arguments: dict | None,
            tool_use_id: str | None = None,
        ) -> str:
            del tool_name, server_name, arguments, tool_use_id
            return "tool-call-1"

        async def on_tool_progress(
            self,
            tool_call_id: str,
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            del tool_call_id, progress, total, message

        async def on_tool_complete(
            self,
            tool_call_id: str,
            success: bool,
            content,
            error: str | None,
        ) -> None:
            del tool_call_id, success, content, error

        async def get_tool_call_id_for_tool_use(self, tool_use_id: str) -> str | None:
            del tool_use_id
            return None

    connection = _Connection()
    tool_handler = _ToolHandler()
    runtime = ACPFilesystemRuntime(
        connection=connection,
        session_id="session",
        activation_reason="for test",
        tool_handler=tool_handler,
        permission_handler=_PermissionHandler(),
    )

    result = await runtime.write_text_file(
        {"path": "secret.txt", "content": "replacement"},
        tool_use_id="tool-use-1",
    )

    assert result.isError is True
    assert connection.read_calls == 0
    assert connection.write_calls == 0
    assert connection.update_calls == 0
    assert tool_handler.ensured == [
        (
            "tool-use-1",
            "write_text_file",
            ACP_FILESYSTEM_TOOL_SOURCE,
            {"path": "secret.txt", "content": "replacement"},
        ),
    ]
    assert tool_handler.denied == [
        ("write_text_file", ACP_FILESYSTEM_TOOL_SOURCE, "tool-use-1", "no")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["line", "limit"])
async def test_acp_filesystem_read_rejects_boolean_line_or_limit(field: str) -> None:
    class _Connection:
        read_calls = 0

        async def read_text_file(self, **_kwargs):
            self.read_calls += 1
            raise AssertionError("invalid read arguments must not reach ACP client")

    connection = _Connection()
    runtime = ACPFilesystemRuntime(
        connection=connection,
        session_id="session",
        activation_reason="for test",
    )

    result = await runtime.read_text_file({"path": "example.txt", field: True})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "must be an integer greater than or equal to 1" in result.content[0].text
    assert connection.read_calls == 0


@pytest.mark.asyncio
async def test_acp_filesystem_read_forwards_valid_line_and_limit() -> None:
    class _Response:
        content = "selected lines"

    class _Connection:
        def __init__(self) -> None:
            self.read_kwargs: dict[str, object] | None = None

        async def read_text_file(self, **kwargs):
            self.read_kwargs = kwargs
            return _Response()

    connection = _Connection()
    runtime = ACPFilesystemRuntime(
        connection=connection,
        session_id="session",
        activation_reason="for test",
    )

    result = await runtime.read_text_file({"path": "example.txt", "line": 2, "limit": 3})

    assert result.isError is False
    assert connection.read_kwargs == {
        "path": "example.txt",
        "session_id": "session",
        "line": 2,
        "limit": 3,
    }


def test_skill_reader_stamps_read_skill_as_skill() -> None:
    runtime = SkillReader([], logging.getLogger(__name__))

    assert runtime.tool.name == READ_SKILL_TOOL_NAME
    assert tool_source(runtime.tool) == SKILL_TOOL_SOURCE
