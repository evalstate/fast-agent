from mcp.types import CallToolRequest, CallToolRequestParams

from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.message_display_helpers import (
    build_tool_use_additional_message,
    tool_use_requests_file_read_access,
    tool_use_requests_shell_access,
)


def _tool_use_message(tool_name: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[],
        stop_reason=LlmStopReason.TOOL_USE,
        tool_calls={
            "1": CallToolRequest(
                params=CallToolRequestParams(name=tool_name, arguments={"command": "pwd"})
            )
        },
    )


def test_tool_use_requests_shell_access_for_execute_when_assumed() -> None:
    message = _tool_use_message("execute")

    assert tool_use_requests_shell_access(message, assume_execute_is_shell=True)


def test_tool_use_requests_shell_access_ignores_execute_without_context() -> None:
    message = _tool_use_message("execute")

    assert not tool_use_requests_shell_access(message)


def test_build_tool_use_additional_message_uses_shell_access_copy() -> None:
    message = _tool_use_message("execute")

    additional = build_tool_use_additional_message(message, shell_access=True)

    assert additional is not None
    assert additional.plain == "The assistant requested shell access"


def test_tool_use_requests_file_read_access_for_read_text_file() -> None:
    message = _tool_use_message("read_text_file")

    assert tool_use_requests_file_read_access(message)


def test_build_tool_use_additional_message_uses_file_read_copy() -> None:
    message = _tool_use_message("read_text_file")

    additional = build_tool_use_additional_message(message, file_read=True)

    assert additional is None


def test_build_tool_use_additional_message_pluralizes_file_reads() -> None:
    message = PromptMessageExtended(
        role="assistant",
        content=[],
        stop_reason=LlmStopReason.TOOL_USE,
        tool_calls={
            "1": CallToolRequest(
                params=CallToolRequestParams(name="read_text_file", arguments={"path": "/tmp/a"})
            ),
            "2": CallToolRequest(
                params=CallToolRequestParams(name="read_text_file", arguments={"path": "/tmp/b"})
            ),
        },
    )

    additional = build_tool_use_additional_message(message, file_read=True)

    assert additional is None
