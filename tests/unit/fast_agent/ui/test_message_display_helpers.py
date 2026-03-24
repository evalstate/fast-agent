from mcp.types import CallToolRequest, CallToolRequestParams, ImageContent

from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.message_display_helpers import (
    build_tool_use_additional_message,
    build_user_message_display,
    extract_user_attachments,
    resolve_highlight_index,
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


def test_resolve_highlight_index_for_string_target() -> None:
    assert resolve_highlight_index(["shell", "web"], "web") == 1


def test_resolve_highlight_index_for_first_list_target() -> None:
    assert resolve_highlight_index(["shell", "web"], ["web", "shell"]) == 1


def test_resolve_highlight_index_only_uses_first_list_candidate() -> None:
    assert resolve_highlight_index(["shell", "web"], ["missing", "web"]) is None


def test_resolve_highlight_index_ignores_empty_string_target() -> None:
    assert resolve_highlight_index(["shell", "web"], "") is None


def test_resolve_highlight_index_handles_empty_candidate_list() -> None:
    assert resolve_highlight_index(["shell", "web"], []) is None


def test_resolve_highlight_index_returns_none_without_items() -> None:
    assert resolve_highlight_index(None, "shell") is None


def test_extract_user_attachments_includes_local_image_source_uri() -> None:
    image = ImageContent(
        type="image",
        data="ZmFrZQ==",
        mimeType="image/png",
    )
    image.meta = {"fast_agent_source_uri": "file:///tmp/photo.png"}
    message = PromptMessageExtended(
        role="user",
        content=[image],
    )

    assert extract_user_attachments(message) == ["image (file:///tmp/photo.png)"]


def test_build_user_message_display_prefers_original_text_metadata() -> None:
    image = ImageContent(type="image", data="ZmFrZQ==", mimeType="image/png")
    image.meta = {"fast_agent_source_uri": "file:///tmp/photo.png"}
    text = PromptMessageExtended.model_validate(
        {
            "role": "user",
            "content": [{"type": "text", "text": "can you see"}],
        }
    )
    text.content[0].meta = {"fast_agent_original_text": "can you see ^file:/tmp/photo.png"}

    message = PromptMessageExtended(role="user", content=[text.content[0], image])

    message_text, attachments = build_user_message_display([message])

    assert message_text == "can you see ^file:/tmp/photo.png"
    assert attachments == ["image (file:///tmp/photo.png)"]
