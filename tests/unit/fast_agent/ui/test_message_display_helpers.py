import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, ImageContent, TextContent

from fast_agent.constants import FAST_AGENT_SAFETY_DETAILS
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.message_display_helpers import (
    _content_metadata,
    build_safety_additional_message,
    build_tool_use_additional_message,
    build_user_message_display,
    build_user_message_image_previews,
    extract_user_attachments,
    extract_user_local_image_previews,
    resolve_highlight_indexes,
    tool_use_requests_file_read_access,
    tool_use_requests_process_lifecycle,
    tool_use_requests_shell_access,
)


class _BrokenMetadataContent:
    @property
    def meta(self) -> object:
        raise AttributeError("meta unavailable")


class _NoMetadataContent:
    pass


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


def _tool_use_message_with_names(*tool_names: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="assistant",
        content=[],
        stop_reason=LlmStopReason.TOOL_USE,
        tool_calls={
            str(index): CallToolRequest(params=CallToolRequestParams(name=tool_name, arguments={}))
            for index, tool_name in enumerate(tool_names, start=1)
        },
    )


def test_tool_use_requests_shell_access_for_execute_when_assumed() -> None:
    message = _tool_use_message("execute")

    assert tool_use_requests_shell_access(message, assume_execute_is_shell=True)


def test_tool_use_requests_shell_access_ignores_execute_without_context() -> None:
    message = _tool_use_message("execute")

    assert not tool_use_requests_shell_access(message)


def test_tool_use_requests_shell_access_normalizes_explicit_shell_tool_name() -> None:
    message = _tool_use_message("server.execute")

    assert tool_use_requests_shell_access(message, shell_tool_name="execute")


def test_tool_use_requests_shell_access_rejects_mixed_tool_calls() -> None:
    message = _tool_use_message_with_names("bash", "read_text_file")

    assert not tool_use_requests_shell_access(message)


def test_build_tool_use_additional_message_uses_shell_access_copy() -> None:
    message = _tool_use_message("execute")

    additional = build_tool_use_additional_message(message, shell_access=True)

    assert additional is not None
    assert additional.plain == "The assistant requested shell access"


def test_tool_use_requests_file_read_access_for_read_text_file() -> None:
    message = _tool_use_message("read_text_file")

    assert tool_use_requests_file_read_access(message)


def test_tool_use_requests_file_read_access_normalizes_explicit_read_tool_name() -> None:
    message = _tool_use_message("server.local_read")

    assert tool_use_requests_file_read_access(message, read_tool_name="local_read")


def test_tool_use_requests_file_read_access_rejects_mixed_tool_calls() -> None:
    message = _tool_use_message_with_names("read_text_file", "bash")

    assert not tool_use_requests_file_read_access(message)


def test_build_tool_use_additional_message_uses_file_read_copy() -> None:
    message = _tool_use_message("read_text_file")

    additional = build_tool_use_additional_message(message, file_read=True)

    assert additional is None


def test_process_lifecycle_tool_use_omits_generic_additional_message() -> None:
    message = _tool_use_message_with_names("poll_process")

    assert tool_use_requests_process_lifecycle(message)
    assert build_tool_use_additional_message(message) is None


def test_mixed_process_and_other_tool_use_keeps_generic_additional_message() -> None:
    message = _tool_use_message_with_names("poll_process", "read_text_file")

    assert not tool_use_requests_process_lifecycle(message)
    additional = build_tool_use_additional_message(message)
    assert additional is not None
    assert additional.plain == "The assistant requested tool calls"


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


def test_build_safety_additional_message_includes_anthropic_refusal_category() -> None:
    message = PromptMessageExtended(
        role="assistant",
        content=[],
        stop_reason=LlmStopReason.SAFETY,
        channels={
            FAST_AGENT_SAFETY_DETAILS: [
                TextContent(type="text", text='{"type": "refusal", "category": "cyber"}')
            ]
        },
    )

    additional = build_safety_additional_message(message)

    assert additional is not None
    assert additional.plain == "\n\nRequest refused by safety classifier (cyber)."


def test_build_safety_additional_message_handles_missing_category() -> None:
    message = PromptMessageExtended(
        role="assistant",
        content=[],
        stop_reason=LlmStopReason.SAFETY,
    )

    additional = build_safety_additional_message(message)

    assert additional is not None
    assert additional.plain == "\n\nRequest refused by safety classifier."


def test_resolve_highlight_indexes_for_string_target() -> None:
    assert resolve_highlight_indexes(["shell", "web"], "web") == [1]


def test_resolve_highlight_indexes_for_multiple_targets() -> None:
    assert resolve_highlight_indexes(["shell", "web"], ["web", "shell"]) == [0, 1]


def test_resolve_highlight_indexes_ignores_missing_candidates() -> None:
    assert resolve_highlight_indexes(["shell", "web"], ["missing", "web"]) == [1]


def test_resolve_highlight_indexes_ignores_empty_string_target() -> None:
    assert resolve_highlight_indexes(["shell", "web"], "") == []


def test_resolve_highlight_indexes_handles_empty_candidate_list() -> None:
    assert resolve_highlight_indexes(["shell", "web"], []) == []


def test_resolve_highlight_indexes_returns_empty_without_items() -> None:
    assert resolve_highlight_indexes(None, "shell") == []


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


def test_extract_user_local_image_previews_only_includes_file_sources() -> None:
    local_image = ImageContent(type="image", data="ZmFrZQ==", mimeType="image/png")
    local_image.meta = {"fast_agent_source_uri": "file:///tmp/photo.png"}
    remote_image = ImageContent(type="image", data="ZmFrZQ==", mimeType="image/png")
    remote_image.meta = {"fast_agent_source_uri": "https://example.test/photo.png"}
    inline_image = ImageContent(type="image", data="ZmFrZQ==", mimeType="image/png")
    message = PromptMessageExtended(
        role="user",
        content=[local_image, remote_image, inline_image],
    )

    previews = extract_user_local_image_previews(message)

    assert len(previews) == 1
    assert previews[0].artifact.mime_type == "image/png"


def test_build_user_message_image_previews_combines_messages() -> None:
    first_image = ImageContent(type="image", data="ZmFrZQ==", mimeType="image/png")
    first_image.meta = {"fast_agent_source_uri": "file:///tmp/one.png"}
    second_image = ImageContent(type="image", data="ZmFrZQ==", mimeType="image/png")
    second_image.meta = {"fast_agent_source_uri": "file:///tmp/two.png"}

    previews = build_user_message_image_previews(
        [
            PromptMessageExtended(role="user", content=[first_image]),
            PromptMessageExtended(role="user", content=[second_image]),
        ]
    )

    assert len(previews) == 2


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


def test_content_metadata_returns_none_without_metadata_protocol() -> None:
    assert _content_metadata(_NoMetadataContent()) is None


def test_content_metadata_does_not_swallow_broken_meta_property() -> None:
    with pytest.raises(AttributeError, match="meta unavailable"):
        _content_metadata(_BrokenMetadataContent())
