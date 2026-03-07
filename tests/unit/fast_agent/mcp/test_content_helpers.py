from __future__ import annotations

import base64

from mcp.types import CallToolResult, ImageContent, TextContent

from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    tool_result_text_for_llm,
)


class _LoggerSpy:
    def __init__(self) -> None:
        self.warning_calls: list[tuple[str, dict[str, object]]] = []

    def warning(self, message: str, **data: object) -> None:
        self.warning_calls.append((message, data))


def test_canonicalize_tool_result_content_preserves_original_content_without_structured_content() -> None:
    image_data = base64.b64encode(b"fake-image").decode("utf-8")
    text_block = TextContent(type="text", text="hello")
    image_block = ImageContent(type="image", data=image_data, mimeType="image/jpeg")
    result = CallToolResult(content=[text_block, image_block], isError=False)

    canonical = canonicalize_tool_result_content_for_llm(result)

    assert canonical is not result.content
    assert canonical[0] is text_block
    assert canonical[1] is image_block


def test_canonicalize_tool_result_content_prefers_structured_content_and_preserves_non_text() -> None:
    image_data = base64.b64encode(b"fake-image").decode("utf-8")
    image_block = ImageContent(type="image", data=image_data, mimeType="image/jpeg")
    result = CallToolResult(
        content=[TextContent(type="text", text="stale summary"), image_block],
        isError=False,
    )
    setattr(result, "structuredContent", {"z": 3, "a": 1})

    canonical = canonicalize_tool_result_content_for_llm(result)

    assert len(canonical) == 2
    assert isinstance(canonical[0], TextContent)
    assert canonical[0].text == '{"a":1,"z":3}'
    assert canonical[1] is image_block


def test_canonicalize_tool_result_content_warns_for_multiple_text_blocks() -> None:
    logger = _LoggerSpy()
    result = CallToolResult(
        content=[
            TextContent(type="text", text="first"),
            TextContent(type="text", text="second"),
        ],
        isError=False,
    )
    setattr(result, "structuredContent", {"fresh": True})

    canonicalize_tool_result_content_for_llm(result, logger=logger, source="test.helper")

    assert len(logger.warning_calls) == 1
    message, data = logger.warning_calls[0]
    assert "structuredContent" in message
    assert data == {"text_block_count": 2, "source": "test.helper"}


def test_tool_result_text_for_llm_uses_structured_content_json() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="stale summary")],
        isError=False,
    )
    setattr(result, "structuredContent", {"b": 2, "a": 1})

    text = tool_result_text_for_llm(result)

    assert text == '{"a":1,"b":2}'
