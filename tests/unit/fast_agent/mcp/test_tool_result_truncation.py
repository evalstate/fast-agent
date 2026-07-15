from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from fast_agent.mcp.helpers.content_helpers import canonicalize_tool_result_content_for_llm
from fast_agent.mcp.tool_result_truncation import truncate_tool_result_for_llm


def test_tool_result_within_budget_is_unchanged() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="small")])

    assert truncate_tool_result_for_llm(result, byte_limit=5) is result


def test_tool_result_truncates_canonical_structured_content() -> None:
    image = ImageContent(type="image", data="aGVsbG8=", mimeType="image/png")
    result = CallToolResult(
        content=[TextContent(type="text", text="ignored"), image],
        structuredContent={"value": "x" * 100},
    )

    truncated = truncate_tool_result_for_llm(result, byte_limit=40)
    canonical = canonicalize_tool_result_content_for_llm(truncated)

    assert truncated is not result
    assert truncated.structuredContent is None
    assert len(canonical) == 2
    assert isinstance(canonical[0], TextContent)
    assert canonical[0].text.startswith('{"value":"xxxxxxxxxx')
    assert "[Tool result truncated:" in canonical[0].text
    assert canonical[0].text.endswith('xxxxxxxxxxxxxxxxxx"}')
    assert canonical[1] == image


def test_tool_result_uses_one_budget_across_text_blocks() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="a" * 30),
            TextContent(type="text", text="b" * 30),
        ]
    )

    truncated = truncate_tool_result_for_llm(result, byte_limit=40)

    assert len(truncated.content) == 1
    content = truncated.content[0]
    assert isinstance(content, TextContent)
    assert content.text.startswith("a" * 20)
    assert content.text.endswith("b" * 20)
    assert "of 61 bytes" in content.text


def test_tool_result_truncates_embedded_text_resources() -> None:
    resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("file:///large.txt"),
            text="x" * 100,
            mimeType="text/plain",
        ),
    )
    result = CallToolResult(content=[resource])

    truncated = truncate_tool_result_for_llm(result, byte_limit=40)

    assert len(truncated.content) == 1
    content = truncated.content[0]
    assert isinstance(content, TextContent)
    assert "[Tool result truncated:" in content.text
    assert "of 100 bytes" in content.text
