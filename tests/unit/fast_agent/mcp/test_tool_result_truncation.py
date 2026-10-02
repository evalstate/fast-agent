import pytest
from mcp_types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from fast_agent.mcp.helpers.content_helpers import canonicalize_tool_result_content_for_llm
from fast_agent.mcp.tool_result_truncation import bound_tool_result_for_llm


async def _no_retained_copy() -> None:
    return None


async def _bound(result: CallToolResult, byte_limit: int) -> CallToolResult:
    return await bound_tool_result_for_llm(result, byte_limit=byte_limit, retain=_no_retained_copy)


@pytest.mark.asyncio
async def test_tool_result_within_budget_is_unchanged() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="small")])

    assert await _bound(result, 5) is result


@pytest.mark.asyncio
async def test_tool_result_truncates_canonical_structured_content() -> None:
    image = ImageContent(type="image", data="aGVsbG8=", mime_type="image/png")
    result = CallToolResult(
        content=[TextContent(type="text", text="ignored"), image],
        structured_content={"value": "x" * 100},
    )

    truncated = await _bound(result, 40)
    canonical = canonicalize_tool_result_content_for_llm(truncated)

    assert truncated is not result
    assert truncated.structured_content is None
    assert len(canonical) == 2
    assert isinstance(canonical[0], TextContent)
    assert canonical[0].text.startswith('{"value":"xxxxxxxxxx')
    assert "[Tool result truncated:" in canonical[0].text
    assert canonical[0].text.endswith('xxxxxxxxxxxxxxxxxx"}')
    assert canonical[1] == image


@pytest.mark.asyncio
async def test_tool_result_uses_one_budget_across_text_blocks() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="a" * 30),
            TextContent(type="text", text="b" * 30),
        ]
    )

    truncated = await _bound(result, 40)

    assert len(truncated.content) == 1
    content = truncated.content[0]
    assert isinstance(content, TextContent)
    assert content.text.startswith("a" * 20)
    assert content.text.endswith("b" * 20)
    assert "of 61 bytes" in content.text


@pytest.mark.asyncio
async def test_tool_result_truncates_embedded_text_resources() -> None:
    resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri="file:///large.txt",
            text="x" * 100,
            mime_type="text/plain",
        ),
    )
    result = CallToolResult(content=[resource])

    truncated = await _bound(result, 40)

    assert len(truncated.content) == 1
    content = truncated.content[0]
    assert isinstance(content, TextContent)
    assert "[Tool result truncated:" in content.text
    assert "of 100 bytes" in content.text


@pytest.mark.asyncio
async def test_retained_copy_guidance_replaces_default_advice() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="x" * 100)])

    async def retain() -> str:
        return "The complete result is at /tmp/full.txt."

    truncated = await bound_tool_result_for_llm(result, byte_limit=40, retain=retain)

    content = truncated.content[0]
    assert isinstance(content, TextContent)
    assert "The complete result is at /tmp/full.txt." in content.text
    assert "narrower query" not in content.text
