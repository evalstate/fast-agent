"""
E2E smoke tests for MCP tools that return mixed text/image content.

These tests use direct MCP tool calls against a real MCP server. They verify that
mixed-content tool results are returned intact through the agent/MCP stack.

This test uses a real MCP server (mixed_content_server.py) that provides:
- get_page_data: Returns pure text (simulates browser_snapshot)
- take_screenshot: Returns text + image (simulates browser_take_screenshot)

Issue #314 involved OpenAI request validation after mixed-content tool results.
Direct ``call_tool`` coverage does not exercise provider request serialization;
that behavior belongs in provider/converter tests.
"""

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent


def _require_tool_result(value: CallToolResult | BaseException, label: str) -> CallToolResult:
    if isinstance(value, BaseException):
        pytest.fail(f"{label} failed with: {value}")
    return value


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parallel_tool_calls_with_mixed_content_ordering(fast_agent):
    """
    Test that parallel direct tool calls preserve mixed content.

    - Tool 1 (get_page_data) returns pure text
    - Tool 2 (take_screenshot) returns mixed content (text + image)
    - Uses the real mixed_content_server.py MCP server
    """
    import asyncio

    fast = fast_agent

    # Define the agent with the mixed content server
    @fast.agent(
        "test_agent",
        instruction="You are a test agent for testing parallel tool calls.",
        model="passthrough",
        servers=["mixed_content_server"],
    )
    async def test_agent():
        async with fast.run() as agent_app:
            # Get the actual agent instance
            agent = agent_app.test_agent

            # Manually trigger parallel tool calls - this is the exact scenario that caused issue #314
            # Execute both tools in parallel - this triggers the message ordering issue
            # Tool 1: Returns pure text
            task1 = agent.call_tool("get_page_data", {})
            # Tool 2: Returns mixed content (text + image)
            task2 = agent.call_tool("take_screenshot", {})

            # Wait for both to complete - this creates the mixed content scenario
            results = await asyncio.gather(task1, task2, return_exceptions=True)

            # Validate both tools executed successfully
            assert len(results) == 2

            # Validate tool results
            page_data_result = _require_tool_result(results[0], "get_page_data")
            screenshot_result = _require_tool_result(results[1], "take_screenshot")

            # Tool 1 should return pure text
            assert len(page_data_result.content) == 1  # Single text content

            # Tool 2 should return mixed content (text + image)
            assert len(screenshot_result.content) == 2  # Text + image content

            # Verify content types
            text_contents = [c for c in screenshot_result.content if isinstance(c, TextContent)]
            image_contents = [c for c in screenshot_result.content if isinstance(c, ImageContent)]

            assert len(text_contents) >= 1, "Screenshot tool should return text content"
            assert len(image_contents) >= 1, "Screenshot tool should return image content"

    await test_agent()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parallel_mixed_and_text_tool_results(fast_agent):
    """
    Test parallel direct tool execution with one mixed-content and one text result.
    """
    import asyncio

    fast = fast_agent

    @fast.agent(
        "validation_test_agent",
        instruction="Test agent for mixed content tool result handling.",
        model="passthrough",
        servers=["mixed_content_server"],
    )
    async def validation_agent():
        async with fast.run() as agent_app:
            agent = agent_app.validation_test_agent

            results = await asyncio.gather(
                agent.call_tool("get_both_data", {}),
                agent.call_tool("get_page_data", {}),
                return_exceptions=True,
            )

            assert len(results) == 2
            _require_tool_result(results[0], "get_both_data")
            _require_tool_result(results[1], "get_page_data")

    await validation_agent()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_single_mixed_content_tool(fast_agent):
    """
    Test that a single tool returning mixed content works correctly.

    This tests the get_both_data tool which returns multiple text blocks + image
    in a single tool call response - validating mixed content handling without parallel calls.
    """
    fast = fast_agent

    @fast.agent(
        "single_tool_agent",
        instruction="Test agent for single mixed content tool.",
        model="passthrough",
        servers=["mixed_content_server"],
    )
    async def single_tool_agent():
        async with fast.run() as agent_app:
            agent = agent_app.single_tool_agent

            # Directly call the mixed content tool
            # Execute the single mixed content tool
            result = await agent.call_tool("get_both_data", {})

            # Validate result structure
            assert len(result.content) >= 2  # Should have multiple content blocks

            # Verify mixed content: text + image
            text_contents = [c for c in result.content if isinstance(c, TextContent)]
            image_contents = [c for c in result.content if isinstance(c, ImageContent)]

            assert len(text_contents) >= 2, "get_both_data should return multiple text blocks"
            assert len(image_contents) >= 1, "get_both_data should return image content"

    await single_tool_agent()
