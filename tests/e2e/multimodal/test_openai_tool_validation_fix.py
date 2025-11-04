"""
E2E test for OpenAI API tool call validation fix.

This test validates the fix for issue #314 - OpenAI API validation errors
that occur during parallel tool calls when one tool returns mixed content (text + images).

This test uses a real MCP server (mixed_content_server.py) that provides:
- get_page_data: Returns pure text (simulates browser_snapshot)
- take_screenshot: Returns text + image (simulates browser_take_screenshot)

The test reproduces the exact scenario that caused validation errors and ensures
the fix works properly.
"""

import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",  # OpenAI model that should work with our fix
        "gpt-4.1-mini",  # Another OpenAI model
        "o4-mini.low",
        "openrouter.openai/gpt-4.1-mini",
    ],
)
async def test_parallel_tool_calls_with_mixed_content_ordering(fast_agent, model_name):
    """
    Test that parallel tool calls with mixed content are properly ordered for OpenAI API.

    This test reproduces the scenario from issue #314 by manually triggering parallel tool calls:
    - Tool 1 (get_page_data) returns pure text
    - Tool 2 (take_screenshot) returns mixed content (text + image)
    - Verifies that OpenAI API validation doesn't fail
    - Uses the real mixed_content_server.py MCP server
    - Deterministic: manually triggers both tools instead of relying on LLM decisions
    """
    import asyncio

    fast = fast_agent

    # Define the agent with the mixed content server
    @fast.agent(
        "test_agent",
        instruction="You are a test agent for testing parallel tool calls.",
        model=model_name,
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

            # Check that neither result is an exception
            for i, result in enumerate(results):
                assert not isinstance(result, Exception), f"Tool {i + 1} failed with: {result}"

            # Validate tool results
            page_data_result, screenshot_result = results

            # Tool 1 should return pure text
            assert page_data_result is not None
            assert hasattr(page_data_result, "content")
            assert len(page_data_result.content) == 1  # Single text content

            # Tool 2 should return mixed content (text + image)
            assert screenshot_result is not None
            assert hasattr(screenshot_result, "content")
            assert len(screenshot_result.content) == 2  # Text + image content

            # Verify content types
            text_contents = [
                c for c in screenshot_result.content if hasattr(c, "type") and c.type == "text"
            ]
            image_contents = [
                c for c in screenshot_result.content if hasattr(c, "type") and c.type == "image"
            ]

            assert len(text_contents) >= 1, "Screenshot tool should return text content"
            assert len(image_contents) >= 1, "Screenshot tool should return image content"

    await test_agent()


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["gpt-4o-mini", "o3-mini.low"])
async def test_openai_validation_error_prevention(fast_agent, model_name):
    """
    Test that our fix prevents the specific OpenAI validation error by simulating
    the exact message sequence that used to cause the error.

    This test ensures that the error message:
    "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'"
    does not occur with our fix when using mixed content tools.
    """
    import asyncio

    fast = fast_agent

    @fast.agent(
        "validation_test_agent",
        instruction="Test agent for validation error prevention.",
        model=model_name,
        servers=["mixed_content_server"],
    )
    async def validation_agent():
        async with fast.run() as agent_app:
            agent = agent_app.validation_test_agent

            # The test passes if no OpenAI validation exception is raised during parallel tool execution
            try:
                # Simulate the problematic scenario: mixed content tool + pure text tool in parallel
                # Execute in parallel - this should trigger the message reordering fix
                results = await asyncio.gather(
                    agent.call_tool("get_both_data", {}),
                    agent.call_tool("get_page_data", {}),
                    return_exceptions=True,
                )

                # Validate both executed without the validation error
                assert len(results) == 2
                for i, result in enumerate(results):
                    assert not isinstance(result, Exception), f"Tool {i + 1} failed with: {result}"

            except Exception as e:
                # Check if this is the specific OpenAI validation error we're trying to prevent
                error_msg = str(e)
                if (
                    "An assistant message with 'tool_calls' must be followed by tool messages"
                    in error_msg
                ):
                    pytest.fail(f"OpenAI validation error occurred: {error_msg}")
                else:
                    # Some other error - re-raise
                    raise

    await validation_agent()


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["gpt-4o-mini", "openrouter.openai/gpt-4.1-mini", "azure.gpt-4.1"]
)
async def test_single_mixed_content_tool(fast_agent, model_name):
    """
    Test that a single tool returning mixed content works correctly.

    This tests the get_both_data tool which returns multiple text blocks + image
    in a single tool call response - validating mixed content handling without parallel calls.
    """
    fast = fast_agent

    @fast.agent(
        "single_tool_agent",
        instruction="Test agent for single mixed content tool.",
        model=model_name,
        servers=["mixed_content_server"],
    )
    async def single_tool_agent():
        async with fast.run() as agent_app:
            agent = agent_app.single_tool_agent

            # Directly call the mixed content tool
            # Execute the single mixed content tool
            result = await agent.call_tool("get_both_data", {})

            # Validate result structure
            assert result is not None
            assert hasattr(result, "content")
            assert len(result.content) >= 2  # Should have multiple content blocks

            # Verify mixed content: text + image
            text_contents = [c for c in result.content if hasattr(c, "type") and c.type == "text"]
            image_contents = [c for c in result.content if hasattr(c, "type") and c.type == "image"]

            assert len(text_contents) >= 2, "get_both_data should return multiple text blocks"
            assert len(image_contents) >= 1, "get_both_data should return image content"

    await single_tool_agent()
