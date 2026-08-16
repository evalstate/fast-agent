"""
Integration tests for sampling with tools feature.

These tests verify that:
1. Sampling requests with tools are properly handled
2. Backward compatibility with sampling without tools
3. Tool result handling in multi-turn conversations
"""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sampling_with_tools_basic(fast_agent):
    """Test that sampling with tools works - server can send tools in request."""
    fast = fast_agent

    @fast.agent(servers=["sampling_tools_test"])
    async def agent_function():
        async with fast.run() as agent:
            # Use ***CALL_TOOL to directly invoke the server tool
            result = await agent('***CALL_TOOL test_sampling_with_tools {"message": "hello"}')
            # Should complete without error
            assert "Sampling completed" in result or "stopReason" in result
            assert "error" not in result.lower() or "iserror" not in result.lower()

            status = (await agent.default.get_server_status())["sampling_tools_test"]
            assert status.protocol_mode == "modern"
            assert status.protocol_era == "modern"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sampling_without_tools_backward_compatible(fast_agent):
    """Test that sampling without tools still works (backward compatibility)."""
    fast = fast_agent

    @fast.agent(servers=["sampling_tools_test"])
    async def agent_function():
        async with fast.run() as agent:
            # Use ***CALL_TOOL to directly invoke the server tool
            result = await agent(
                '***CALL_TOOL test_sampling_without_tools {"message": "hello world"}'
            )
            # Should complete without error
            assert "Response" in result
            assert "hello world" in result  # Passthrough echoes

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_result_handling(fast_agent):
    """Test multi-turn tool conversation with tool results."""
    fast = fast_agent

    @fast.agent(servers=["sampling_tools_test"])
    async def agent_function():
        async with fast.run() as agent:
            # Use ***CALL_TOOL to directly invoke the server tool
            result = await agent("***CALL_TOOL test_tool_result_handling")
            assert "Multi-turn completed" in result
            assert "echo: hello" in result

    await agent_function()
