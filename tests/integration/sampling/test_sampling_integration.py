import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sampling_feature(fast_agent):
    """Test that the default message is returned."""
    fast = fast_agent

    # Define the agent
    @fast.agent(servers=["sampling_test"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent("***CALL_TOOL sample")
            assert "hello, world" in result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sampling_config(fast_agent):
    """Test that the config loads sampling configuration."""
    fast = fast_agent

    @fast.agent(name="empty")
    async def agent_function():
        async with fast.run():
            assert "passthrough" == fast.context.config.mcp.servers["sampling_test"].sampling.model

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sampling_passback(fast_agent):
    """Test that the passthrough LLM is hooked up"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(servers=["sampling_test"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent('***CALL_TOOL sample {"to_sample": "llmindset"}')
            assert "llmindset" in result
            assert '"iserror": true' not in result.lower()

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sampling_multi_message_passback(fast_agent):
    """Test that the passthrough LLM is hooked up"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(servers=["sampling_test"])
    async def agent_function():
        async with fast.run() as agent:
            result = await agent("***CALL_TOOL sample_many")
            assert "message 1" in result
            assert "message 2" in result

    await agent_function()
