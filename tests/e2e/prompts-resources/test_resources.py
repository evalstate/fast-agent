import pytest


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "haiku",
    ],
)
async def test_using_resource_blob(fast_agent, model_name):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
        servers=["prompt_server"],
    )
    async def agent_function():
        async with fast.run() as agent:
            assert "fast-agent" in await agent.with_resource(
                "Summarise this PDF please, be sure to include the product name",
                "resource://fast-agent/sample.pdf",
                "prompt_server",
            )

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "haiku",
    ],
)
async def test_using_resource_text(fast_agent, model_name):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
        servers=["prompt_server"],
    )
    async def agent_function():
        async with fast.run() as agent:
            answer = await agent.agent.with_resource(
                "What colour are buttons in this file?",
                "resource://fast-agent/style.css",
                "prompt_server",
            )
            assert "white" in answer.lower()

    await agent_function()
