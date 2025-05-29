# integration_tests/mcp_agent/test_agent_with_image.py
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",  # OpenAI model
        "haiku35",  # Anthropic model
        "gemini25",  # Google Gemini model -> Works. DONE.
    ],
)
async def test_agent_with_simple_prompt(fast_agent, model_name):
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
            response = await agent.apply_prompt("simple", {"name": "llmindset"})
            assert "llmindset" in response

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",  # OpenAI model
        "haiku35",  # Anthropic model
        # "gemini25",  # Google Gemini model -> This involves opening a PDF. It is not supported by Google Gemini with the OpenAI format. Unless the format is changed to the native Gemini format, this will not work.
    ],
)
async def test_agent_with_prompt_attachment(fast_agent, model_name):
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
            response = await agent.apply_prompt("with_attachment")
            assert any(term in response.lower() for term in ["llmindset", "fast-agent"])

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",  # OpenAI model
        "haiku35",  # Anthropic model
        "gemini25",  # Google Gemini model -> Works. DONE.
    ],
)
async def test_agent_multiturn_prompt(fast_agent, model_name):
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
            response = await agent.agent.apply_prompt("multiturn")
            assert "testcaseok" in response.lower()

    await agent_function()
