# integration_tests/mcp_agent/test_agent_with_image.py
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mcp_agent.core.prompt import Prompt

if TYPE_CHECKING:
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",  # OpenAI model
        "sonnet",  # Anthropic model
        "gemini25",
    ],
)
async def test_agent_with_image_prompt(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            prompt = Prompt.user(
                "what is the user name contained in this image?",
                Path("image.png"),
            )
            response = await agent.send(prompt)

            assert "evalstate" in response

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",  # OpenAI
        "sonnet",  # Anthropic
        "gemini25",
    ],
)
async def test_agent_with_mcp_image(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent. Do not ask any questions.",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send the prompt and get the response

            response = await agent.send(
                "Use the image fetch tool to get the sample image and tell me what user name contained in this image?"
            )
            assert "evalstate" in response

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",  # OpenAI model
        "haiku35",  # Anthropic model
    ],
)
async def test_agent_with_mcp_pdf(fast_agent, model_name):
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send the prompt and get the response

            response = await agent.send(
                "Can you summarise the sample PDF, make sure it includes the product name in the summary"
            )
            assert "fast-agent" in response.lower()

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",  # OpenAI model
        "haiku35",  # Anthropic model
    ],
)
async def test_agent_with_pdf_prompt(fast_agent, model_name):
    """Test that the agent can process a PDF document and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response = await agent.send(
                message=Prompt.user(
                    "summarize this document - include the company that made it",
                    Path("sample.pdf"),
                )
            )

            # Send the prompt and get the response
            assert "llmindset".lower() in response.lower()

    # Execute the agent function
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "sonnet",  # Anthropic model
    ],
)
async def test_agent_includes_tool_results_in_multipart_result_anthropic(fast_agent, model_name):
    """Test that the agent can process a PDF document and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response: PromptMessageMultipart = await agent.agent.generate(
                [
                    Prompt.user(
                        "Use the image fetch tool to get the sample image and tell me what user name contained in this image?"
                    )
                ]
            )
            # we are expecting response message, tool call, tool response (1* text, 1 * image), final response
            assert 4 == len(response.content)
            assert "evalstate" in response.all_text()
            assert 4 == len(agent.agent._llm.message_history[1].content)

    # Execute the agent function
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",  # OpenAI model
    ],
)
async def test_agent_includes_tool_results_in_multipart_result_openai(fast_agent, model_name):
    """Test that the agent can process a PDF document and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response: PromptMessageMultipart = await agent.agent.generate(
                [
                    Prompt.user(
                        "Use the image fetch tool to get the sample image and tell me what user name contained in this image?"
                    )
                ]
            )
            # OpenAI returns None for the first function call - different from Anthropic
            # we are expecting a  tool call, tool response (1* text, 1 * image), final response
            assert 3 == len(response.content)
            assert "evalstate" in response.all_text()
            # make sure it's available in the history
            assert 3 == len(agent.agent._llm.message_history[1].content)

    # Execute the agent function
    await agent_function()
