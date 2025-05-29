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
        "gemini25",  # Not yet turned on as it runs into token limits.
        "azure.gpt-4.1",
    ],
)
async def test_agent_with_image_prompt(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "default",
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
        "gpt-4.1-mini",  # OpenAI model
        "sonnet",  # Anthropic model
        "azure.gpt-4.1",
        "gemini25",
        #    "gemini2",
    ],
)
async def test_agent_with_mcp_image(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "default",
        instruction="You are a helpful AI Agent. Do not ask any questions.",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send the prompt and get the response

            response = await agent.send(
                "Use the image fetch tool to get the sample image and tell me the user name contained in this image?"
            )
            assert "evalstate" in response

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gemini25",  # Google Gemini model -> Works sometimes, but not always. DONE.
        # And Gemini 2.5 only works with a prompt that is a bit more specific.
        #    "gemini2",
    ],
)
async def test_agent_with_mcp_image_google(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "default",
        instruction="You are a helpful AI Agent.",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send the prompt and get the response

            response = await agent.send(
                "Use the image fetch tool to get the sample image. Then tell me the user name contained in this image."
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
        #    "gemini25",  # This currently uses the OpenAI format. Google Gemini cannot process PDFs with the OpenAI format. It can only do so with the native Gemini format.
    ],
)
async def test_agent_with_mcp_pdf(fast_agent, model_name):
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent. You have PDF support and summarisation capabilities.",
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
        "gemini25",  # This currently uses the OpenAI format. Google Gemini cannot process PDFs with the OpenAI format. It can only do so with the native Gemini format.
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
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent. You have vision capabilities and can analyse the image.",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response: PromptMessageMultipart = await agent.agent.generate(
                [
                    Prompt.user(
                        "Use the image fetch tool to get the sample image and tell me the user name contained in this image?"
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
        instruction="You are a helpful AI Agent. You have vision capabilities.",
        servers=["image_server"],
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response: PromptMessageMultipart = await agent.agent.generate(
                [
                    Prompt.user(
                        "Use the image fetch tool to get the sample image and tell me the user name contained in this image?"
                    )
                ]
            )
            # Import TextContent for type checking
            from mcp_agent.mcp_types import TextContent

            def is_thought_part(part_content):
                # Check if it's a TextContent and if its text starts with "thought" (case-insensitive)
                return isinstance(
                    part_content, TextContent
                ) and part_content.text.strip().lower().startswith("thought")

            # Filter out thought parts from the response content
            filtered_response_content = [
                part for part in response.content if not is_thought_part(part)
            ]

            # Filter out thought parts from the history message content
            # Assuming message_history[1] is the relevant assistant message after the tool call
            filtered_history_content = []
            if (
                len(agent.agent._llm.message_history) > 1
                and agent.agent._llm.message_history[1].content
            ):
                filtered_history_content = [
                    part
                    for part in agent.agent._llm.message_history[1].content
                    if not is_thought_part(part)
                ]

            # After filtering thoughts, we expect 3 semantic parts in the response:
            # 1. TextContent introduction for the image (from tool result)
            # 2. ImageContent (from tool result)
            # 3. TextContent with the final LLM answer
            assert 3 == len(filtered_response_content)
            assert (
                "evalstate" in response.all_text()
            )  # response.all_text() will include thoughts, which is fine for this check.

            # Ensure the filtered history also reflects the 3 semantic parts
            assert 3 == len(filtered_history_content)

    # Execute the agent function
    await agent_function()
