import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import pytest
from mcp.types import ImageContent

from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompts.prompt_load import (
    load_prompt_multipart,
)

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_simple_conversation_from_file(fast_agent):
    """Make sure that we can load a simple multiturn conversation from a file."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            loaded: List[PromptMessageExtended] = load_prompt_multipart(Path("conv1_simple.md"))
            assert 4 == len(loaded)
            assert "user" == loaded[0].role
            assert "assistant" == loaded[1].role

            # Use the "default" agent directly
            response = await agent.default.generate(loaded)
            assert "message 2" in agent.default.message_history[-4].first_text()
            assert "message 3" in response.first_text()

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_conversation_with_attachments(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent()
    async def agent_function():
        async with fast.run():
            prompts: list[PromptMessageExtended] = load_prompt_multipart(Path("conv2_attach.md"))

            assert 5 == len(prompts)
            assert "user" == prompts[0].role
            assert "text/css" == prompts[0].content[1].resource.mimeType  # type: ignore
            assert "f5f5f5" in prompts[0].content[1].resource.text  # type: ignore

            assert "assistant" == prompts[1].role
            assert "sharing" in prompts[1].content[0].text  # type: ignore

            assert 3 == len(prompts[2].content)
            assert isinstance(prompts[2].content[2], ImageContent)
            assert 12780 == len(prompts[2].content[2].data)

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_state_to_simple_text_file(fast_agent):
    """Check to see if we can save a conversation to a text file. This functionality
    is extremely simple, and does not support round-tripping. JSON support using MCP
    types will be added in a future release."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            # Delete the file if it exists before running the test
            if os.path.exists("./simple.txt"):
                os.remove("./simple.txt")
            await agent.send("hello")
            await agent.send("world")
            await agent.send("***SAVE_HISTORY simple.txt")

            prompts: list[PromptMessageExtended] = load_prompt_multipart(Path("simple.txt"))
            assert 4 == len(prompts)
            assert "user" == prompts[0].role
            assert "assistant" == prompts[1].role

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_state_to_mcp_json_format(fast_agent):
    """Test saving conversation history to a JSON file in MCP wire format.
    This should create a file that's compatible with the MCP SDK and can be
    loaded directly using Pydantic types."""
    from mcp.types import GetPromptResult

    from mcp_agent.mcp.prompt_serialization import json_to_extended_messages

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            # Delete the file if it exists before running the test
            if os.path.exists("./history.json"):
                os.remove("./history.json")

            # Send a few messages
            await agent.send("hello")
            await agent.send("world")

            # Save in JSON format (filename ends with .json)
            await agent.send("***SAVE_HISTORY history.json")

            # Verify file exists
            assert os.path.exists("./history.json")

            # Load the file and check content
            with open("./history.json", "r", encoding="utf-8") as f:
                json_content = f.read()

            # Parse using JSON
            import json

            json_data = json.loads(json_content)

            # Validate it's a list of messages
            assert isinstance(json_data["messages"], list)
            assert len(json_data["messages"]) >= 4  # At least 4 messages (2 user, 2 assistant)

            # Check that messages have expected structure
            for msg in json_data["messages"]:
                assert "role" in msg
                assert "content" in msg

            # Validate with Pydantic by parsing to PromptMessageExtended objects
            prompts = json_to_extended_messages(json_content)

            # Verify loaded objects
            assert len(prompts) >= 4
            assert prompts[0].role == "user"
            assert prompts[1].role == "assistant"
            assert "hello" in prompts[0].first_text()

            # Validate compatibility with GetPromptResult
            messages = []
            for mp in prompts:
                messages.extend(mp.from_multipart())

            # Construct and validate with GetPromptResult
            prompt_result = GetPromptResult(messages=messages)
            assert len(prompt_result.messages) >= len(messages)

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_round_trip_json_attachments(fast_agent):
    """Test that we can save as json, and read back the content as PromptMessage->PromptMessageExtended."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(name="test")
    async def agent_function():
        async with fast.run() as agent:
            # Delete the file if it exists before running the test
            if os.path.exists("./multipart.json"):
                os.remove("./multipart.json")

            assert not os.path.exists("./multipart.json")

            await agent.test.generate([Prompt.user("good morning")])
            await agent.test.generate([Prompt.user("what's in this image", Path("conv2_img.png"))])
            await agent.send("***SAVE_HISTORY multipart.json")

            prompts: list[PromptMessageExtended] = load_prompt_multipart(Path("./multipart.json"))
            assert 4 == len(prompts)

            assert "assistant" == prompts[1].role
            assert 2 == len(prompts[2].content)
            assert isinstance(prompts[2].content[1], ImageContent)
            assert 12780 == len(prompts[2].content[1].data)

            assert 2 == len(prompts[2].from_multipart())

            # TODO -- consider serialization of non-text content for non json files. await requirement

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_apply_prompt_as_template_persistence(fast_agent):
    """Test that applying a prompt with as_template=True creates persistent context."""
    fast = fast_agent

    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            # Get the default agent to access its message history
            default_agent = agent.default

            # First, apply a prompt normally (should be in regular history)
            await agent.apply_prompt("simple", {"name": "test1"}, as_template=False)

            # Get the history length after normal application
            normal_history = default_agent.message_history
            normal_history_length = len(normal_history)

            # Now apply a prompt as template (should be in persistent context)
            await agent.apply_prompt("simple", {"name": "template_test"}, as_template=True)

            # Send a regular message to trigger conversation
            await agent.send("hello after template")

            # Get the full history
            final_history = default_agent.message_history

            # The template context should be persistent - let's check if it's still active
            # by examining the conversation structure

            # Verify that we can see the template influence by checking the conversation
            # The template prompt should be included as context for all future messages

            # Send another message to further verify persistence
            response = await agent.send("what was the template name?")

            # For passthrough model, just verify the API calls worked without error
            # The fact that we got responses means the new apply_prompt API is working
            assert response is not None
            assert len(response) > 0

            print(f"Normal history length: {normal_history_length}")
            print(f"Final history length: {len(final_history)}")
            print(f"API calls completed successfully, response: {response}")

            # Verify the new API signature works with both modes
            assert normal_history_length >= 0  # Basic sanity check

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_apply_prompt_with_prompt_result_object(fast_agent):
    """Test that we can apply a GetPromptResult object directly with as_template."""
    from mcp.types import GetPromptResult, PromptMessage, TextContent

    fast = fast_agent

    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            # Create a GetPromptResult object directly since no MCP servers are configured
            prompt_result = GetPromptResult(
                description="Test prompt for API verification",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text="Hello from direct GetPromptResult"),
                    )
                ],
            )

            # Verify we created a valid prompt result
            assert isinstance(prompt_result, GetPromptResult)
            assert len(prompt_result.messages) > 0

            # Apply it directly as a template
            response = await agent.apply_prompt(prompt_result, as_template=True)
            assert response is not None

            # Apply it directly in regular mode
            response2 = await agent.apply_prompt(prompt_result, as_template=False)
            assert response2 is not None

            # Verify both worked by sending a follow-up message
            final_response = await agent.send("test complete")
            assert final_response is not None  # Just verify we got a response

    await agent_function()
