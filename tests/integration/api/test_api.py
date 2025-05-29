import pytest

from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_api_with_simple_prompt(fast_agent):
    """Test that the agent can process a simple prompt using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
    )
    async def agent_function():
        async with fast.run() as agent:
            assert "test1" in await agent.agent1.send("test1")
            assert "test2" in await agent["agent1"].send("test2")
            assert "test3" in await agent.send("test3")
            assert "test4" in await agent("test4")
            assert "test5" in await agent.send("test5", "agent1")
            assert "test6" in await agent("test6", "agent1")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_api_with_prompt_messages(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
    )
    async def agent_function():
        async with fast.run() as agent:
            assert "test1" in await agent.agent1.send(Prompt.user("test1"))

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_api_with_basic_playback(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
        model="playback",
        servers=["prompts"],
    )
    async def agent_function():
        async with fast.run() as agent:
            await agent.agent1.apply_prompt("playback")
            assert "assistant1" in await agent.agent1.send("ignored")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_api_with_default_calls(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
        model="passthrough",
    )
    async def agent_function():
        async with fast.run() as agent:
            assert "message 1" == await agent("message 1")
            assert "message 2" == await agent["agent1"]("message 2")

        # assert "assistant1" in await agent.agent1.send("ignored")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mixed_message_types(fast_agent):
    """Test that the agent can handle mixed message types seamlessly."""
    from mcp.types import PromptMessage, TextContent

    from mcp_agent.core.prompt import Prompt
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
        model="passthrough",
    )
    async def agent_function():
        async with fast.run() as agent:
            # Test with string
            assert "string message" == await agent.send("string message")

            # Test with PromptMessage
            prompt_message = PromptMessage(
                role="user", content=TextContent(type="text", text="prompt message")
            )
            assert "prompt message" == await agent.send(prompt_message)

            # Test with PromptMessageMultipart
            multipart = PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text="multipart message")]
            )
            assert "multipart message" == await agent.send(multipart)

            # Test message history access
            response = await agent.send("checking history")
            # Verify agent's message history is accessible and contains our messages
            message_history = agent.agent1.message_history

            # Basic assertions
            assert len(message_history) >= 8  # 4 user messages + 4 assistant responses
            assert all(isinstance(msg, PromptMessageMultipart) for msg in message_history)

            # Create role/content pairs for easier verification
            message_pairs = [(msg.role, msg.first_text()) for msg in message_history]

            # Check for specific messages with correct roles
            user_messages = [text for role, text in message_pairs if role == "user"]
            assistant_messages = [text for role, text in message_pairs if role == "assistant"]

            # Check our specific user messages are there
            assert "string message" in user_messages
            assert "prompt message" in user_messages
            assert "multipart message" in user_messages
            assert "checking history" in user_messages

            # Check corresponding assistant responses
            assert "string message" in assistant_messages  # Passthrough returns same text
            assert "prompt message" in assistant_messages
            assert "multipart message" in assistant_messages
            assert "checking history" in assistant_messages

            # Find a user message and verify the next message is from assistant
            for i in range(len(message_pairs) - 1):
                if message_pairs[i][0] == "user":
                    assert message_pairs[i + 1][0] == "assistant", (
                        "User message should be followed by assistant"
                    )

            # Test directly with conversion from GetPromptResult
            # Simulating a GetPromptResult with a placeholder
            pm = PromptMessage(
                role="user", content=TextContent(type="text", text="simulated prompt result")
            )
            multipart_msgs = PromptMessageMultipart.to_multipart([pm])
            response = await agent.agent1.generate(multipart_msgs, None)
            assert "simulated prompt result" == response.first_text()

            # Test with EmbeddedResource directly in Prompt.user()
            from mcp.types import EmbeddedResource, TextResourceContents
            from pydantic import AnyUrl

            # Create a resource
            text_resource = TextResourceContents(
                uri=AnyUrl("file:///test/example.txt"),
                text="Sample text from resource",
                mimeType="text/plain",
            )
            embedded_resource = EmbeddedResource(type="resource", resource=text_resource)

            # Create a message with text and resource
            message = Prompt.user("Text message with resource", embedded_resource)
            response = await agent.send(message)
            assert response  # Just verify we got a response

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_specify_cwd_for_server(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent1",
        instruction="You are a helpful AI Agent",
        model="playback",
        servers=["cwd_test"],
    )
    async def agent_function():
        async with fast.run() as agent:
            await agent.agent1.apply_prompt("multi")
            assert "how may i" in await agent.agent1.send("cwd_test")

    await agent_function()
