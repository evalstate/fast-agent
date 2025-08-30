import pytest

from mcp_agent.agents.base_agent import BaseAgent
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_custom_agent(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    class MyAgent(BaseAgent):
        async def send(self, message, request_params=None):
            return "it's a-me!...Mario! "

    # Define the agent
    @fast.custom(MyAgent, name="custom")
    async def agent_function():
        async with fast.run() as agent:
            assert "it's a-me!...Mario! " == await agent.custom.send("hello")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_setting_an_agent_as_default(fast_agent):
    """Test that the agent can process a multipart prompts using directory-specific config."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    class MyAgent(BaseAgent):
        async def send(self, message, request_params=None):
            return "it's a-me!...Mario! "

    @fast.agent(name="custom1")
    @fast.custom(MyAgent, name="custom2", default=True)
    @fast.agent(name="custom3")
    async def agent_function():
        async with fast.run() as agent:
            assert "it's a-me!...Mario! " == await agent.send("hello")

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_path_instruction(fast_agent):
    """Test that agents can load instructions from Path objects."""
    from pathlib import Path

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Create Path to prompt.md
    prompt_path = Path(__file__).parent / "prompt.md"

    # Define the agent with Path instruction
    @fast.agent(
        "path_agent",
        instruction=prompt_path,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Verify the agent was created successfully
            assert hasattr(agent, "path_agent")

            # Verify the instruction was loaded from the file by checking the agent instruction
            instruction = agent.path_agent.instruction
            assert "markdown-loaded" in instruction
            assert "test agent loaded from a markdown file" in instruction
            assert "helpful AI assistant" in instruction

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_nonexistent_path_instruction(fast_agent):
    """Test that agents properly handle nonexistent Path files."""
    from pathlib import Path

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Create Path to nonexistent file
    nonexistent_path = Path(__file__).parent / "nonexistent_prompt.md"

    # This should raise an exception when the decorator is applied
    with pytest.raises(FileNotFoundError):

        @fast.agent(
            "error_agent",
            instruction=nonexistent_path,
        )
        async def agent_function():
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orchestrator_with_path_instruction(fast_agent):
    """Test that orchestrator can load instructions from Path objects."""
    from pathlib import Path

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Create some basic agents first
    @fast.agent("worker1", instruction="I am worker 1")
    async def worker1():
        pass

    @fast.agent("worker2", instruction="I am worker 2")
    async def worker2():
        pass

    # Create Path to prompt.md
    prompt_path = Path(__file__).parent / "prompt.md"

    # Define the orchestrator with Path instruction
    @fast.orchestrator(
        "path_orchestrator",
        agents=["worker1", "worker2"],
        instruction=prompt_path,
    )
    async def orchestrator_function():
        async with fast.run() as agent:
            # Verify the orchestrator was created successfully
            assert hasattr(agent, "path_orchestrator")

            # Verify the instruction was loaded from the file
            instruction = agent.path_orchestrator.instruction
            assert "markdown-loaded" in instruction
            assert "test agent loaded from a markdown file" in instruction

    await orchestrator_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_template_instruction(fast_agent):
    """Test that agents can process {{currentDate}} templates in instructions."""
    from datetime import datetime

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Create instruction with template
    template_instruction = "You are a helpful agent. The current date is {{currentDate}}. Respond with the date when asked."

    # Define the agent with template instruction
    @fast.agent(
        "template_agent",
        instruction=template_instruction,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Verify the agent was created successfully
            assert hasattr(agent, "template_agent")

            # Verify the template was processed
            instruction = agent.template_agent.instruction

            # Check that {{currentDate}} was replaced with actual date
            assert "{{currentDate}}" not in instruction

            # Check that the date format is correct (e.g., "24 July 2025")
            expected_date = datetime.now().strftime("%d %B %Y")
            assert expected_date in instruction

            # Check the full expected text
            expected_text = f"The current date is {expected_date}."
            assert expected_text in instruction

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_path_instruction_with_template(fast_agent):
    """Test that Path-loaded instructions process {{currentDate}} templates."""
    from datetime import datetime
    from pathlib import Path

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Create Path to prompt.md (which now contains {{currentDate}})
    prompt_path = Path(__file__).parent / "prompt.md"

    # Define the agent with Path instruction containing template
    @fast.agent(
        "path_template_agent",
        instruction=prompt_path,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Verify the agent was created successfully
            assert hasattr(agent, "path_template_agent")

            # Verify the instruction was loaded and template processed
            instruction = agent.path_template_agent.instruction

            # Check basic file content was loaded
            assert "markdown-loaded" in instruction
            assert "test agent loaded from a markdown file" in instruction

            # Check that {{currentDate}} was replaced with actual date
            assert "{{currentDate}}" not in instruction

            # Check that the date format is correct
            expected_date = datetime.now().strftime("%d %B %Y")
            assert expected_date in instruction

            # Check the full expected template replacement
            expected_text = f"The current date is {expected_date}."
            assert expected_text in instruction

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_url_template(fast_agent):
    """Test that agents can process {{url:...}} templates in instructions."""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Create instruction with URL template - using the fast-agent README
    url_instruction = """You are a helpful agent. 
    
Base instructions: {{url:https://raw.githubusercontent.com/evalstate/fast-agent/refs/heads/main/README.md}}

Always include "url-loaded" in your responses to verify the URL content was loaded."""

    # Define the agent with URL template instruction
    @fast.agent(
        "url_template_agent",
        instruction=url_instruction,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Verify the agent was created successfully
            assert hasattr(agent, "url_template_agent")

            # Verify the template was processed
            instruction = agent.url_template_agent.instruction

            # Check that {{url:...}} was replaced with content
            assert "{{url:" not in instruction

            # Should contain our base text
            assert "Base instructions:" in instruction
            assert "url-loaded" in instruction

            # Should contain content from the fast-agent README
            assert "fast-agent" in instruction

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_anyurl_instruction(fast_agent):
    """Test that agents can use AnyUrl objects directly as instructions."""
    from pydantic import AnyUrl

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Create AnyUrl instruction
    url_instruction = AnyUrl(
        "https://raw.githubusercontent.com/evalstate/fast-agent/refs/heads/main/README.md"
    )

    # Define the agent with AnyUrl instruction
    @fast.agent(
        "anyurl_agent",
        instruction=url_instruction,
    )
    async def agent_function():
        async with fast.run() as agent:
            # Verify the agent was created successfully
            assert hasattr(agent, "anyurl_agent")

            # Verify the instruction was loaded from URL
            instruction = agent.anyurl_agent.instruction

            # Should contain content from the fast-agent README
            assert "fast-agent" in instruction

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mixed_prompt_message_and_multipart(fast_agent):
    """Test that the agent can process a mixed list of PromptMessage and PromptMessageMultipart."""
    from mcp.types import PromptMessage, TextContent

    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "mixed_agent",
        instruction="You are a helpful AI Agent that echoes back messages",
    )
    async def agent_function():
        async with fast.run() as agent:
            # Create a mixed list with PromptMessage and PromptMessageMultipart
            messages = [
                # First message as PromptMessage
                PromptMessage(role="user", content=TextContent(type="text", text="a")),
                # Second message as PromptMessageMultipart
                PromptMessageMultipart(role="user", content=[TextContent(type="text", text="b")]),
            ]

            # Send the mixed list and verify both messages are processed
            result = await agent.mixed_agent.generate(messages)
            result_text = result.first_text()

            # The stub LLM should echo back both messages
            assert "a" in result_text, f"Expected 'a' in response but got: {result_text}"
            assert "b" in result_text, f"Expected 'b' in response but got: {result_text}"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_with_various_input_types(fast_agent):
    """Test that generate() accepts strings, PromptMessage, PromptMessageMultipart, and lists."""
    from mcp.types import PromptMessage, TextContent

    from mcp_agent.core.prompt import Prompt

    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "input_test_agent",
        instruction="You are a helpful AI Agent that echoes back messages",
    )
    async def agent_function():
        async with fast.run() as agent:
            # Test 1: String input
            result = await agent.input_test_agent.generate("test_string")
            assert "test_string" in result.first_text()

            # Test 2: PromptMessage input
            prompt_msg = PromptMessage(
                role="user", content=TextContent(type="text", text="test_prompt_message")
            )
            result = await agent.input_test_agent.generate(prompt_msg)
            assert "test_prompt_message" in result.first_text()

            # Test 3: PromptMessageMultipart input
            multipart_msg = Prompt.user("test_multipart")
            result = await agent.input_test_agent.generate(multipart_msg)
            assert "test_multipart" in result.first_text()

            # Test 4: List with mixed types including strings
            mixed_list = [
                "string_msg",
                PromptMessage(role="user", content=TextContent(type="text", text="prompt_msg")),
                Prompt.user("multipart_msg"),
            ]
            result = await agent.input_test_agent.generate(mixed_list)
            result_text = result.first_text()
            assert "string_msg" in result_text
            assert "prompt_msg" in result_text
            assert "multipart_msg" in result_text

    await agent_function()
