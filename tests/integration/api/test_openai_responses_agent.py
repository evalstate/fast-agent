import pytest
import os
import asyncio # Though not explicitly used in basic test, good for async context

from mcp_agent.core.fastagent import FastAgent # For potential future use
from mcp_agent.agents.agent import Agent
from mcp_agent.mcp.prompts.prompt_message import PromptMessageMultipart, TextContent # Removed Prompt
from mcp_agent.llm.message_role import MessageRole # For asserting role

from pydantic import BaseModel, Field
from mcp_agent.mcp.tools import ToolContext # Renamed from tools to ToolContext
from mcp_agent.mcp.tools import tool as tool_decorator


# from dotenv import load_dotenv # Uncomment if using python-dotenv locally
# load_dotenv()

TEST_OPENAI_ASSISTANT_ID = os.environ.get("TEST_OPENAI_ASSISTANT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
# Note: The OpenAIResponsesAgent also looks for OPENAI_API_KEY in llm_config.credentials
# Ensuring the environment variable is picked up by the agent's default config or that
# the test explicitly provides it via llm_config might be necessary if not automatically handled.
# For integration tests, it's often assumed the environment provides necessary auth.

@pytest.mark.skipif(not TEST_OPENAI_ASSISTANT_ID, reason="TEST_OPENAI_ASSISTANT_ID environment variable not set")
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY environment variable not set (required for client init)")
@pytest.mark.asyncio
async def test_assistant_basic_response():
    """
    Tests a basic interaction with an OpenAI Assistant using OpenAIResponsesAgent.
    Requires TEST_OPENAI_ASSISTANT_ID and OPENAI_API_KEY environment variables.
    The specified assistant should be a simple one, e.g., instructed to be a helpful assistant.
    """

    # We rely on the ModelFactory to select OpenAIResponsesAgent based on "gpt-4-assistant"
    # and that llm_kwargs are passed through Agent -> AugmentedLLM -> OpenAIResponsesAgent
    agent_instance = Agent(
        name="test_openai_responses_assistant", # Agent name
        instruction="You are a test assistant for integration testing.", # This instruction is for the Agent, not directly for OAI Assistant (which has its own)
        model="gpt-4-assistant", # This model alias should resolve to Provider.OPENAI_RESPONSES
        llm_kwargs={"oai_agent_id": TEST_OPENAI_ASSISTANT_ID}
        # If API key needs explicit passing and isn't picked from env by OpenAI client:
        # llm_config_override = {"credentials": {"api_key": OPENAI_API_KEY, "openai_responses_api_key": OPENAI_API_KEY}}
        # agent_instance = Agent(..., llm_config_override=llm_config_override)
    )

    # Create a prompt message.
    # The generate method of Agent expects a PromptMessage (or subclass like PromptMessageMultipart)
    # or a simple string which it converts.
    user_message = PromptMessageMultipart(
        role=MessageRole.USER,
        content=[TextContent(text="Hello, assistant! This is a test.")]
    )

    # Generate a response
    # The Agent.generate method calls _call_llm, which eventually calls 
    # _execute_provider_call and then _apply_prompt_provider_specific in OpenAIResponsesAgent
    response_message = await agent_instance.generate(user_message)

    assert response_message is not None, "Agent did not return a response message."
    assert isinstance(response_message, PromptMessageMultipart), "Response is not a PromptMessageMultipart."
    assert response_message.role == MessageRole.ASSISTANT, f"Response role is not 'assistant', got '{response_message.role}'."
    
    response_text_content = response_message.first_text_content()
    assert response_text_content is not None, "Response message does not contain text content."
    
    response_text = response_text_content.text.strip()
    assert len(response_text) > 0, "Assistant returned an empty text response."
    
    print(f"\nAssistant ID: {TEST_OPENAI_ASSISTANT_ID}")
    print(f"User prompt: {user_message.content[0].text if user_message.content else ''}")
    print(f"Assistant response: {response_text}")


# Tool Definition for the test
class WeatherParams(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

@tool_decorator(args_schema=WeatherParams)
async def get_weather(params: WeatherParams, context: ToolContext) -> str:
    """Gets the current weather in a given location."""
    # Ensure context is used to avoid lint errors if context methods are not called
    _ = context 
    if "paris" in params.location.lower():
        return "The weather in Paris is sunny."
    return f"Weather in {params.location}: Mostly sunny, 25 C."


@pytest.mark.skipif(not TEST_OPENAI_ASSISTANT_ID, reason="TEST_OPENAI_ASSISTANT_ID environment variable not set")
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY environment variable not set (required for client init)")
@pytest.mark.asyncio
async def test_assistant_with_tools():
    """
    Tests an OpenAI Assistant's ability to use a defined tool.
    Requires the Assistant (TEST_OPENAI_ASSISTANT_ID) to be configured on the OpenAI platform
    with a function tool named 'get_weather' matching WeatherParams schema.
    """
    
    # The `get_weather` tool is defined in this file and decorated with `@tool_decorator`.
    # This should register it with the default_tool_registry.
    # The Agent, by default, uses an MCPAggregator that sources tools from this registry.

    agent_instance = Agent(
        name="test_tool_assistant",
        instruction="You are a helpful assistant. Use tools if necessary.",
        model="gpt-4-assistant", # This model alias should resolve to Provider.OPENAI_RESPONSES
        llm_kwargs={"oai_agent_id": TEST_OPENAI_ASSISTANT_ID}
    )

    user_prompt_text = "What's the weather like in Paris?"
    prompt_message = PromptMessageMultipart(
        role=MessageRole.USER,
        content=[TextContent(text=user_prompt_text)]
    )
    
    print(f"\nTesting with Assistant ID: {TEST_OPENAI_ASSISTANT_ID}")
    print(f"User prompt for tool use: {user_prompt_text}")

    response_message = await agent_instance.generate(prompt_message)

    assert response_message is not None, "Agent did not return a response message."
    assert isinstance(response_message, PromptMessageMultipart), "Response is not a PromptMessageMultipart."
    assert response_message.role == MessageRole.ASSISTANT, f"Response role is not 'assistant', got '{response_message.role}'."

    response_text_content = response_message.first_text_content()
    assert response_text_content is not None, "Response message does not contain text content."
    
    response_text = response_text_content.text.strip()
    assert len(response_text) > 0, "Assistant returned an empty text response."

    print(f"Assistant response (tool test): {response_text}")

    # Verify that the tool's output is incorporated into the response
    expected_tool_output = "The weather in Paris is sunny."
    assert expected_tool_output.lower() in response_text.lower(), \
        f"Expected tool output '{expected_tool_output}' not found in assistant response: '{response_text}'"
