import pytest
import uuid
import json
from unittest.mock import MagicMock, AsyncMock, patch

from mcp.types import (
    TextContent,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    Tool,
)
from tensorzero.types import (
    ChatInferenceResponse,
    Text as T0Text,
    ToolCall as T0ToolCall,
    Usage,
    FinishReason,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.llm.providers.augmented_llm_tensorzero import TensorZeroAugmentedLLM
from mcp_agent.core.request_params import RequestParams
from mcp_agent.core.exceptions import ModelConfigError
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=Agent)
    agent.name = "mock_agent_name"
    agent.instruction = "mock instruction"

    # Mock the context and config structure directly ON THE AGENT MOCK
    mock_config = MagicMock(name="agent_config_mock")
    mock_tensorzero_config = MagicMock(name="agent_tensorzero_config_mock")
    mock_tensorzero_config.base_url = "http://mock-t0-url"
    mock_config.tensorzero = mock_tensorzero_config

    mock_context = MagicMock(name="agent_context_mock")
    mock_context.config = mock_config
    agent.context = mock_context

    # Add mocked display object with async methods
    agent.display = MagicMock(name="agent_display_mock")
    agent.display.show_tool_call = MagicMock()  # Use regular mock if display methods aren't async
    agent.display.show_oai_tool_result = MagicMock()
    agent.display.show_assistant_message = AsyncMock()

    # Other agent mocks
    agent.logger = MagicMock(name="agent_logger_mock")
    agent.aggregator = AsyncMock(name="agent_aggregator_mock")
    return agent


@pytest.fixture
def t0_llm(mock_agent):
    llm = TensorZeroAugmentedLLM(
        agent=mock_agent,  # This agent now has the correct context structure
        model="tensorzero.test_chat",
        t0_system_template_vars={"TEST_VARIABLE_1": "Test value"},
    )

    # No longer need to manually set llm.context as it should be handled by __init__

    # Assign mocks for methods/attributes NOT copied from agent or needed explicitly
    llm.call_tool = AsyncMock()
    llm.logger = mock_agent.logger  # Continue using the agent's logger mock
    llm.display = mock_agent.display
    llm.show_tool_call = MagicMock()  # Mock the methods on llm directly
    llm.show_oai_tool_result = MagicMock()
    llm.show_assistant_message = AsyncMock()

    return llm


@pytest.mark.asyncio
async def test_adapt_t0_text_response(t0_llm):
    """Test adapting a simple text response from T0."""
    t0_completion = ChatInferenceResponse(
        inference_id=uuid.uuid4(),
        episode_id=uuid.uuid4(),
        variant_name="test_variant",
        content=[T0Text(type="text", text="Hello there!")],
        usage=Usage(input_tokens=10, output_tokens=5),
        finish_reason=FinishReason.STOP,
    )

    content_parts, executed_results, raw_tool_calls = await t0_llm._adapt_t0_native_completion(
        t0_completion
    )

    assert len(content_parts) == 1
    assert isinstance(content_parts[0], TextContent)
    assert content_parts[0].text == "Hello there!"
    assert executed_results == []
    assert raw_tool_calls == []


def test_prepare_t0_system_params(t0_llm):
    """Test preparation of the system parameters dictionary."""
    # Base params from fixture
    request_params = RequestParams(model="tensorzero.test_chat")
    system_params = t0_llm._prepare_t0_system_params(request_params)
    assert system_params == {"TEST_VARIABLE_1": "Test value"}

    # With metadata arguments
    request_params_meta = RequestParams(
        model="tensorzero.test_chat",
        metadata={"tensorzero_arguments": {"TEST_VARIABLE_2": "Meta value"}},
    )
    system_params_meta = t0_llm._prepare_t0_system_params(request_params_meta)
    assert system_params_meta == {
        "TEST_VARIABLE_1": "Test value",
        "TEST_VARIABLE_2": "Meta value",
    }


@pytest.mark.asyncio
async def test_prepare_t0_tools(t0_llm):
    """Test fetching and formatting tools."""
    tool_schema = {
        "type": "object",
        "properties": {"input_text": {"type": "string"}},
        "required": ["input_text"],
    }
    # Create a proper Tool instance
    mcp_tool = Tool(
        name="tester-example_tool",
        description="Reverses text.",
        inputSchema=tool_schema,
    )

    t0_llm.aggregator.list_tools.return_value = ListToolsResult(tools=[mcp_tool])

    formatted_tools = await t0_llm._prepare_t0_tools()

    assert formatted_tools == [
        {
            "name": "tester-example_tool",
            "description": "Reverses text.",
            "parameters": tool_schema,
        }
    ]


@pytest.mark.asyncio
async def test_prepare_t0_tools_empty(t0_llm):
    """Test when no tools are available."""
    t0_llm.aggregator.list_tools.return_value = ListToolsResult(tools=[])
    formatted_tools = await t0_llm._prepare_t0_tools()
    assert formatted_tools is None


def test_initialize_default_params(t0_llm):
    """Test the creation of default request parameters."""
    t0_llm.instruction = "Test System Prompt"
    default_params = t0_llm._initialize_default_params({})
    assert default_params.model == "tensorzero.test_chat"
    assert default_params.systemPrompt == "Test System Prompt"
    assert default_params.maxTokens == 4096
    assert default_params.use_history is True
    assert default_params.max_iterations == 10
    assert default_params.parallel_tool_calls is True


def test_block_to_dict():
    """Test converting various block types to dictionaries."""

    # Pydantic-like model
    class MockModel:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def model_dump(self, mode=None):
            return {"a": self.a, "b": self.b}

    pydantic_block = MockModel(1, "x")
    assert TensorZeroAugmentedLLM.block_to_dict(pydantic_block) == {"a": 1, "b": "x"}

    # Object with __dict__
    class SimpleObj:
        def __init__(self, name):
            self.name = name

    dict_obj = SimpleObj("test")
    assert TensorZeroAugmentedLLM.block_to_dict(dict_obj) == {"name": "test"}

    # Primitives
    assert TensorZeroAugmentedLLM.block_to_dict("hello") == {"type": "raw", "content": "hello"}
    assert TensorZeroAugmentedLLM.block_to_dict(123) == {"type": "raw", "content": 123}
    assert TensorZeroAugmentedLLM.block_to_dict(None) == {"type": "raw", "content": None}
    assert TensorZeroAugmentedLLM.block_to_dict([1, 2]) == {"type": "raw", "content": [1, 2]}

    # T0 Text type
    t0_text = T0Text(type="text", text="fallback")
    block_dict = TensorZeroAugmentedLLM.block_to_dict(t0_text)
    assert block_dict.get("type") == "text"
    assert block_dict.get("text") == "fallback"

    # Fallback (Unknown object)
    class UnknownObj:
        pass

    unknown = UnknownObj()
    # Check type and content separately
    result_dict = TensorZeroAugmentedLLM.block_to_dict(unknown)
    assert result_dict.get("type") == "unknown"
    assert result_dict.get("content") == str(unknown)
    # Check the full dict if the parts are correct
    assert result_dict == {
        "type": "unknown",
        "content": str(unknown),
    }
