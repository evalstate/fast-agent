from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import (
    ListToolsResult,
    TextContent,
    Tool,
)
from tensorzero.types import (
    ChatInferenceResponse,
    FinishReason,
    Usage,
)
from tensorzero.types import (
    Text as T0Text,
)
from tensorzero.util import uuid7

from mcp_agent.agents.agent import Agent
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.providers.augmented_llm_tensorzero import TensorZeroAugmentedLLM


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=Agent)
    agent.name = "mock_agent_name"
    agent.instruction = "mock instruction"

    mock_config = MagicMock(name="agent_config_mock")
    mock_tensorzero_config = MagicMock(name="agent_tensorzero_config_mock")
    mock_tensorzero_config.base_url = "http://mock-t0-url"
    mock_config.tensorzero = mock_tensorzero_config

    mock_context = MagicMock(name="agent_context_mock")
    mock_context.config = mock_config
    agent.context = mock_context

    agent.display = MagicMock(name="agent_display_mock")
    agent.display.show_tool_call = MagicMock()
    agent.display.show_tool_result = MagicMock()
    agent.display.show_assistant_message = AsyncMock()

    agent.logger = MagicMock(name="agent_logger_mock")
    agent.aggregator = AsyncMock(name="agent_aggregator_mock")
    return agent


@pytest.fixture
def t0_llm(mock_agent):
    default_vars = {"TEST_VARIABLE_1": "Test value"}
    llm = TensorZeroAugmentedLLM(
        agent=mock_agent,
        model="tensorzero.test_chat",
        request_params=RequestParams(template_vars=default_vars),
    )

    llm.call_tool = AsyncMock()
    llm.logger = mock_agent.logger
    llm.display = mock_agent.display
    llm.show_tool_call = MagicMock()
    llm.show_tool_result = MagicMock()
    llm.show_assistant_message = AsyncMock()

    return llm


@pytest.mark.asyncio
async def test_adapt_t0_text_response(t0_llm):
    """Test adapting a simple text response from T0."""
    t0_completion = ChatInferenceResponse(
        inference_id=uuid7(),
        episode_id=uuid7(),
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
    # Scenario 1: template_vars provided in RequestParams
    # These vars are now expected to be part of the RequestParams passed to _prepare_t0_system_params
    vars_in_params = {"PARAM_VAR_1": "Value from params"}
    request_params_with_vars = RequestParams(
        model="tensorzero.test_chat", template_vars=vars_in_params.copy()
    )
    system_params = t0_llm._prepare_t0_system_params(request_params_with_vars)
    assert system_params == vars_in_params

    # Scenario 2: template_vars in RequestParams, plus metadata arguments
    vars_in_params_2 = {"PARAM_VAR_2": "Another value"}
    request_params_meta = RequestParams(
        model="tensorzero.test_chat",
        template_vars=vars_in_params_2.copy(),
        metadata={"tensorzero_arguments": {"TEST_VARIABLE_2": "Meta value"}},
    )
    system_params_meta = t0_llm._prepare_t0_system_params(request_params_meta)
    # _prepare_t0_system_params starts with a copy of template_vars from the input RequestParams,
    # then updates it with metadata.
    expected_meta_params = vars_in_params_2.copy()
    expected_meta_params.update({"TEST_VARIABLE_2": "Meta value"})
    assert system_params_meta == expected_meta_params

    # Scenario 3: Empty template_vars in RequestParams (default_factory=dict)
    request_params_empty_vars = RequestParams(
        model="tensorzero.test_chat"
    )  # template_vars will be {}
    system_params_empty = t0_llm._prepare_t0_system_params(request_params_empty_vars)
    assert system_params_empty == {}

    # Scenario 4: Empty template_vars in RequestParams, with metadata
    request_params_empty_with_meta = RequestParams(
        model="tensorzero.test_chat",
        metadata={"tensorzero_arguments": {"META_ONLY": "Meta only value"}},
    )  # template_vars will be {}
    system_params_empty_meta = t0_llm._prepare_t0_system_params(request_params_empty_with_meta)
    assert system_params_empty_meta == {"META_ONLY": "Meta only value"}


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
    assert default_params.max_iterations == 20
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
