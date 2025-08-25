from typing import List
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam

from mcp_agent.agents.agent import Agent
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.providers.augmented_llm_tensorzero_openai import TensorZeroOpenAIAugmentedLLM

# --- Fixtures ---


@pytest.fixture
def mock_agent():
    """Provides a mocked Agent object with a default, empty config."""
    agent = MagicMock(spec=Agent)
    agent._name = "mock_agent"
    agent.instruction = "mock_instruction"
    agent.context = MagicMock()
    agent.context.config = MagicMock()  # Base config mock
    agent.logger = MagicMock()
    return agent


@pytest.fixture
def t0_llm(mock_agent):
    """Provides a standard instance of the class under test."""
    return TensorZeroOpenAIAugmentedLLM(agent=mock_agent, model="test_chat", episode_id="ep-12345")


# --- Tests for _initialize_default_params ---


def test_initialize_default_params_adds_prefix(t0_llm):
    """Tests that the model name is correctly prefixed if it doesn't have it."""
    params = t0_llm._initialize_default_params({"model": "my-model"})
    assert params.model == "tensorzero::function_name::my-model"


def test_initialize_default_params_keeps_existing_prefix(t0_llm):
    """Tests that an existing prefix on the model name is preserved."""
    params = t0_llm._initialize_default_params({"model": "tensorzero::model_name::some-model"})
    assert params.model == "tensorzero::model_name::some-model"


def test_initialize_default_params_sets_defaults(t0_llm):
    """Tests that standard RequestParams are set."""
    t0_llm.instruction = "Test System Prompt"
    params = t0_llm._initialize_default_params({"model": "test_chat"})
    assert params.model == "tensorzero::function_name::test_chat"
    assert params.systemPrompt == "Test System Prompt"
    assert params.parallel_tool_calls is True
    assert params.max_iterations == 10
    assert params.use_history is True


def test_base_url_uses_default_when_config_missing(mock_agent):
    """Tests that the default URL is used when the config is not set."""
    # To test this, ensure the `tensorzero` attribute doesn't exist on the config mock
    del mock_agent.context.config.tensorzero

    llm = TensorZeroOpenAIAugmentedLLM(agent=mock_agent, model="test")
    assert llm._base_url() == "http://localhost:3000/openai/v1"


# --- Tests for _prepare_api_request ---


@patch("mcp_agent.llm.providers.augmented_llm_openai.OpenAIAugmentedLLM._prepare_api_request")
def test_prepare_api_request_with_template_vars(mock_super_prepare, t0_llm):
    """Tests injection of template_vars into a new system message."""
    messages: List[ChatCompletionMessageParam] = []
    # The super call's return value has its own 'messages' list. We ignore it.
    mock_super_prepare.return_value = {"model": "test_chat", "messages": []}
    request_params = RequestParams(template_vars={"var1": "value1"})

    # The method modifies the 'messages' list we pass in.
    t0_llm._prepare_api_request(messages, [], request_params)

    # Assert against the 'messages' list that was passed in and modified.
    assert len(messages) == 1
    system_message = messages[0]
    assert system_message["role"] == "system"
    assert system_message["content"] == [{"var1": "value1"}]


@patch("mcp_agent.llm.providers.augmented_llm_openai.OpenAIAugmentedLLM._prepare_api_request")
def test_prepare_api_request_merges_metadata(mock_super_prepare, t0_llm):
    """Tests merging of tensorzero_arguments from metadata."""
    initial_system_message = ChatCompletionSystemMessageParam(
        role="system", content=[{"var1": "original"}]
    )
    messages: List[ChatCompletionMessageParam] = [initial_system_message]
    mock_super_prepare.return_value = {"model": "test_chat", "messages": messages}
    request_params = RequestParams(metadata={"tensorzero_arguments": {"var2": "metadata_val"}})

    t0_llm._prepare_api_request(messages, [], request_params)

    assert len(messages) == 1
    system_message = messages[0]
    assert system_message["content"] == [{"var1": "original", "var2": "metadata_val"}]


@patch("mcp_agent.llm.providers.augmented_llm_openai.OpenAIAugmentedLLM._prepare_api_request")
def test_prepare_api_request_adds_episode_id(mock_super_prepare, t0_llm):
    """Tests that episode_id is added to extra_body."""
    mock_super_prepare.return_value = {"model": "test_chat", "messages": []}
    request_params = RequestParams()

    arguments = t0_llm._prepare_api_request([], [], request_params)

    assert "extra_body" in arguments
    assert arguments["extra_body"]["tensorzero::episode_id"] == "ep-12345"


@patch("mcp_agent.llm.providers.augmented_llm_openai.OpenAIAugmentedLLM._prepare_api_request")
def test_prepare_api_request_all_features(mock_super_prepare, t0_llm):
    """Tests all features working together."""
    initial_system_message = ChatCompletionSystemMessageParam(
        role="system", content="Original prompt"
    )
    messages: List[ChatCompletionMessageParam] = [initial_system_message]
    mock_super_prepare.return_value = {
        "model": "test_chat",
        "messages": messages,
        "extra_body": {"existing": "val"},
    }
    request_params = RequestParams(
        template_vars={"var1": "value1"},
        metadata={"tensorzero_arguments": {"var2": "metadata_val"}},
    )

    arguments = t0_llm._prepare_api_request(messages, [], request_params)

    # Check system message (modified in-place)
    system_message = messages[0]
    assert system_message["role"] == "system"
    assert system_message["content"] == [{"var1": "value1", "var2": "metadata_val"}]

    # Check extra_body (returned in arguments)
    assert arguments["extra_body"]["tensorzero::episode_id"] == "ep-12345"
    assert arguments["extra_body"]["existing"] == "val"
