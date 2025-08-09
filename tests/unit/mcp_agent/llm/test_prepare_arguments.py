from typing import List

from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


# Create a minimal testable subclass of AugmentedLLM
class StubLLM(AugmentedLLM):
    """Minimal implementation of AugmentedLLM for testing purposes"""

    def __init__(self, *args, **kwargs):
        # The __init__ is now safe to use.
        super().__init__(provider=Provider.FAST_AGENT, *args, **kwargs)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """Implement the abstract method with minimal functionality"""
        # Return the last message for simplicity, or None if the list is empty.
        return multipart_messages[-1] if multipart_messages else None


class TestRequestParamsInLLM:
    """Test suite for RequestParams handling in LLM classes"""

    def test_base_prepare_provider_arguments(self):
        """Test the base prepare_provider_arguments method"""
        # Create a testable LLM instance
        llm = StubLLM()

        # Test with minimal base arguments
        base_args = {"model": "test-model"}
        params = RequestParams(temperature=0.7)

        # Prepare arguments
        result = llm.prepare_provider_arguments(base_args, params)

        # Verify results
        assert result["model"] == "test-model"
        assert result["temperature"] == 0.7

    def test_prepare_arguments_with_exclusions(self):
        """Test prepare_provider_arguments with field exclusions"""
        llm = StubLLM()

        # Test with exclusions
        base_args = {"model": "test-model"}
        params = RequestParams(model="different-model", temperature=0.7, maxTokens=1000)

        # Exclude model and maxTokens fields
        exclude_fields = {AugmentedLLM.PARAM_MODEL, AugmentedLLM.PARAM_MAX_TOKENS}
        result = llm.prepare_provider_arguments(base_args, params, exclude_fields)

        # Verify results - model should remain from base_args, maxTokens should be excluded,
        # but temperature should be included
        assert result["model"] == "test-model"  # From base_args, not overridden
        assert "maxTokens" not in result  # Excluded
        assert result["temperature"] == 0.7  # Included from params

    def test_prepare_arguments_with_metadata(self):
        """Test prepare_provider_arguments with metadata override"""
        llm = StubLLM()

        # Test with metadata
        base_args = {"model": "test-model", "temperature": 0.2}
        params = RequestParams(temperature=0.7, metadata={"temperature": 0.9, "top_p": 0.95})

        result = llm.prepare_provider_arguments(base_args, params)

        # Verify results - metadata should override both base_args and params fields
        assert result["model"] == "test-model"  # From base_args
        assert result["temperature"] == 0.9  # From metadata, overriding both base_args and params
        assert result["top_p"] == 0.95  # From metadata

    def test_response_format_handling(self):
        """Test handling of response_format parameter"""
        llm = StubLLM()

        json_format = {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {"message": {"type": "string"}}},
        }

        # Test with response_format in params
        base_args = {"model": "test-model"}
        params = RequestParams(response_format=json_format)

        result = llm.prepare_provider_arguments(base_args, params)

        # Verify response_format is included
        assert result["model"] == "test-model"
        assert result["response_format"] == json_format

    def test_openai_provider_arguments(self):
        """Test prepare_provider_arguments with OpenAI provider"""
        # Create an OpenAI LLM instance without initializing provider connections
        llm = OpenAIAugmentedLLM()

        # Basic setup
        base_args = {"model": "gpt-4.1", "messages": [], "max_tokens": 1000}

        # Create params with regular fields, metadata, and response_format
        params = RequestParams(
            model="gpt-4.1",
            temperature=0.7,
            maxTokens=2000,
            systemPrompt="You are a helpful assistant",
            response_format={"type": "json_object"},
            use_history=True,
            max_iterations=5,
            parallel_tool_calls=True,
            metadata={"seed": 42},
        )

        # Prepare arguments with OpenAI-specific exclusions
        result = llm.prepare_provider_arguments(base_args, params, llm.OPENAI_EXCLUDE_FIELDS)

        # Verify results
        assert result["model"] == "gpt-4.1"
        assert result["max_tokens"] == 1000
        assert result["temperature"] == 0.7
        assert result["response_format"] == {"type": "json_object"}
        assert result["seed"] == 42
        assert "maxTokens" not in result
        assert "systemPrompt" not in result
        assert "use_history" not in result
        assert "max_iterations" not in result
        assert "parallel_tool_calls" not in result

    def test_anthropic_provider_arguments(self, mocker):
        """Test prepare_provider_arguments with Anthropic provider"""
        mocker.patch(
            'mcp_agent.llm.providers.augmented_llm_anthropic.AnthropicAugmentedLLM._initialize_client',
            return_value=mocker.MagicMock()
        )

        # Create an Anthropic LLM instance
        llm = AnthropicAugmentedLLM()

        # Basic setup
        base_args = {
            "model": "claude-3-7-sonnet",
            "messages": [],
            "max_tokens": 1000,
            "system": "You are a helpful assistant",
        }

        # Create params with various fields
        params = RequestParams(
            model="claude-3-7-sonnet",
            temperature=0.7,
            maxTokens=2000,
            systemPrompt="You are a helpful assistant",
            use_history=True,
            max_iterations=5,
            parallel_tool_calls=True,
            metadata={"top_k": 10},
        )

        # Prepare arguments with Anthropic-specific exclusions
        result = llm.prepare_provider_arguments(base_args, params, llm.ANTHROPIC_EXCLUDE_FIELDS)

        # Verify results
        assert result["model"] == "claude-3-7-sonnet"
        assert result["max_tokens"] == 1000
        assert result["system"] == "You are a helpful assistant"
        assert result["temperature"] == 0.7
        assert result["top_k"] == 10
        assert "maxTokens" not in result
        assert "systemPrompt" not in result
        assert "use_history" not in result
        assert "max_iterations" not in result
        assert "parallel_tool_calls" not in result

    def test_params_dont_overwrite_base_args(self):
        """Test that params don't overwrite base_args with the same key"""
        llm = StubLLM()

        # Set up conflicting keys
        base_args = {"model": "base-model", "temperature": 0.5}
        params = RequestParams(model="param-model", temperature=0.7)

        # Exclude nothing
        result = llm.prepare_provider_arguments(base_args, params, set())

        # base_args should take precedence
        assert result["model"] == "base-model"
        assert result["temperature"] == 0.5

    def test_none_values_not_included(self):
        """Test that None values from params are not included"""
        llm = StubLLM()

        base_args = {"model": "test-model"}
        params = RequestParams(temperature=None, top_p=0.9)

        result = llm.prepare_provider_arguments(base_args, params)

        # None values should be excluded
        assert "temperature" not in result
        assert result["top_p"] == 0.9