from typing import List

from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


# Create a minimal testable subclass of AugmentedLLM
class StubLLM(FastAgentLLM):
    """Minimal implementation of AugmentedLLM for testing purposes"""

    def __init__(self, *args, **kwargs):
        super().__init__(provider=Provider.FAST_AGENT, *args, **kwargs)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageExtended"],
        request_params: RequestParams | None = None,
        tools=None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """Implement the abstract method with minimal functionality"""
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
        exclude_fields = {FastAgentLLM.PARAM_MODEL, FastAgentLLM.PARAM_MAX_TOKENS}
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
        llm = OpenAILLM()

        # Basic setup
        base_args = {"model": "gpt-4.1", "messages": [], "max_tokens": 1000}

        # Create params with regular fields, metadata, and response_format
        params = RequestParams(
            model="gpt-4.1",
            temperature=0.7,
            maxTokens=2000,  # This should be excluded and not conflict with max_tokens
            systemPrompt="You are a helpful assistant",  # This should be excluded
            response_format={"type": "json_object"},
            use_history=True,  # This should be excluded
            max_iterations=5,  # This should be excluded
            parallel_tool_calls=True,  # This should be excluded
            metadata={"seed": 42},
        )

        # Prepare arguments with OpenAI-specific exclusions
        result = llm.prepare_provider_arguments(base_args, params, llm.OPENAI_EXCLUDE_FIELDS)

        # Verify results
        assert result["model"] == "gpt-4.1"  # From base_args
        assert result["max_tokens"] == 1000  # From base_args
        assert result["temperature"] == 0.7  # From params
        assert result["response_format"] == {"type": "json_object"}  # From params
        assert result["seed"] == 42  # From metadata
        assert "maxTokens" not in result  # Should be excluded
        assert "systemPrompt" not in result  # Should be excluded
        assert "use_history" not in result  # Should be excluded
        assert "max_iterations" not in result  # Should be excluded
        assert "parallel_tool_calls" not in result  # Should be excluded

    def test_anthropic_provider_arguments(self):
        """Test prepare_provider_arguments with Anthropic provider"""
        # Create an Anthropic LLM instance without initializing provider connections
        llm = AnthropicLLM()

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
            maxTokens=2000,  # This should be excluded
            systemPrompt="You are a helpful assistant",  # This should be excluded
            use_history=True,  # This should be excluded
            max_iterations=5,  # This should be excluded
            parallel_tool_calls=True,  # This should be excluded
            metadata={"top_k": 10},
        )

        # Prepare arguments with Anthropic-specific exclusions
        result = llm.prepare_provider_arguments(base_args, params, llm.ANTHROPIC_EXCLUDE_FIELDS)

        # Verify results
        assert result["model"] == "claude-3-7-sonnet"  # From base_args
        assert result["max_tokens"] == 1000  # From base_args
        assert result["system"] == "You are a helpful assistant"  # From base_args
        assert result["temperature"] == 0.7  # From params
        assert result["top_k"] == 10  # From metadata
        assert "maxTokens" not in result  # Should be excluded
        assert "systemPrompt" not in result  # Should be excluded
        assert "use_history" not in result  # Should be excluded
        assert "max_iterations" not in result  # Should be excluded
        assert "parallel_tool_calls" not in result  # Should be excluded

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
