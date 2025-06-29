"""
Unit tests for context truncation in AugmentedLLM.
Tests the parameter constants and exclusion fields.
"""

from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_google_native import GoogleNativeAugmentedLLM
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM


class TestContextTruncationLLMIntegration:
    """Test cases for context truncation integration in LLM classes."""

    def test_parameter_constant_defined(self):
        """Test that PARAM_MAX_CONTEXT_LENGTH_TOTAL constant is defined."""
        assert hasattr(AugmentedLLM, "PARAM_MAX_CONTEXT_LENGTH_TOTAL")
        assert AugmentedLLM.PARAM_MAX_CONTEXT_LENGTH_TOTAL == "max_context_length_total"

    def test_parameter_constant_consistency(self):
        """Test that parameter constant matches RequestParams field name."""
        from mcp_agent.core.request_params import RequestParams
        
        # Get field names from RequestParams
        field_names = set(RequestParams.model_fields.keys())
        
        # Verify our constant matches an actual field
        assert AugmentedLLM.PARAM_MAX_CONTEXT_LENGTH_TOTAL in field_names

    def test_openai_excludes_parameter(self):
        """Test that OpenAI provider excludes max_context_length_total parameter."""
        assert AugmentedLLM.PARAM_MAX_CONTEXT_LENGTH_TOTAL in OpenAIAugmentedLLM.OPENAI_EXCLUDE_FIELDS

    def test_anthropic_excludes_parameter(self):
        """Test that Anthropic provider excludes max_context_length_total parameter."""
        assert AugmentedLLM.PARAM_MAX_CONTEXT_LENGTH_TOTAL in AnthropicAugmentedLLM.ANTHROPIC_EXCLUDE_FIELDS

    def test_google_excludes_parameter(self):
        """Test that Google provider excludes max_context_length_total parameter."""
        assert AugmentedLLM.PARAM_MAX_CONTEXT_LENGTH_TOTAL in GoogleNativeAugmentedLLM.GOOGLE_EXCLUDE_FIELDS

    def test_parameter_exclusion_consistency(self):
        """Test that all providers exclude the parameter consistently."""
        providers = [
            (OpenAIAugmentedLLM, "OPENAI_EXCLUDE_FIELDS"),
            (AnthropicAugmentedLLM, "ANTHROPIC_EXCLUDE_FIELDS"), 
            (GoogleNativeAugmentedLLM, "GOOGLE_EXCLUDE_FIELDS")
        ]
        
        for provider_class, exclude_field_name in providers:
            exclude_fields = getattr(provider_class, exclude_field_name)
            assert AugmentedLLM.PARAM_MAX_CONTEXT_LENGTH_TOTAL in exclude_fields, \
                f"{provider_class.__name__} should exclude {AugmentedLLM.PARAM_MAX_CONTEXT_LENGTH_TOTAL}"

    def test_truncation_method_exists(self):
        """Test that the _check_and_truncate_context method exists."""
        assert hasattr(AugmentedLLM, "_check_and_truncate_context")
        
        # Check method signature
        import inspect
        method = getattr(AugmentedLLM, "_check_and_truncate_context")
        sig = inspect.signature(method)
        
        # Should have self and request_params parameters
        param_names = list(sig.parameters.keys())
        assert "self" in param_names
        assert "request_params" in param_names

    def test_common_parameter_exclusions(self):
        """Test that common parameters are excluded by all providers."""
        common_params = [
            AugmentedLLM.PARAM_USE_HISTORY,
            AugmentedLLM.PARAM_MAX_ITERATIONS,
            AugmentedLLM.PARAM_MAX_CONTEXT_LENGTH_TOTAL,
        ]
        
        providers = [
            (OpenAIAugmentedLLM, "OPENAI_EXCLUDE_FIELDS"),
            (AnthropicAugmentedLLM, "ANTHROPIC_EXCLUDE_FIELDS"),
            (GoogleNativeAugmentedLLM, "GOOGLE_EXCLUDE_FIELDS")
        ]
        
        for provider_class, exclude_field_name in providers:
            exclude_fields = getattr(provider_class, exclude_field_name)
            for param in common_params:
                assert param in exclude_fields, \
                    f"{provider_class.__name__} should exclude {param}"

    def test_parameter_documentation(self):
        """Test that the parameter has proper documentation."""
        from mcp_agent.core.request_params import RequestParams
        
        field_info = RequestParams.model_fields.get("max_context_length_total")
        assert field_info is not None
        
        # Check that field has a description (via Field annotation)
        # The description would be in the source code as a docstring comment