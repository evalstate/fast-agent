"""
Unit tests for context truncation functionality.
Tests the max_context_length_total parameter and related logic.
"""

from mcp_agent.core.request_params import RequestParams


class TestContextTruncationParameter:
    """Test cases for the max_context_length_total parameter."""

    def test_parameter_creation_with_max_context_length_total(self):
        """Test that RequestParams can be created with max_context_length_total."""
        params = RequestParams(
            max_context_length_total=50000,
            use_history=True,
            max_iterations=10
        )
        
        assert params.max_context_length_total == 50000
        assert params.use_history
        assert params.max_iterations == 10

    def test_parameter_defaults_to_none(self):
        """Test that max_context_length_total defaults to None."""
        params = RequestParams()
        assert params.max_context_length_total is None

    def test_parameter_via_dict_creation(self):
        """Test parameter can be set via dictionary (like RouterAgent pattern)."""
        param_dict = {
            "max_context_length_total": 75000,
            "use_history": True,
            "systemPrompt": "Test prompt"
        }
        params = RequestParams(**param_dict)
        assert params.max_context_length_total == 75000
        assert params.use_history

    def test_parameter_with_none_value(self):
        """Test parameter can be explicitly set to None."""
        params = RequestParams(max_context_length_total=None)
        assert params.max_context_length_total is None

    def test_parameter_with_zero_value(self):
        """Test parameter can be set to zero (edge case)."""
        params = RequestParams(max_context_length_total=0)
        assert params.max_context_length_total == 0

    def test_parameter_with_large_value(self):
        """Test parameter works with large values."""
        large_value = 1_000_000
        params = RequestParams(max_context_length_total=large_value)
        assert params.max_context_length_total == large_value

    def test_parameter_type_annotation(self):
        """Test that parameter accepts int or None as expected."""
        # This should work
        params1 = RequestParams(max_context_length_total=12345)
        params2 = RequestParams(max_context_length_total=None)
        
        assert isinstance(params1.max_context_length_total, int)
        assert params2.max_context_length_total is None

    def test_parameter_compatibility_with_existing_params(self):
        """Test that new parameter works alongside all existing parameters."""
        params = RequestParams(
            max_context_length_total=50000,
            maxTokens=2048,
            model="gpt-4",
            use_history=True,
            max_iterations=20,
            parallel_tool_calls=True,
            template_vars={"key": "value"}
        )
        
        # Verify all parameters are set correctly
        assert params.max_context_length_total == 50000
        assert params.maxTokens == 2048
        assert params.model == "gpt-4"
        assert params.use_history
        assert params.max_iterations == 20
        assert params.parallel_tool_calls
        assert params.template_vars == {"key": "value"}

    def test_parameter_serialization(self):
        """Test that parameter is included in model serialization."""
        params = RequestParams(max_context_length_total=30000)
        model_dict = params.model_dump()
        
        assert "max_context_length_total" in model_dict
        assert model_dict["max_context_length_total"] == 30000

    def test_parameter_model_copy(self):
        """Test that parameter is preserved during model_copy operations."""
        original = RequestParams(max_context_length_total=40000, use_history=False)
        copied = original.model_copy()
        
        assert copied.max_context_length_total == 40000
        assert not copied.use_history
        
        # Test model_copy with updates
        updated = original.model_copy(update={"max_context_length_total": 60000})
        assert updated.max_context_length_total == 60000
        assert not updated.use_history  # Original value preserved