"""Tests for model options parsing and validation."""

from fast_agent.llm.model_database import ModelDatabase


def test_get_known_options_for_reasoning_model():
    """Test getting known options for a model that supports reasoning"""
    known = ModelDatabase.get_known_options("o3")
    assert "reasoning" in known


def test_get_known_options_for_thinking_model():
    """Test getting known options for Anthropic models that support thinking"""
    known = ModelDatabase.get_known_options("claude-sonnet-4-5")
    assert "thinking" in known
    assert "budget" in known


def test_get_known_options_for_unknown_model():
    """Test that unknown models return empty known options"""
    known = ModelDatabase.get_known_options("unknown-model-xyz")
    assert known == []


def test_validate_options_valid():
    """Test validating known options for a model"""
    is_valid, unknown = ModelDatabase.validate_options(
        "o3", {"reasoning": "high"}, strict=True
    )
    assert is_valid is True
    assert unknown == []


def test_validate_options_unknown_strict():
    """Test that unknown options fail validation in strict mode"""
    is_valid, unknown = ModelDatabase.validate_options(
        "o3", {"unknown_option": "value"}, strict=True
    )
    assert is_valid is False
    assert "unknown_option" in unknown


def test_validate_options_unknown_non_strict():
    """Test that unknown options pass validation in non-strict mode"""
    is_valid, unknown = ModelDatabase.validate_options(
        "o3", {"unknown_option": "value"}, strict=False
    )
    assert is_valid is True
    assert "unknown_option" in unknown


def test_validate_options_mixed():
    """Test validating a mix of known and unknown options"""
    is_valid, unknown = ModelDatabase.validate_options(
        "o3", {"reasoning": "high", "unknown": "value"}, strict=True
    )
    assert is_valid is False
    assert "unknown" in unknown
    assert "reasoning" not in unknown


def test_validate_options_empty():
    """Test that empty options always pass validation"""
    is_valid, unknown = ModelDatabase.validate_options("o3", {}, strict=True)
    assert is_valid is True
    assert unknown == []


def test_validate_options_model_without_known_options():
    """Test validation for a model that has no known options defined"""
    # gpt-4.1 doesn't have known_options defined
    is_valid, unknown = ModelDatabase.validate_options(
        "gpt-4.1", {"some_option": "value"}, strict=True
    )
    # In strict mode, any option for a model without known options is unknown
    assert is_valid is False
    assert "some_option" in unknown

    # In non-strict mode, all options are allowed
    is_valid, unknown = ModelDatabase.validate_options(
        "gpt-4.1", {"some_option": "value"}, strict=False
    )
    assert is_valid is True
