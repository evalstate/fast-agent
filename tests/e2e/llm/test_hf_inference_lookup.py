"""End-to-end tests for HuggingFace inference provider lookup.

These tests make actual HTTP requests to the HuggingFace API to verify
that the inference provider lookup functionality works correctly.
"""

import pytest

from fast_agent.llm.hf_inference_lookup import (
    InferenceProviderStatus,
    format_inference_lookup_message,
    lookup_inference_providers,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_lookup_model_with_providers():
    """Test looking up a model that has inference providers (Kimi-K2-Thinking)."""
    result = await lookup_inference_providers("moonshotai/Kimi-K2-Thinking")

    assert result.exists is True
    assert result.error is None
    assert result.model_id == "moonshotai/Kimi-K2-Thinking"
    assert result.has_providers is True
    assert len(result.live_providers) > 0

    # Verify at least one known provider exists
    provider_names = [p.name for p in result.live_providers]
    known_providers = {"novita", "nebius", "together", "featherless-ai"}
    assert any(name in known_providers for name in provider_names), (
        f"Expected at least one known provider, got: {provider_names}"
    )

    # Verify provider details
    for provider in result.live_providers:
        assert provider.status == InferenceProviderStatus.LIVE
        assert provider.provider_id  # Should have a provider ID
        assert provider.task  # Should have a task type


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_lookup_model_without_providers():
    """Test looking up a model that exists but has no inference providers."""
    # This model exists but has empty inferenceProviderMapping
    result = await lookup_inference_providers("Nanbeige/Nanbeige4-3B-Thinking-2511")

    assert result.exists is True
    assert result.error is None
    assert result.model_id == "Nanbeige/Nanbeige4-3B-Thinking-2511"
    assert result.has_providers is False
    assert len(result.providers) == 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_lookup_nonexistent_model():
    """Test looking up a model that does not exist."""
    result = await lookup_inference_providers("definitely-not-a-real-org/fake-model-xyz-123")

    assert result.exists is False
    assert result.error is not None
    assert "not found" in result.error.lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_lookup_strips_hf_prefix():
    """Test that hf. prefix is correctly stripped from model ID."""
    result = await lookup_inference_providers("hf.moonshotai/Kimi-K2-Thinking")

    assert result.model_id == "moonshotai/Kimi-K2-Thinking"
    assert result.exists is True
    assert result.has_providers is True


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_lookup_strips_provider_suffix():
    """Test that :provider suffix is correctly stripped from model ID."""
    result = await lookup_inference_providers("moonshotai/Kimi-K2-Thinking:together")

    assert result.model_id == "moonshotai/Kimi-K2-Thinking"
    assert result.exists is True
    assert result.has_providers is True


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_lookup_strips_both_prefix_and_suffix():
    """Test that both hf. prefix and :provider suffix are correctly stripped."""
    result = await lookup_inference_providers("hf.moonshotai/Kimi-K2-Thinking:novita")

    assert result.model_id == "moonshotai/Kimi-K2-Thinking"
    assert result.exists is True
    assert result.has_providers is True


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_format_model_strings():
    """Test that format_model_strings returns correct model:provider strings."""
    result = await lookup_inference_providers("moonshotai/Kimi-K2-Thinking")

    assert result.has_providers
    model_strings = result.format_model_strings()

    assert len(model_strings) == len(result.live_providers)
    for model_str in model_strings:
        assert model_str.startswith("moonshotai/Kimi-K2-Thinking:")
        provider_name = model_str.split(":")[-1]
        assert any(p.name == provider_name for p in result.live_providers)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_format_provider_list():
    """Test that format_provider_list returns comma-separated provider names."""
    result = await lookup_inference_providers("moonshotai/Kimi-K2-Thinking")

    assert result.has_providers
    provider_list = result.format_provider_list()

    # Should be comma-separated
    providers = provider_list.split(", ")
    assert len(providers) == len(result.live_providers)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_format_inference_lookup_message_with_providers():
    """Test formatting a lookup result with providers."""
    result = await lookup_inference_providers("moonshotai/Kimi-K2-Thinking")
    message = format_inference_lookup_message(result)

    assert "moonshotai/Kimi-K2-Thinking" in message
    assert "inference provider" in message.lower()
    assert "/set-model" in message
    assert "hf." in message


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_format_inference_lookup_message_no_providers():
    """Test formatting a lookup result without providers."""
    result = await lookup_inference_providers("Nanbeige/Nanbeige4-3B-Thinking-2511")
    message = format_inference_lookup_message(result)

    assert "Nanbeige/Nanbeige4-3B-Thinking-2511" in message
    assert "no inference providers" in message.lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_format_inference_lookup_message_not_found():
    """Test formatting a lookup result for non-existent model."""
    result = await lookup_inference_providers("fake-org/fake-model-xyz")
    message = format_inference_lookup_message(result)

    assert "error" in message.lower() or "not found" in message.lower()
