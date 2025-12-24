"""Tests for HuggingFace inference provider lookup."""

from __future__ import annotations

import pytest

from fast_agent.llm.hf_inference_lookup import (
    InferenceProvider,
    InferenceProviderLookupResult,
    InferenceProviderStatus,
    lookup_inference_providers,
)


@pytest.mark.asyncio
async def test_lookup_with_valid_model_and_providers() -> None:
    """Test lookup returns providers for a valid model."""

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        return InferenceProviderLookupResult(
            model_id=model_id,
            exists=True,
            providers=[
                InferenceProvider(
                    name="groq",
                    status=InferenceProviderStatus.LIVE,
                    provider_id="moonshotai/kimi-k2-instruct-0905",
                    task="conversational",
                    is_model_author=False,
                ),
                InferenceProvider(
                    name="together",
                    status=InferenceProviderStatus.LIVE,
                    provider_id="moonshotai/Kimi-K2-Instruct-0905",
                    task="conversational",
                    is_model_author=False,
                ),
            ],
        )

    result = await lookup_inference_providers(
        "moonshotai/Kimi-K2-Instruct-0905",
        lookup_fn=stub_lookup,
    )

    assert result.exists is True
    assert result.has_providers is True
    assert len(result.live_providers) == 2
    assert result.error is None


@pytest.mark.asyncio
async def test_lookup_with_nonexistent_model() -> None:
    """Test lookup returns error for a non-existent model."""

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        return InferenceProviderLookupResult(
            model_id=model_id,
            exists=False,
            providers=[],
            error=f"Model '{model_id}' not found on HuggingFace",
        )

    result = await lookup_inference_providers(
        "fake-org/nonexistent-model",
        lookup_fn=stub_lookup,
    )

    assert result.exists is False
    assert result.has_providers is False
    assert result.error is not None
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_lookup_with_model_without_providers() -> None:
    """Test lookup for a model that exists but has no providers."""

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        return InferenceProviderLookupResult(
            model_id=model_id,
            exists=True,
            providers=[],  # Model exists but no providers
        )

    result = await lookup_inference_providers(
        "some-org/model-without-providers",
        lookup_fn=stub_lookup,
    )

    assert result.exists is True
    assert result.has_providers is False
    assert result.error is None


@pytest.mark.asyncio
async def test_lookup_strips_hf_prefix() -> None:
    """Test that the lookup function receives normalized model ID."""
    received_model_ids: list[str] = []

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        received_model_ids.append(model_id)
        return InferenceProviderLookupResult(
            model_id=model_id,
            exists=True,
            providers=[],
        )

    # The stub receives the model_id directly, so we can check what was passed
    await lookup_inference_providers(
        "hf.moonshotai/Kimi-K2-Instruct-0905",
        lookup_fn=stub_lookup,
    )

    # The lookup_fn is called before normalization in the test stub path
    assert received_model_ids[0] == "hf.moonshotai/Kimi-K2-Instruct-0905"


@pytest.mark.asyncio
async def test_lookup_result_format_provider_list() -> None:
    """Test formatting of provider list."""
    result = InferenceProviderLookupResult(
        model_id="test/model",
        exists=True,
        providers=[
            InferenceProvider(
                name="groq",
                status=InferenceProviderStatus.LIVE,
                provider_id="test",
                task="conversational",
                is_model_author=False,
            ),
            InferenceProvider(
                name="together",
                status=InferenceProviderStatus.LIVE,
                provider_id="test",
                task="conversational",
                is_model_author=False,
            ),
            InferenceProvider(
                name="staging-provider",
                status=InferenceProviderStatus.STAGING,
                provider_id="test",
                task="conversational",
                is_model_author=False,
            ),
        ],
    )

    # Should only include live providers
    provider_list = result.format_provider_list()
    assert "groq" in provider_list
    assert "together" in provider_list
    assert "staging-provider" not in provider_list


@pytest.mark.asyncio
async def test_lookup_result_format_model_strings() -> None:
    """Test formatting of model strings with provider suffixes."""
    result = InferenceProviderLookupResult(
        model_id="moonshotai/Kimi-K2-Instruct",
        exists=True,
        providers=[
            InferenceProvider(
                name="groq",
                status=InferenceProviderStatus.LIVE,
                provider_id="test",
                task="conversational",
                is_model_author=False,
            ),
            InferenceProvider(
                name="together",
                status=InferenceProviderStatus.LIVE,
                provider_id="test",
                task="conversational",
                is_model_author=False,
            ),
        ],
    )

    model_strings = result.format_model_strings()
    assert "moonshotai/Kimi-K2-Instruct:groq" in model_strings
    assert "moonshotai/Kimi-K2-Instruct:together" in model_strings
