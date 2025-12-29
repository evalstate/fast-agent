"""Tests for /set-model validation in ACP mode."""

from __future__ import annotations

import pytest

from fast_agent.llm.hf_inference_lookup import (
    InferenceProvider,
    InferenceProviderLookupResult,
    InferenceProviderStatus,
    ModelValidationResult,
    validate_hf_model,
)


def _make_valid_model_lookup() -> InferenceProviderLookupResult:
    """Create a lookup result for a valid model with providers."""
    return InferenceProviderLookupResult(
        model_id="moonshotai/Kimi-K2-Instruct-0905",
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


def _make_nonexistent_model_lookup(model_id: str) -> InferenceProviderLookupResult:
    """Create a lookup result for a non-existent model."""
    return InferenceProviderLookupResult(
        model_id=model_id,
        exists=False,
        providers=[],
        error=f"Model '{model_id}' not found on HuggingFace",
    )


def _make_no_providers_lookup(model_id: str) -> InferenceProviderLookupResult:
    """Create a lookup result for a model without providers."""
    return InferenceProviderLookupResult(
        model_id=model_id,
        exists=True,
        providers=[],
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_rejects_nonexistent_model() -> None:
    """Test that validation rejects models that don't exist on HuggingFace."""

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        return _make_nonexistent_model_lookup(model_id)

    result = await validate_hf_model(
        "hf.fake-org/nonexistent-model",
        lookup_fn=stub_lookup,
    )

    assert isinstance(result, ModelValidationResult)
    assert result.valid is False
    assert result.error is not None
    assert "not found" in result.error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_rejects_model_without_providers() -> None:
    """Test that validation rejects models without inference providers."""

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        return _make_no_providers_lookup(model_id)

    result = await validate_hf_model(
        "hf.some-org/model-without-providers",
        lookup_fn=stub_lookup,
    )

    assert isinstance(result, ModelValidationResult)
    assert result.valid is False
    assert result.error is not None
    assert "no inference providers" in result.error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_accepts_valid_model_with_providers() -> None:
    """Test that validation accepts models with inference providers."""

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        return _make_valid_model_lookup()

    result = await validate_hf_model(
        "hf.moonshotai/Kimi-K2-Instruct-0905",
        lookup_fn=stub_lookup,
    )

    assert isinstance(result, ModelValidationResult)
    assert result.valid is True
    assert result.error is None
    assert result.display_message != ""
    assert "Available providers" in result.display_message


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_accepts_model_with_provider_suffix() -> None:
    """Test that validation works with model:provider format."""

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        # Model ID should have the provider suffix stripped
        assert ":" not in model_id
        return _make_valid_model_lookup()

    result = await validate_hf_model(
        "hf.moonshotai/Kimi-K2-Instruct-0905:together",
        lookup_fn=stub_lookup,
    )

    assert result.valid is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_skips_non_hf_models() -> None:
    """Test that validation skips models without org/model format."""
    lookup_called = False

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        nonlocal lookup_called
        lookup_called = True
        return _make_valid_model_lookup()

    # Non-HF model (no slash) - should skip validation
    result = await validate_hf_model("gpt-4o", lookup_fn=stub_lookup)

    assert result.valid is True
    assert lookup_called is False  # Lookup should not be called


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_handles_lookup_exception() -> None:
    """Test that validation handles lookup failures gracefully."""

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        raise Exception("Network error")

    result = await validate_hf_model(
        "hf.some-org/some-model",
        lookup_fn=stub_lookup,
    )

    assert result.valid is False
    assert result.error is not None
    assert "Failed to validate" in result.error


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_resolves_aliases_to_hf_models() -> None:
    """Test that aliases like 'kimi' are resolved and show provider info."""
    from fast_agent.llm.model_factory import ModelFactory

    # Find an alias that resolves to an HF model
    hf_alias = None
    resolved_model = None
    for alias, model in ModelFactory.MODEL_ALIASES.items():
        if model.startswith("hf."):
            hf_alias = alias
            resolved_model = model
            break

    if hf_alias is None or resolved_model is None:
        pytest.skip("No HF model aliases found in MODEL_ALIASES")
    assert hf_alias is not None
    assert resolved_model is not None

    # Extract the expected model ID from the resolved model
    expected_model_id = resolved_model[3:]  # Strip "hf."
    if ":" in expected_model_id:
        expected_model_id = expected_model_id.rsplit(":", 1)[0]

    lookup_called_with: list[str] = []

    async def stub_lookup(model_id: str) -> InferenceProviderLookupResult:
        lookup_called_with.append(model_id)
        return InferenceProviderLookupResult(
            model_id=model_id,
            exists=True,
            providers=[
                InferenceProvider(
                    name="groq",
                    status=InferenceProviderStatus.LIVE,
                    provider_id="test",
                    task="conversational",
                    is_model_author=False,
                ),
            ],
        )

    # Call with the alias (e.g., "kimi") and pass aliases dict
    result = await validate_hf_model(
        hf_alias, aliases=ModelFactory.MODEL_ALIASES, lookup_fn=stub_lookup
    )

    # Should have resolved the alias and looked up the HF model
    assert result.valid is True
    assert len(lookup_called_with) == 1
    assert lookup_called_with[0] == expected_model_id
    assert "Available providers" in result.display_message
