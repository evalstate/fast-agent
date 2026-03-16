from __future__ import annotations

from fast_agent.llm.llamacpp_discovery import (
    LlamaCppDiscoveredModel,
    LlamaCppModelListing,
    build_llamacpp_overlay_manifest,
    normalize_llamacpp_url,
)


def test_normalize_llamacpp_url_accepts_root_and_v1_urls() -> None:
    root = normalize_llamacpp_url("http://localhost:8080")
    assert root.server_url == "http://localhost:8080"
    assert root.request_base_url == "http://localhost:8080/v1"
    assert root.models_urls()[0] == "http://localhost:8080/v1/models"

    v1 = normalize_llamacpp_url("http://localhost:8080/v1")
    assert v1.server_url == "http://localhost:8080"
    assert v1.request_base_url == "http://localhost:8080/v1"
    assert v1.models_urls()[0] == "http://localhost:8080/v1/models"


def test_build_llamacpp_overlay_manifest_maps_runtime_defaults_and_modalities() -> None:
    manifest = build_llamacpp_overlay_manifest(
        overlay_name="qwen-local",
        discovered_model=LlamaCppDiscoveredModel(
            listing=LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=262144,
            ),
            props_url="http://localhost:8080/props?model=unsloth%2FQwen3.5-9B-GGUF",
            runtime_context_window=75264,
            max_output_tokens=2048,
            temperature=0.800000011920929,
            top_k=40,
            top_p=0.949999988079071,
            min_p=0.05000000074505806,
            tokenizes=("text/plain", "image/jpeg", "image/png", "image/webp"),
            model_alias="Qwen local",
        ),
        base_url="http://localhost:8080/v1",
        auth="none",
        api_key_env=None,
        secret_ref=None,
        current=True,
    )

    payload = manifest.model_dump(mode="json", exclude_none=True)
    assert payload["provider"] == "openresponses"
    assert payload["connection"]["base_url"] == "http://localhost:8080/v1"
    assert payload["defaults"]["temperature"] == 0.8
    assert payload["defaults"]["top_k"] == 40
    assert payload["defaults"]["top_p"] == 0.95
    assert payload["defaults"]["min_p"] == 0.05
    assert payload["defaults"]["max_tokens"] == 2048
    assert payload["metadata"]["context_window"] == 75264
    assert payload["metadata"]["max_output_tokens"] == 2048
    assert payload["metadata"]["tokenizes"] == [
        "text/plain",
        "image/jpeg",
        "image/png",
        "image/webp",
    ]
    assert payload["picker"]["label"] == "Qwen local"
    assert payload["picker"]["description"] == "Imported from llama.cpp"
