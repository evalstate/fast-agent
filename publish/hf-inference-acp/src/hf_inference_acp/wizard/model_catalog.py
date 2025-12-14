"""Curated model catalog for the setup wizard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CuratedModel:
    """A recommended model for the wizard."""

    id: str
    display_name: str
    description: str


# Curated list of recommended models for HuggingFace inference
CURATED_MODELS: list[CuratedModel] = [
    CuratedModel(
        id="hf.moonshotai/Kimi-K2-Instruct-0905",
        display_name="Kimi K2 Instruct",
        description="Fast, capable instruct model - good general purpose choice",
    ),
    CuratedModel(
        id="hf.deepseek-ai/DeepSeek-R1",
        display_name="DeepSeek R1",
        description="Advanced reasoning model with strong capabilities",
    ),
    CuratedModel(
        id="hf.Qwen/Qwen3-235B-A22B",
        display_name="Qwen3 235B",
        description="Large parameter model, high capability",
    ),
    CuratedModel(
        id="hf.meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        display_name="Llama 4 Maverick 17B",
        description="Meta's latest Llama model with strong performance",
    ),
]

# Special option for custom model entry
CUSTOM_MODEL_OPTION = CuratedModel(
    id="__custom__",
    display_name="Custom model...",
    description="Enter a model ID manually",
)


def get_all_model_options() -> list[CuratedModel]:
    """Get all model options including custom."""
    return CURATED_MODELS + [CUSTOM_MODEL_OPTION]


def build_model_selection_schema() -> dict:
    """Build JSON schema for model selection form."""
    options = []
    for model in CURATED_MODELS:
        options.append({
            "const": model.id,
            "title": f"{model.display_name} - {model.description}",
        })
    # Add custom option
    options.append({
        "const": CUSTOM_MODEL_OPTION.id,
        "title": f"{CUSTOM_MODEL_OPTION.display_name} - {CUSTOM_MODEL_OPTION.description}",
    })

    return {
        "type": "object",
        "title": "Select Default Model",
        "properties": {
            "model": {
                "type": "string",
                "title": "Choose your default inference model",
                "oneOf": options,
            }
        },
        "required": ["model"],
    }


def get_model_by_id(model_id: str) -> CuratedModel | None:
    """Find a curated model by its ID."""
    for model in CURATED_MODELS:
        if model.id == model_id:
            return model
    return None
