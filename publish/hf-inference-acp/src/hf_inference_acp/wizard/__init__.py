"""Hugging Face Inference Setup Wizard package."""

from hf_inference_acp.wizard.model_catalog import (
    CURATED_MODELS,
    CUSTOM_MODEL_OPTION,
    CuratedModel,
    build_model_selection_schema,
    get_all_model_options,
    get_model_by_id,
)
from hf_inference_acp.wizard.stages import WizardStage, WizardState
from hf_inference_acp.wizard.wizard_llm import WizardSetupLLM

__all__ = [
    # Main LLM class
    "WizardSetupLLM",
    # Stages
    "WizardStage",
    "WizardState",
    # Model catalog
    "CuratedModel",
    "CURATED_MODELS",
    "CUSTOM_MODEL_OPTION",
    "get_all_model_options",
    "get_model_by_id",
    "build_model_selection_schema",
]
