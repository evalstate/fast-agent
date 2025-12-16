"""Curated model catalog for the setup wizard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CuratedModel:
    """A recommended model for the wizard."""

    id: str
    display_name: str
    description: str


def _get_model_string(alias: str) -> str:
    """Look up the full model string from MODEL_ALIASES."""
    from fast_agent.llm.model_factory import ModelFactory

    return ModelFactory.MODEL_ALIASES.get(alias, alias)


# Curated list of recommended models for HuggingFace inference
CURATED_MODELS: list[CuratedModel] = [
    CuratedModel(
        id="kimi",
        display_name="Kimi K2 Instruct",
        description="Kimi K2-Instruct-0905 is the latest, most capable version of Kimi K2. (default)",
    ),
    CuratedModel(
        id="kimithink",
        display_name="Kimi K2 Thinking",
        description="Advanced reasoning model with extended thinking",
    ),
    CuratedModel(
        id="gpt-oss",
        display_name="OpenAI gpt-oss-120b",
        description="OpenAI's open-weight models designed for powerful reasoning, agentic tasks, and versatile developer use cases.",
    ),
    CuratedModel(
        id="glm",
        display_name="GLM 4.6",
        description="ZAI GLM-4.6: Advanced Agentic, Reasoning and Coding Capabilities",
    ),
    CuratedModel(
        id="minimax",
        display_name="MiniMax M2",
        description="MiniMax-M2, a Mini model built for Max coding & agentic workflows",
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
        options.append(
            {
                "const": model.id,
                "title": f"{model.display_name} - {model.description}",
            }
        )
    # Add custom option
    options.append(
        {
            "const": CUSTOM_MODEL_OPTION.id,
            "title": f"{CUSTOM_MODEL_OPTION.display_name} - {CUSTOM_MODEL_OPTION.description}",
        }
    )

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


def format_model_list_help() -> str:
    """Format the curated model list for display in /set-model help.

    Returns a markdown-formatted string showing available models and explaining
    the Hugging Face model format including provider routing.
    """
    lines: list[str] = []

    lines.extend(
        [
            "",
            "## Usage",
            "",
            "```",
            "/set-model <alias>",
            "/set-model <model-string>",
            "/set-model <org>/<model-name>   # looks up available providers",
            "```",
            "",
            "**Examples:**",
            "- `/set-model kimi` - Use the Kimi K2 model",
            "- `/set-model glm` - Use GLM 4.6",
            "- `/set-model moonshotai/Kimi-K2-Thinking` - Set model (autoroute) and show providers",
            "",
            "## Model String Format",
            "",
            "You can specify a full model string with optional provider routing:",
            "",
            "```",
            "hf.<org>/<model-name>:<provider>",
            "```",
            "",
            "- `hf.` prefix indicates Hugging Face Inference API",
            "- `<org>/<model-name>` is the Hugging Face model identifier",
            "- `:<provider>` routes to a specific inference provider",
            "",
            "**Find out more:** https://huggingface.co/docs/inference-providers/index",
            "",
        ]
    )

    lines.extend(
        [
            "## Aliased Models\n",
            "| Alias | Model String | Description |",
            "|-------|--------------|-------------|",
        ]
    )

    for model in CURATED_MODELS:
        model_string = _get_model_string(model.id)
        lines.append(f"| `{model.id}` | `{model_string}` | {model.description} |")

    return "\n".join(lines)
