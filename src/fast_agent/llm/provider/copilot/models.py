"""Immutable public Copilot model and wire-protocol contracts."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from fast_agent.core.exceptions import ModelConfigError


@dataclass(frozen=True)
class CopilotModelSpec:
    model_id: str
    wire_api: Literal["messages", "responses"]
    supports_named_tool_choice: bool
    transports: tuple[Literal["sse", "websocket"], ...]
    supports_web_search: bool = False
    supports_threshold_compaction: bool = False
    """Anthropic threshold compaction passes the gateway (on-demand compaction does not)."""
    supports_native_compaction: bool = False
    """OpenAI ``compaction_trigger`` checkpoints pass the gateway."""


def _messages_model(
    model_id: str, named_tool_choice: bool, *, threshold_compaction: bool = True
) -> CopilotModelSpec:
    """Messages-route contract, with threshold compaction except for legacy Haiku."""
    return CopilotModelSpec(
        model_id,
        "messages",
        named_tool_choice,
        ("sse",),
        supports_threshold_compaction=threshold_compaction,
    )


def _responses_model(
    model_id: str,
    *,
    web_search: bool = True,
    websocket: bool = True,
    native_compaction: bool = True,
) -> CopilotModelSpec:
    return CopilotModelSpec(
        model_id,
        "responses",
        True,
        ("sse", "websocket") if websocket else ("sse",),
        supports_web_search=web_search,
        supports_native_compaction=native_compaction,
    )


COPILOT_MODELS: Final[Mapping[str, CopilotModelSpec]] = MappingProxyType(
    {
        spec.model_id: spec
        for spec in (
            _messages_model("claude-fable-5.1", False),
            _messages_model("claude-opus-5.5", False),
            _messages_model("claude-sonnet-5.5", False),
            _messages_model("claude-opus-4-8", True),
            _messages_model("claude-opus-5", True),
            _messages_model("claude-sonnet-5", True),
            _messages_model("claude-haiku-4.5", True, threshold_compaction=False),
            _messages_model("claude-haiku-5.5", True),
            _messages_model("claude-fable-5", True),
            *(
                _responses_model(model_id)
                for model_id in (
                    "gpt-6-astra",
                    "gpt-6.1-sol",
                    "gpt-6-sol",
                    "gpt-6-luna",
                    "gpt-5.6-sol",
                )
            ),
            _responses_model("gpt-5.6-terra", web_search=False),
            _responses_model("gpt-5.6-luna"),
            # xAI on the Copilot gateway: SSE only, and no compaction_trigger support.
            *(
                _responses_model(model_id, websocket=False, native_compaction=False)
                for model_id in ("grok-4.7", "grok-4.6")
            ),
        )
    }
)


def get_copilot_model(model_id: str) -> CopilotModelSpec:
    """Resolve an exact wire model ID, rejecting unsupported Copilot models."""
    try:
        return COPILOT_MODELS[model_id]
    except KeyError as exc:
        raise ModelConfigError(
            f"Unknown Copilot model: {model_id}",
            f"Supported Copilot models: {', '.join(COPILOT_MODELS)}",
        ) from exc
