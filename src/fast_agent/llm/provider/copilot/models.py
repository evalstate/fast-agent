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


COPILOT_MODELS: Final[Mapping[str, CopilotModelSpec]] = MappingProxyType(
    {
        spec.model_id: spec
        for spec in (
            CopilotModelSpec("claude-fable-5.1", "messages", False, ("sse",)),
            CopilotModelSpec("claude-opus-5", "messages", True, ("sse",)),
            CopilotModelSpec("claude-sonnet-5", "messages", True, ("sse",)),
            CopilotModelSpec("claude-haiku-4.5", "messages", True, ("sse",)),
            CopilotModelSpec("claude-fable-5", "messages", True, ("sse",)),
            CopilotModelSpec(
                "gpt-6-astra",
                "responses",
                True,
                ("sse", "websocket"),
                supports_web_search=True,
            ),
            CopilotModelSpec(
                "gpt-5.6-sol",
                "responses",
                True,
                ("sse", "websocket"),
                supports_web_search=True,
            ),
            CopilotModelSpec("gpt-5.6-terra", "responses", True, ("sse", "websocket")),
            CopilotModelSpec(
                "gpt-5.6-luna",
                "responses",
                True,
                ("sse", "websocket"),
                supports_web_search=True,
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
