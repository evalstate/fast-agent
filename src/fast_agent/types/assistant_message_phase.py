"""Assistant message phase metadata aligned with the OpenAI Responses SDK."""

from typing import Final, TypeGuard

from typing_extensions import Literal

# Stainless does not currently export a standalone alias for the shared `phase`
# field, so we mirror the SDK's EasyInputMessage / ResponseOutputMessage literal.
type AssistantMessagePhase = Literal["commentary", "final_answer"]

COMMENTARY_PHASE: Final[AssistantMessagePhase] = "commentary"
FINAL_ANSWER_PHASE: Final[AssistantMessagePhase] = "final_answer"
ASSISTANT_MESSAGE_PHASE_VALUES: Final[tuple[AssistantMessagePhase, ...]] = (
    COMMENTARY_PHASE,
    FINAL_ANSWER_PHASE,
)


def is_assistant_message_phase(value: object) -> TypeGuard[AssistantMessagePhase]:
    """Return True when value matches the OpenAI SDK assistant phase literals."""
    return isinstance(value, str) and value in ASSISTANT_MESSAGE_PHASE_VALUES


def coerce_assistant_message_phase(value: object) -> AssistantMessagePhase | None:
    """Return a validated assistant phase value or None for unsupported values."""
    if is_assistant_message_phase(value):
        return value
    return None
