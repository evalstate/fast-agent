from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from fast_agent.types import LlmStopReason
from fast_agent.ui.interactive_diagnostics import write_interactive_trace

if TYPE_CHECKING:
    from acp.schema import StopReason
else:
    StopReason = Literal["end_turn", "refusal", "max_tokens", "cancelled"]


def coerce_registry_version(value: object) -> int:
    return value if isinstance(value, int) else 0


END_TURN: StopReason = "end_turn"
REFUSAL: StopReason = "refusal"
MAX_TOKENS: StopReason = "max_tokens"
CANCELLED: StopReason = "cancelled"


def clear_current_task_cancellation_requests(*, session_id: str) -> int:
    """Clear latent cancellation requests on the active ACP prompt task."""
    task = asyncio.current_task()
    if task is None:
        return 0

    cleared = 0
    while task.cancelling() > 0:
        task.uncancel()
        cleared += 1

    if cleared:
        write_interactive_trace(
            "acp.prompt.task_uncancelled",
            session_id=session_id,
            cleared=cleared,
        )
    return cleared


def map_llm_stop_reason_to_acp(llm_stop_reason: LlmStopReason | None) -> StopReason:
    """
    Map fast-agent LlmStopReason to ACP StopReason.

    Args:
        llm_stop_reason: The stop reason from the LLM response

    Returns:
        The corresponding ACP StopReason value
    """
    if llm_stop_reason is None:
        return END_TURN

    key = (
        llm_stop_reason.value
        if isinstance(llm_stop_reason, LlmStopReason)
        else str(llm_stop_reason)
    )
    mapping: dict[str, StopReason] = {
        LlmStopReason.END_TURN.value: END_TURN,
        LlmStopReason.STOP_SEQUENCE.value: END_TURN,
        LlmStopReason.MAX_TOKENS.value: MAX_TOKENS,
        LlmStopReason.TOOL_USE.value: END_TURN,
        LlmStopReason.PAUSE.value: END_TURN,
        LlmStopReason.ERROR.value: REFUSAL,
        LlmStopReason.TIMEOUT.value: REFUSAL,
        LlmStopReason.SAFETY.value: REFUSAL,
        LlmStopReason.CANCELLED.value: CANCELLED,
    }

    return mapping.get(key, END_TURN)


def format_agent_name_as_title(agent_name: str) -> str:
    """
    Format agent name as title case for display.

    Examples:
        code_expert -> Code Expert
        general_assistant -> General Assistant
    """
    return agent_name.replace("_", " ").title()


def truncate_description(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length, taking the first line only."""
    first_line = text.split("\n")[0]
    if len(first_line) > max_length:
        return first_line[:max_length]
    return first_line
