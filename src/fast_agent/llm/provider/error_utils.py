"""Utility functions for LLM provider error handling."""

import json

from mcp_types import TextContent

from fast_agent.constants import (
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_PROVIDER_ERROR,
    OPENAI_COMPACTION_ITEMS,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.text import strip_str_to_none

PROVIDER_ERROR_SCHEMA = "fast-agent.provider-error/v1"


class EmptyCompletionError(RuntimeError):
    """The model ended its turn with no content and no tool call, again after a retry."""

    def __init__(self) -> None:
        super().__init__("Completion returned no assistant content or tool calls after one retry.")


def _status(error: Exception) -> int | None:
    """The HTTP status a provider error carries, when it has one."""
    from fast_agent.llm.provider.openai.responses_websocket import ResponsesWebSocketError

    if isinstance(error, ResponsesWebSocketError):
        return error.status
    status = getattr(error, "status_code", None)  # OpenAI and Anthropic SDK status errors
    return status if isinstance(status, int) else None


def build_stream_failure_response(
    provider: Provider | str,
    error: Exception,
    model_name: str,
) -> PromptMessageExtended:
    """Convert streaming API errors into a graceful assistant reply.

    Args:
        provider: The LLM provider (enum or string label).
        error: The exception that occurred during streaming.
        model_name: The model that was being called.

    Returns:
        A PromptMessageExtended with an assistant message describing the error
        and the raw error details in the FAST_AGENT_ERROR_CHANNEL.
    """
    provider_label = provider.value if isinstance(provider, Provider) else str(provider)

    detail = strip_str_to_none(getattr(error, "message", None)) or str(error).strip()

    parts: list[str] = [f"{provider_label} request failed"]
    if model_name:
        parts.append(f"for model '{model_name}'")
    code = getattr(error, "code", None)
    if code:
        parts.append(f"(code: {code})")
    status = _status(error)
    if status:
        parts.append(f"(status={status})")

    message = " ".join(parts)
    if detail:
        message = f"{message}: {detail}"

    user_summary = " ".join(message.split()) if message else ""
    if user_summary and len(user_summary) > 280:
        user_summary = user_summary[:277].rstrip() + "..."

    if user_summary:
        assistant_text = f"I hit an internal error while calling the model: {user_summary}"
        if not assistant_text.endswith((".", "!", "?")):
            assistant_text += "."
        assistant_text += " See fast-agent-error for additional details."
    else:
        assistant_text = (
            "I hit an internal error while calling the model; see fast-agent-error for details."
        )

    assistant_block = text_content(assistant_text)
    error_block = text_content(message)

    # Codes only (no message text), for trace exports: why this call failed.
    provider_error = {
        "schema": PROVIDER_ERROR_SCHEMA,
        "error_type": type(error).__name__,
        "status": status,
    }
    return PromptMessageExtended(
        role="assistant",
        content=[assistant_block],
        channels={
            FAST_AGENT_ERROR_CHANNEL: [error_block],
            FAST_AGENT_PROVIDER_ERROR: [text_content(json.dumps(provider_error))],
        },
        stop_reason=LlmStopReason.ERROR,
    )


def is_empty_completion(message: PromptMessageExtended) -> bool:
    """An end_turn with no content and no tool call (reasoning alone doesn't count).

    It would end the agent loop with nothing done. Providers retry it once: a stream
    cut off mid-reasoning is often recorded this way. A compaction request legitimately
    returns only its compaction items.
    """
    has_content = any(
        not isinstance(block, TextContent) or bool(block.text.strip()) for block in message.content
    )
    return (
        message.stop_reason == LlmStopReason.END_TURN
        and not has_content
        and not message.tool_calls
        and OPENAI_COMPACTION_ITEMS not in (message.channels or {})
    )
