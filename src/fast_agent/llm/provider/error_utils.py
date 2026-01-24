"""Utility functions for LLM provider error handling."""

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason


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

    detail = getattr(error, "message", None) or str(error)
    detail = detail.strip() if isinstance(detail, str) else ""

    parts: list[str] = [f"{provider_label} request failed"]
    if model_name:
        parts.append(f"for model '{model_name}'")
    code = getattr(error, "code", None)
    if code:
        parts.append(f"(code: {code})")
    status = getattr(error, "status_code", None)
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

    return PromptMessageExtended(
        role="assistant",
        content=[assistant_block],
        channels={FAST_AGENT_ERROR_CHANNEL: [error_block]},
        stop_reason=LlmStopReason.ERROR,
    )
