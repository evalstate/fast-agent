"""Utility functions for LLM provider error handling."""

import json
from dataclasses import asdict, dataclass

from mcp_types import TextContent

from fast_agent.constants import (
    FAST_AGENT_EMPTY_COMPLETION,
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_PROVIDER_ERROR,
    OPENAI_COMPACTION_ITEMS,
    REASONING,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.text import strip_str_to_none

PROVIDER_ERROR_SCHEMA = "fast-agent.provider-error/v1"


@dataclass(frozen=True, slots=True)
class EmptyCompletionDiagnostics:
    """Counts only: never retain discarded reasoning text.

    Responses uses its raw terminal status (or incomplete reason) as finish_reason.
    A missing field is unknown, not zero.
    """

    finish_reason: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    reasoning_chars: int | None = None
    streamed: bool | None = None
    unusable_output: str | None = None


def attach_empty_completion_diagnostics(
    message: PromptMessageExtended, diagnostics: EmptyCompletionDiagnostics
) -> None:
    """Carry attempt-local metadata from normalization to the empty-reply retry."""
    if is_empty_completion(message):
        message.channels = {
            **(message.channels or {}),
            FAST_AGENT_EMPTY_COMPLETION: [text_content(json.dumps(asdict(diagnostics)))],
        }


def empty_completion_diagnostics(message: PromptMessageExtended) -> EmptyCompletionDiagnostics:
    for block in (message.channels or {}).get(FAST_AGENT_EMPTY_COMPLETION, ()):
        if isinstance(block, TextContent):
            return EmptyCompletionDiagnostics(**json.loads(block.text))
    # Providers overriding finalization may not supply raw response metadata.
    reasoning = (message.channels or {}).get(REASONING)
    return EmptyCompletionDiagnostics(
        reasoning_chars=(
            sum(len(block.text) for block in reasoning if isinstance(block, TextContent))
            if reasoning is not None
            else None
        )
    )


class EmptyCompletionError(RuntimeError):
    """The model ended its turn with no content and no tool call, again after a retry."""

    def __init__(self, attempts: tuple[EmptyCompletionDiagnostics, ...] = ()) -> None:
        discarded = next(
            (attempt.unusable_output for attempt in attempts if attempt.unusable_output), None
        )
        detail = "Completion returned no assistant content or tool calls after one retry."
        if discarded:
            detail += f" Discarded provider output: {discarded}."
        super().__init__(detail)
        self.attempts = attempts


def _status(error: Exception) -> int | None:
    """The HTTP status a provider error carries, when it has one."""
    from fast_agent.llm.provider.openai.responses_websocket import ResponsesWebSocketError

    if isinstance(error, ResponsesWebSocketError):
        return error.status
    status = getattr(error, "status_code", None)  # OpenAI and Anthropic SDK status errors
    return status if isinstance(status, int) else None


def _code(error: Exception) -> str | None:
    from fast_agent.llm.provider.openai.responses_websocket import ResponsesWebSocketError

    if isinstance(error, ResponsesWebSocketError):
        return error.error_code
    code = getattr(error, "code", None)  # Provider SDK boundary.
    return str(code) if isinstance(code, (str, int)) else None


def provider_policy_code(error: Exception) -> str | None:
    """Recognize provider refusals, not arbitrary errors mentioning security/content."""
    from fast_agent.llm.provider.openai.responses_websocket import ResponsesWebSocketError

    if isinstance(error, ResponsesWebSocketError):
        code, message = error.error_code, str(error)
    else:
        # Keep optional provider SDKs deferred, as in retry classification.
        if not any(cls.__module__.partition(".")[0] == "openai" for cls in type(error).__mro__):
            return None
        from openai import APIError

        if not isinstance(error, APIError):
            return None
        code, message = error.code, error.message

    if isinstance(code, str) and code.casefold() in {
        "cyber_policy",
        "content_policy_violation",
        "content_filter",
    }:
        return code.casefold()
    # Older Responses events sometimes omit the code. Match the provider's refusal
    # sentence, never a keyword anywhere in a timeout, validation error, or content.
    if code in (None, "unknown") and message.startswith(
        "This content was flagged for possible cybersecurity risk."
    ):
        return "cyber_policy"
    return None


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
    code = _code(error)
    policy_code = provider_policy_code(error)
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

    if policy_code:
        assistant_text = (
            "The provider declined this request under its safety policy. "
            "See fast-agent-error for additional details."
        )

    assistant_block = text_content(assistant_text)
    error_block = text_content(message)

    # Codes and count-only diagnostics (no message text), for trace exports.
    provider_error: dict[str, object] = {
        "schema": PROVIDER_ERROR_SCHEMA,
        "error_type": type(error).__name__,
        "status": status,
    }
    if isinstance(error, EmptyCompletionError) and error.attempts:
        provider_error["empty_completion_attempts"] = [asdict(item) for item in error.attempts]
    if code:
        provider_error["code"] = code
    if policy_code:
        provider_error["policy_code"] = policy_code
    return PromptMessageExtended(
        role="assistant",
        content=[assistant_block],
        channels={
            FAST_AGENT_ERROR_CHANNEL: [error_block],
            FAST_AGENT_PROVIDER_ERROR: [text_content(json.dumps(provider_error))],
        },
        stop_reason=LlmStopReason.SAFETY if policy_code else LlmStopReason.ERROR,
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
