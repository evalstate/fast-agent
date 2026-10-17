from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

from mcp_types import TextContent

from fast_agent.constants import FAST_AGENT_RETRY
from fast_agent.llm.provider.error_utils import EmptyCompletionError
from fast_agent.llm.provider.streaming_timeouts import StreamIdleTimeoutError
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

RetryBoundaryKind = Literal[
    "conversation_start",
    "user_message",
    "completed_tool_call",
    "assistant_message",
]


@dataclass(frozen=True, slots=True)
class RetryBoundary:
    kind: RetryBoundaryKind
    message_index: int | None
    tool_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderRetry:
    attempt: int
    max_attempts: int
    wait_seconds: float
    error_type: str
    error_message: str
    reason: Literal["stream_idle", "provider_error", "empty_completion"]
    boundary: RetryBoundary
    stream_events_received: int | None = None


def retry_boundary(messages: object) -> RetryBoundary:
    if not isinstance(messages, list) or not messages:
        return RetryBoundary(kind="conversation_start", message_index=None)

    last = messages[-1]
    if not isinstance(last, PromptMessageExtended):
        return RetryBoundary(kind="conversation_start", message_index=None)
    index = len(messages) - 1
    if last.tool_results:
        return RetryBoundary(
            kind="completed_tool_call",
            message_index=index,
            tool_call_ids=tuple(last.tool_results),
        )
    if last.role == "user":
        return RetryBoundary(kind="user_message", message_index=index)
    return RetryBoundary(kind="assistant_message", message_index=index)


def provider_retry(
    error: Exception,
    *,
    attempt: int,
    max_attempts: int,
    wait_seconds: float,
    boundary: RetryBoundary,
    stream_events_received: int | None = None,
) -> ProviderRetry:
    """Describe one provider retry.

    ``stream_events_received`` lets non-idle mid-stream failures (a truncated
    chunked response, for example) report how far the stream got; idle timeouts
    carry that count on the error itself.
    """
    idle_error = error if isinstance(error, StreamIdleTimeoutError) else None
    return ProviderRetry(
        attempt=attempt,
        max_attempts=max_attempts,
        wait_seconds=wait_seconds,
        error_type=type(error).__name__,
        error_message=str(error),
        reason=(
            "stream_idle"
            if idle_error
            else "empty_completion"
            if isinstance(error, EmptyCompletionError)
            else "provider_error"
        ),
        boundary=boundary,
        stream_events_received=(
            idle_error.events_received if idle_error else stream_events_received
        ),
    )


def append_retry_channel(
    response: PromptMessageExtended,
    retries: list[ProviderRetry],
) -> None:
    if not retries:
        return
    # A provider's own retry inside the last attempt (an empty completion asked once
    # more) was recorded first; the attempts before it come first in the record.
    records = [*(asdict(retry) for retry in retries), *_recorded_retries(response)]
    channels = dict(response.channels or {})
    channels[FAST_AGENT_RETRY] = [
        TextContent(
            type="text",
            text=json.dumps(
                {
                    "schema": "fast-agent.retry/v1",
                    "provider_attempts": len(records) + 1,
                    "retries": records,
                }
            ),
        )
    ]
    response.channels = channels


def _recorded_retries(response: PromptMessageExtended) -> list[object]:
    """Retries already recorded on the response (by the provider, inside an attempt)."""
    recorded: list[object] = []
    for block in (response.channels or {}).get(FAST_AGENT_RETRY, ()):
        if isinstance(block, TextContent):
            recorded.extend(json.loads(block.text).get("retries", []))
    return recorded
