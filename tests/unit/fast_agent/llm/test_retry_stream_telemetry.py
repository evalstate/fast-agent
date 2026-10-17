"""Retry telemetry reports how far a failed provider stream got.

Idle timeouts carry that count on the error; other mid-stream failures (a
truncated chunked response, a reset connection) rely on the provider recording it
before raising. This is what distinguishes "died immediately" from "died with the
answer nearly complete", which decides whether a resumable transport is worth it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
from mcp_types import TextContent

from fast_agent.constants import FAST_AGENT_RETRY
from fast_agent.context import Context
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.streaming_timeouts import StreamIdleTimeoutError, StreamTiming
from fast_agent.mcp.prompt import Prompt
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended


def _timing(events_received: int) -> StreamTiming:
    return StreamTiming(
        events_received=events_received,
        first_event_wait_seconds=0.2,
        max_inter_event_wait_seconds=None,
        inter_event_waits_over_threshold=0,
        timed_out_wait_seconds=None,
    )


def _retries(response: PromptMessageExtended) -> list[dict[str, object]]:
    channels = response.channels or {}
    block = channels[FAST_AGENT_RETRY][0]
    assert isinstance(block, TextContent)
    payload = json.loads(block.text)
    return payload["retries"]


def _llm() -> OpenAILLM:
    llm = OpenAILLM(context=Context(), model="zai-org/glm-5.2")
    llm.retry_count = 2
    llm.retry_backoff_seconds = 0.0
    return llm


@pytest.mark.asyncio
async def test_truncated_stream_retry_reports_stream_progress() -> None:
    llm = _llm()
    attempts: list[int] = []

    async def attempt(_messages: object) -> PromptMessageExtended:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            llm._record_stream_failure(_timing(41))
            raise httpx.ReadError("Response payload is not completed")
        return Prompt.assistant("recovered")

    response = await llm._execute_with_retry(attempt, [])

    assert attempts == [0, 1]
    (retry,) = _retries(response)
    assert retry["reason"] == "provider_error"
    assert retry["error_type"] == "ReadError"
    assert retry["stream_events_received"] == 41


@pytest.mark.asyncio
async def test_stream_progress_does_not_leak_between_attempts() -> None:
    llm = _llm()
    attempts: list[int] = []

    async def attempt(_messages: object) -> PromptMessageExtended:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            llm._record_stream_failure(_timing(41))
            raise httpx.ReadError("Response payload is not completed")
        if len(attempts) == 2:
            # Fails before any stream is established, so there is no progress to report.
            raise httpx.ConnectError("connection refused")
        return Prompt.assistant("recovered")

    response = await llm._execute_with_retry(attempt, [])

    first, second = _retries(response)
    assert first["stream_events_received"] == 41
    assert second["stream_events_received"] is None


@pytest.mark.asyncio
async def test_idle_timeout_retry_keeps_its_own_event_count() -> None:
    llm = _llm()
    attempts: list[int] = []

    async def attempt(_messages: object) -> PromptMessageExtended:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            llm._record_stream_failure(_timing(2))
            raise StreamIdleTimeoutError(120.0, events_received=9)
        return Prompt.assistant("recovered")

    response = await llm._execute_with_retry(attempt, [])

    (retry,) = _retries(response)
    assert retry["reason"] == "stream_idle"
    assert retry["stream_events_received"] == 9


@pytest.mark.asyncio
async def test_retry_notice_attempt_counter_matches_telemetry(capsys) -> None:
    """The counter shown to the user must agree with the telemetry it describes.

    ``retry_count`` counts retries, so total attempts is ``retry_count + 1``; the
    notice, the webdebug line and ``max_attempts`` all have to say the same thing.
    """
    llm = _llm()
    attempts: list[int] = []

    async def attempt(_messages: object) -> PromptMessageExtended:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise httpx.ReadError("Response payload is not completed")
        return Prompt.assistant("recovered")

    response = await llm._execute_with_retry(attempt, [])

    (retry,) = _retries(response)
    assert retry["attempt"] == 1
    assert retry["max_attempts"] == 3
    assert "Attempt 1/3" in capsys.readouterr().err


def _status_error(status: int) -> Exception:
    import httpx2
    from openai import APIStatusError

    request = httpx2.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    response = httpx2.Response(status, request=request)
    return APIStatusError(
        f"Error code: {status} - {{'error': 'request entity too large'}}",
        response=response,
        body={"error": "request entity too large"},
    )


@pytest.mark.asyncio
async def test_request_too_large_is_not_resent(capsys) -> None:
    """A 413 refuses the request body itself: resending it fails the same way.

    Regression: an HF-routed run whose attached images grew the conversation past the
    provider's body limit retried the identical request three times (~60s of backoff)
    before ending in the same error. It now ends in that error response at once.
    """
    llm = _llm()
    attempts: list[int] = []

    async def attempt(_messages: object) -> PromptMessageExtended:
        attempts.append(len(attempts))
        raise _status_error(413)

    response = await llm._execute_with_retry(attempt, [])

    assert attempts == [0]
    assert response.stop_reason == LlmStopReason.ERROR
    assert "status=413" in response.all_text()
    assert FAST_AGENT_RETRY not in (response.channels or {})
    assert "Retrying" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_other_status_errors_are_still_retried() -> None:
    llm = _llm()
    attempts: list[int] = []

    async def attempt(_messages: object) -> PromptMessageExtended:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise _status_error(429)
        return Prompt.assistant("recovered")

    response = await llm._execute_with_retry(attempt, [])

    assert attempts == [0, 1]
    assert response.all_text() == "recovered"


@pytest.mark.asyncio
async def test_a_retry_inside_the_last_attempt_follows_the_earlier_attempts() -> None:
    """A provider's own retry (an empty completion asked once more) is recorded on the
    response before the retry loop adds its attempts: both are kept, in order."""
    from fast_agent.llm.retry_telemetry import append_retry_channel

    llm = _llm()
    attempts: list[int] = []

    async def attempt(_messages: object) -> PromptMessageExtended:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise httpx.ReadError("Response payload is not completed")
        response = Prompt.assistant("recovered")
        append_retry_channel(response, [llm._empty_completion_retry()])
        return response

    response = await llm._execute_with_retry(attempt, [])

    assert [r["reason"] for r in _retries(response)] == ["provider_error", "empty_completion"]
    block = (response.channels or {})[FAST_AGENT_RETRY][0]
    assert isinstance(block, TextContent)
    assert json.loads(block.text)["provider_attempts"] == 3
