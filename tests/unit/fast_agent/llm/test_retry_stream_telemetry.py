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
