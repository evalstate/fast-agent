import asyncio
import json

import httpx
import pytest

from fast_agent.constants import FAST_AGENT_TIMING
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageSchema,
)
from fast_agent.mcp.prompt import Prompt


class ReasoningStreamingPassthroughLLM(PassthroughLLM):
    async def _apply_prompt_provider_specific(self, *args, **kwargs):
        await asyncio.sleep(0.01)
        self._notify_stream_listeners(StreamChunk(text="thinking", is_reasoning=True))
        await asyncio.sleep(0.01)
        self._notify_stream_listeners(StreamChunk(text="final answer", is_reasoning=False))
        return await super()._apply_prompt_provider_specific(*args, **kwargs)


class ToolStartStreamingPassthroughLLM(PassthroughLLM):
    async def _apply_prompt_provider_specific(self, *args, **kwargs):
        await asyncio.sleep(0.01)
        self._notify_tool_stream_listeners("start", {"tool_name": "lookup"})
        return await super()._apply_prompt_provider_specific(*args, **kwargs)


def _timing_channel_payload(response) -> dict[str, object]:
    channels = response.channels or {}
    timing_channel = channels.get(FAST_AGENT_TIMING)
    assert timing_channel
    return json.loads(timing_channel[0].text)


@pytest.mark.asyncio
async def test_generate_records_ttft_and_time_to_response_for_reasoning_then_text() -> None:
    llm = ReasoningStreamingPassthroughLLM()

    response = await llm.generate([Prompt.user("hello")])

    payload = _timing_channel_payload(response)
    ttft_ms = payload.get("ttft_ms")
    response_ms = payload.get("time_to_response_ms")
    assert isinstance(ttft_ms, float)
    assert isinstance(response_ms, float)
    assert 0 < ttft_ms < response_ms


@pytest.mark.asyncio
async def test_generate_records_tool_start_as_first_response() -> None:
    llm = ToolStartStreamingPassthroughLLM()

    response = await llm.generate([Prompt.user("***CALL_TOOL lookup {}")])

    payload = _timing_channel_payload(response)
    ttft_ms = payload.get("ttft_ms")
    response_ms = payload.get("time_to_response_ms")
    assert isinstance(ttft_ms, float)
    assert isinstance(response_ms, float)
    assert 0 < ttft_ms <= response_ms


class UsageRecordingPassthroughLLM(ReasoningStreamingPassthroughLLM):
    """Records one provider attempt per call; the first ``fail_attempts`` calls fail mid-stream."""

    fail_attempts = 0

    async def _apply_prompt_provider_specific(self, *args, **kwargs):
        self.usage_accumulator.add_turn(
            TurnUsage(
                provider=Provider.FAST_AGENT,
                usage_schema=UsageSchema.OPENAI_CHAT,
                model="passthrough",
                prompt=PromptTokenUsage(total=10),
                completion=CompletionTokenUsage(total=5),
            )
        )
        if len(self.usage_accumulator.turns) <= self.fail_attempts:
            raise httpx.ReadError("Response payload is not completed")
        return await super()._apply_prompt_provider_specific(*args, **kwargs)


@pytest.mark.asyncio
async def test_generate_records_request_timing_on_final_attempt_usage() -> None:
    llm = UsageRecordingPassthroughLLM()
    llm.fail_attempts = 1
    llm.retry_count = 1
    llm.retry_backoff_seconds = 0.0

    response = await llm.generate([Prompt.user("hello")])

    failed, final = llm.usage_accumulator.turns
    assert failed.timing is None
    timing = final.timing
    assert timing is not None
    assert timing.ttft_ms is not None and timing.time_to_response_ms is not None
    assert 0 < timing.ttft_ms < timing.time_to_response_ms <= timing.duration_ms
    assert _timing_channel_payload(response)["duration_ms"] == timing.duration_ms
