import asyncio
import json

import pytest

from fast_agent.constants import FAST_AGENT_TIMING
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.stream_types import StreamChunk


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
