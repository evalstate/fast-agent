import json

import httpx2
import pytest
from anthropic import AsyncAnthropic

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.request_params import RequestParams

_SSE_RESPONSE = b"""\
event: message_start
data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"","citations":null}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

"""


class _MessagesStreamSimulator:
    def __init__(self) -> None:
        self.body: dict[str, object] | None = None

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.body = json.loads(await request.aread())
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_SSE_RESPONSE,
        )


@pytest.mark.asyncio
async def test_legacy_sampling_arguments_work_with_anthropic_stream_api() -> None:
    settings = Settings(anthropic=AnthropicSettings(api_key="test-key"))
    llm = AnthropicLLM(
        context=Context(config=settings),
        model="claude-sonnet-4-5",
        name="test-agent",
    )
    arguments = llm.prepare_provider_arguments(
        {
            "model": "claude-sonnet-4-5",
            "messages": [],
            "max_tokens": 10,
        },
        RequestParams(temperature=0.7),
        llm.ANTHROPIC_EXCLUDE_FIELDS,
    )
    simulator = _MessagesStreamSimulator()

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(simulator)) as http_client:
        async with AsyncAnthropic(api_key="test-key", http_client=http_client) as client:
            async with client.beta.messages.stream(**arguments) as stream:
                text = [chunk async for chunk in stream.text_stream]

    assert text == ["OK"]
    assert simulator.body is not None
    assert simulator.body["temperature"] == 0.7
