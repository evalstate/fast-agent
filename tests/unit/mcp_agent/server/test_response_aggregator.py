import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "src" / "mcp_agent" / "server" / "response_aggregator.py"
)
spec = importlib.util.spec_from_file_location("response_aggregator", MODULE_PATH)
response_aggregator = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(response_aggregator)

ChainResponseAggregator = response_aggregator.ChainResponseAggregator
SSEEventType = response_aggregator.SSEEventType
send_sse_event = response_aggregator.send_sse_event


@pytest.mark.asyncio
async def test_chain_response_aggregator():
    agg = ChainResponseAggregator("chain", 2)
    await agg.add_agent_response("a1", "one")
    assert not await agg.should_send_response()
    await agg.add_agent_response("a2", "two")
    assert await agg.should_send_response()
    result = await agg.get_aggregated_response()
    assert result["chain"] == "chain"
    assert result["responses"] == {"a1": "one", "a2": "two"}


class _DummyStream:
    def __init__(self) -> None:
        self.sent = []

    async def send(self, data):
        self.sent.append(data)


@pytest.mark.asyncio
async def test_send_sse_event():
    stream = _DummyStream()
    await send_sse_event(SSEEventType.AGENT_START, {"foo": "bar"}, stream)
    assert stream.sent == [{"event": "agent_start", "data": {"foo": "bar"}}]
