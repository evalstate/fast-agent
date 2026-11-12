"""xAI provider-native compaction through ``POST /responses/compact`` (no network).

A local HTTP simulator serves the sanitized live capture from
``scripts/openai_native_compaction_smoke.py --route xai --fixture-out``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer
from mcp_types import TextContent

from fast_agent.config import CompactionSettings, OpenAISettings, Settings, XAISettings
from fast_agent.constants import OPENAI_COMPACTION_ITEMS
from fast_agent.context import Context
from fast_agent.core.exceptions import ProviderCheckpointError
from fast_agent.history.compaction import (
    CompactionError,
    build_summary_message,
    compact_conversation,
    should_auto_compact,
)
from fast_agent.llm.compaction_support import CompactAt, CompactionStrategy, CompactionSupport
from fast_agent.llm.provider.openai.compaction_replay import capture_compaction_replay
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.xai_responses import XAIResponsesLLM
from fast_agent.llm.usage_tracking import UsageAccumulator
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

FIXTURE = (
    Path(__file__).resolve().parents[5]
    / "tests/fixtures/llm_traces/sanitized/xai/grok-4.7/responses_compact"
)


@dataclass
class _CompactSimulator:
    """``/v1/responses/compact`` replaying a recorded response; records each request."""

    response: dict[str, Any] = field(
        default_factory=lambda: json.loads((FIXTURE / "response.json").read_text())
    )
    requests: list[dict[str, Any]] = field(default_factory=list)

    async def compact(self, request: web.Request) -> web.Response:
        self.requests.append(await request.json())
        return web.json_response(self.response)


@pytest_asyncio.fixture
async def simulator() -> AsyncIterator[tuple[_CompactSimulator, str]]:
    sim = _CompactSimulator()
    app = web.Application()
    app.router.add_post("/v1/responses/compact", sim.compact)
    async with TestServer(app) as server:
        yield sim, str(server.make_url("/v1"))


def _llm(
    base_url: str = "http://127.0.0.1:9/v1",
    *,
    native: bool = True,
    compaction: str | None = None,
    compact_at: CompactAt | None = None,
    model: str = "grok-4.7",
) -> XAIResponsesLLM:
    settings = Settings(
        xai=XAISettings(api_key="test", base_url=base_url, native_compaction=native),
        session_history=False,
    )
    return XAIResponsesLLM(
        context=Context(config=settings), model=model, compaction=compaction, compact_at=compact_at
    )


def _user(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(role="user", content=[TextContent(type="text", text=text)])


def _assistant(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(role="assistant", content=[TextContent(type="text", text=text)])


def _checkpoint(origin: str | None) -> PromptMessageExtended:
    item = {"type": "compaction", "id": "cmp_1", "encrypted_content": "x"}
    envelope = (
        capture_compaction_replay(item, origin=origin)
        if origin is not None
        else {
            # Envelopes written before the producing route was recorded (OpenAI only).
            "schema": "fast-agent.openai-responses.compaction-replay",
            "version": 1,
            "item": item,
        }
    )
    return build_summary_message(
        "checkpoint",
        strategy=CompactionStrategy.OPENAI_COMPACTION,
        provider_channels={
            OPENAI_COMPACTION_ITEMS: [TextContent(type="text", text=json.dumps(envelope))]
        },
        prompt_text=None,
        instructions=None,
        messages_compacted=2,
        tokens_before=4_000,
        context_window=500_000,
        model="grok-4.7",
    )


@pytest.mark.parametrize(
    ("native", "mode", "expected"),
    [
        (True, None, CompactionSupport.ON_DEMAND),
        (False, None, CompactionSupport.CLIENT),
        # The model string wins over the route setting, in both directions.
        (True, "client", CompactionSupport.CLIENT),
        (False, "native", CompactionSupport.ON_DEMAND),
    ],
)
def test_xai_native_compaction_setting(
    native: bool, mode: str | None, expected: CompactionSupport
) -> None:
    assert _llm(native=native, compaction=mode).compaction_support == expected


def test_native_compaction_defaults_on_without_an_xai_section() -> None:
    llm = XAIResponsesLLM(context=Context(config=Settings()), model="grok-4.7", api_key="test")

    assert llm.compaction_support == CompactionSupport.ON_DEMAND


class _CompactionAgent:
    name = "xai-compaction"

    def __init__(self, llm: XAIResponsesLLM) -> None:
        self.llm = llm
        self.context = llm.context
        self.usage_accumulator = llm.usage_accumulator
        self.message_history = [
            _user("template instructions"),
            _user("old question"),
            _assistant("old answer"),
            _user("retained question"),
            _assistant("retained answer"),
        ]
        self.message_history[0].is_template = True

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None:
        self.message_history = list(messages or [])


@pytest.mark.asyncio
async def test_harness_compacts_through_the_compact_endpoint_and_replays_the_item(
    simulator: tuple[_CompactSimulator, str],
) -> None:
    sim, base_url = simulator
    llm = _llm(base_url)
    llm.instruction = "You are terse."
    agent = _CompactionAgent(llm)

    result = await compact_conversation(agent, settings=CompactionSettings(keep_turns=1))

    # One compact call: model and the compact region only. No compaction_trigger (xAI
    # rejects it) and no instructions (later requests carry the system prompt).
    [request] = sim.requests
    assert set(request) == {"model", "input"}
    assert request["model"] == "grok-4.7"
    assert [item["content"][0]["text"] for item in request["input"]] == [
        "template instructions",
        "old question",
        "old answer",
    ]
    assert result.is_native
    assert result.strategy == CompactionStrategy.OPENAI_COMPACTION

    # Templates before, retained turns after: the layout verified live on xAI.
    items = llm._convert_to_provider_format([*agent.message_history, _user("new question")])
    assert [item["type"] for item in items] == ["message", "compaction", *["message"] * 3]
    assert items[1] == {
        "type": "compaction",
        "id": sim.response["id"],
        "encrypted_content": "REDACTED-ENCRYPTED-COMPACTION-CHECKPOINT",
    }
    assert "xAI compaction checkpoint" in result.summary_text


@pytest.mark.asyncio
async def test_compaction_usage_counts_reasoning_reported_outside_output(
    simulator: tuple[_CompactSimulator, str],
) -> None:
    sim, base_url = simulator
    llm = _llm(base_url)

    await compact_conversation(_CompactionAgent(llm), settings=CompactionSettings(keep_turns=1))

    usage = sim.response["usage"]
    reasoning = usage["output_tokens_details"]["reasoning_tokens"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"] + reasoning
    [turn] = llm.usage_accumulator.turns
    assert turn.prompt.total == usage["input_tokens"]
    assert turn.completion.total == usage["output_tokens"] + reasoning
    assert turn.completion.reasoning == reasoning


@pytest.mark.asyncio
async def test_compact_reply_without_an_item_leaves_history_unchanged(
    simulator: tuple[_CompactSimulator, str],
) -> None:
    sim, base_url = simulator
    sim.response = {**sim.response, "output": []}
    agent = _CompactionAgent(_llm(base_url))
    history = list(agent.message_history)

    with pytest.raises(CompactionError, match="xAI compaction returned no compaction item"):
        await compact_conversation(agent, settings=CompactionSettings(keep_turns=1))
    assert agent.message_history == history


@pytest.mark.parametrize("origin", ["codexresponses", "responses", "copilot", None])
def test_xai_reads_only_its_own_checkpoints(origin: str | None) -> None:
    llm = _llm()

    [item] = llm._convert_message_to_items(_checkpoint("xai"))
    assert item["type"] == "compaction"
    with pytest.raises(ProviderCheckpointError, match="compacted on"):
        llm._convert_message_to_items(_checkpoint(origin))


def test_openai_routes_reject_xai_checkpoints() -> None:
    settings = Settings(responses=OpenAISettings(api_key="test"))
    llm = ResponsesLLM(context=Context(config=settings), model="gpt-6-sol")

    with pytest.raises(ProviderCheckpointError, match=r"compacted on 'xai'"):
        llm._convert_message_to_items(_checkpoint("xai"))


def _usage(current: int) -> UsageAccumulator:
    usage = UsageAccumulator()
    usage.set_context_window_size(500_000)
    usage.set_context_estimate(current)
    return usage


@pytest.mark.parametrize("model", ["grok-4.7", "grok-4.7-build-fast"])
def test_grok_47_compacts_below_the_long_context_price_tier(model: str) -> None:
    llm = _llm(model=model)
    settings = CompactionSettings()

    # Rates double above 200k: compact at 190k of the 500k window.
    assert should_auto_compact(_usage(190_000), settings, llm=llm)
    assert not should_auto_compact(_usage(189_000), settings, llm=llm)


@pytest.mark.parametrize(
    ("compact_at", "threshold", "fires_at"),
    [
        (None, 0.85, 425_000),  # config overrides the model default
        ("150k", 0.85, 150_000),  # the model string overrides config
        ("60%", None, 300_000),
    ],
)
def test_compaction_trigger_precedence(
    compact_at: str | None, threshold: float | None, fires_at: int
) -> None:
    llm = _llm(compact_at=CompactAt.parse(compact_at) if compact_at else None)
    settings = CompactionSettings(threshold=threshold)

    assert should_auto_compact(_usage(fires_at), settings, llm=llm)
    assert not should_auto_compact(_usage(fires_at - 1), settings, llm=llm)
