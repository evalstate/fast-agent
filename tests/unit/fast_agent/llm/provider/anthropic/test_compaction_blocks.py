"""Anthropic server-side compaction blocks survive history and continue the turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from anthropic.types.beta import (
    BetaCompactionBlock,
    BetaMessage,
    BetaRawContentBlockStartEvent,
    BetaTextBlock,
    BetaUsage,
)
from mcp_types import TextContent

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.anthropic import llm_anthropic
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM, _AnthropicStreamState
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.core.logging.events import Event

MODEL = "claude-test"
COMPACTION_WIRE_BLOCK = {"type": "compaction", "content": "summary of work", "signature": "sig"}


def _llm() -> AnthropicLLM:
    context = Context()
    context.config = Settings(anthropic=AnthropicSettings(api_key="test-key"))
    return AnthropicLLM(context=context, model=MODEL)


def _response(*content: Any, stop_reason: Any = "end_turn") -> BetaMessage:
    return BetaMessage(
        id="msg_1",
        type="message",
        role="assistant",
        content=list(content),
        model=MODEL,
        stop_reason=stop_reason,
        usage=BetaUsage(input_tokens=1, output_tokens=1),
    )


def _compaction_block() -> BetaCompactionBlock:
    return BetaCompactionBlock(type="compaction", content="summary of work", signature="sig")


def _replayed(llm: AnthropicLLM, response: BetaMessage) -> list[Any]:
    """Store a response the way history does, then convert it back for the next request."""
    message = PromptMessageExtended(
        role="assistant",
        content=[
            TextContent(type="text", text=block.text)
            for block in response.content
            if isinstance(block, BetaTextBlock)
        ],
        channels=llm._anthropic_response_channels(response, MODEL, [], None),
    )
    return list(AnthropicConverter.convert_to_anthropic(message)["content"])


@pytest.mark.parametrize(
    "response",
    [
        _response(_compaction_block(), BetaTextBlock(type="text", text="continuing")),
        _response(_compaction_block(), stop_reason="compaction"),
    ],
    ids=["compaction-then-text", "compaction-only"],
)
def test_compaction_block_replays_verbatim_and_first(response: BetaMessage) -> None:
    replayed = _replayed(_llm(), response)

    assert replayed[0] == COMPACTION_WIRE_BLOCK
    assert [block["type"] for block in replayed] == [block.type for block in response.content]


def test_chained_message_param_keeps_compaction_block() -> None:
    response = _response(_compaction_block(), BetaTextBlock(type="text", text="continuing"))

    param = AnthropicLLM.convert_message_to_message_param(response)

    assert list(param["content"])[0] == COMPACTION_WIRE_BLOCK


@pytest.mark.asyncio
async def test_compaction_stop_reason_continues_the_turn() -> None:
    result = await _llm()._anthropic_stop_result(
        _response(_compaction_block(), stop_reason="compaction"), None, None, None, []
    )

    assert result.stop_reason == LlmStopReason.CONTINUE


@pytest.mark.asyncio
async def test_continue_assistant_message_requests_another_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm()
    calls: list[PromptMessageExtended | None] = []

    async def fake_completion(_message_param: object, *_args: object, **kwargs: Any) -> Any:
        calls.append(kwargs["current_extended"])
        return PromptMessageExtended(role="assistant", content=[])

    monkeypatch.setattr(llm, "_anthropic_completion", fake_completion)
    paused = PromptMessageExtended(role="assistant", content=[], stop_reason=LlmStopReason.CONTINUE)
    finished = PromptMessageExtended(
        role="assistant", content=[], stop_reason=LlmStopReason.END_TURN
    )

    await llm._apply_prompt_provider_specific([paused])
    assert await llm._apply_prompt_provider_specific([finished]) is finished

    assert calls == [paused]


def test_compaction_block_start_reports_compacting_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Event] = []
    monkeypatch.setattr(llm_anthropic.logger, "_emit_event", events.append)

    _llm()._handle_anthropic_content_block_start(
        BetaRawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=BetaCompactionBlock(type="compaction"),
        ),
        MODEL,
        _AnthropicStreamState(),
    )

    assert [event.data["data"]["progress_action"] for event in events] == [
        ProgressAction.COMPACTING_IN_REQUEST
    ]


@pytest.mark.parametrize("with_compaction", [True, False])
def test_compaction_reply_is_marked_as_a_provider_checkpoint(with_compaction: bool) -> None:
    from fast_agent.constants import FAST_AGENT_PROVIDER_CHECKPOINT

    blocks: list[Any] = [BetaTextBlock(type="text", text="continuing")]
    if with_compaction:
        blocks.insert(0, _compaction_block())

    channels = _llm()._anthropic_response_channels(_response(*blocks), MODEL, [], None) or {}

    # History rewrites (process-poll folding) must keep checkpoint-bearing replies.
    assert (FAST_AGENT_PROVIDER_CHECKPOINT in channels) is with_compaction
