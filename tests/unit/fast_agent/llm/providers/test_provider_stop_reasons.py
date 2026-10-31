from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, get_args

import botocore.session
import pytest
from anthropic.types.beta import BetaMessage, BetaStopReason, BetaUsage

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.core.logging.logger import Logger
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.core.logging.events import EventContext, EventType


class _RecordingLogger(Logger):
    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.warning_messages: list[str] = []

    def event(
        self,
        etype: EventType,
        ename: str | None,
        message: str,
        context: EventContext | None,
        data: dict,
    ) -> None:
        del ename, context, data
        if etype == "debug":
            self.debug_messages.append(message)
        elif etype == "warning":
            self.warning_messages.append(message)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("tool_use", LlmStopReason.TOOL_USE),
        ("end_turn", LlmStopReason.END_TURN),
        ("stop_sequence", LlmStopReason.STOP_SEQUENCE),
        ("max_tokens", LlmStopReason.MAX_TOKENS),
        ("model_context_window_exceeded", LlmStopReason.MAX_TOKENS),
        ("guardrail_intervened", LlmStopReason.SAFETY),
        ("content_filtered", LlmStopReason.SAFETY),
    ],
)
def test_bedrock_stop_reason_mapping(reason: str, expected: LlmStopReason) -> None:
    llm = object.__new__(BedrockLLM)

    assert BedrockLLM._map_bedrock_stop_reason(llm, reason) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "reason",
    botocore.session.get_session()
    .get_service_model("bedrock-runtime")
    .shape_for("StopReason")
    .enum,
)
def test_bedrock_maps_every_documented_stop_reason(reason: str) -> None:
    llm = object.__new__(BedrockLLM)
    logger = _RecordingLogger()
    llm.logger = logger

    BedrockLLM._map_bedrock_stop_reason(llm, reason)

    assert logger.warning_messages == []


@pytest.mark.unit
def test_bedrock_unknown_stop_reason_defaults_to_end_turn_with_warning() -> None:
    llm = object.__new__(BedrockLLM)
    logger = _RecordingLogger()
    llm.logger = logger

    result = BedrockLLM._map_bedrock_stop_reason(llm, "unexpected")

    assert result is LlmStopReason.END_TURN
    assert logger.warning_messages == [
        "Unknown Bedrock stop reason: unexpected, defaulting to END_TURN"
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("length", LlmStopReason.MAX_TOKENS),
        ("content_filter", LlmStopReason.SAFETY),
        ("sensitive", LlmStopReason.SAFETY),
        ("stop", LlmStopReason.END_TURN),
    ],
)
def test_openai_stop_reason_mapping(finish_reason: str, expected: LlmStopReason) -> None:
    llm = object.__new__(OpenAILLM)
    llm.logger = _RecordingLogger()
    choice = SimpleNamespace(finish_reason=finish_reason)
    message = SimpleNamespace(tool_calls=None)

    result = asyncio.run(OpenAILLM._openai_stop_result(llm, choice, message))

    assert result.stop_reason is expected
    assert result.requested_tool_calls is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "reason",
    [reason for reason in get_args(BetaStopReason) if reason not in ("end_turn", "tool_use")],
)
def test_anthropic_stop_reasons_do_not_collapse_to_end_turn(reason: BetaStopReason) -> None:
    context = Context()
    context.config = Settings(anthropic=AnthropicSettings(api_key="test-key"))
    llm = AnthropicLLM(context=context, model="claude-test")
    response = BetaMessage(
        id="msg_1",
        type="message",
        role="assistant",
        content=[],
        model="claude-test",
        stop_reason=reason,
        usage=BetaUsage(input_tokens=1, output_tokens=1),
    )

    result = asyncio.run(llm._anthropic_stop_result(response, None, None, None, []))

    assert result.stop_reason is not LlmStopReason.END_TURN
