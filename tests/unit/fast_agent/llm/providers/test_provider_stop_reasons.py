from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.types.llm_stop_reason import LlmStopReason


class _Logger:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.warning_messages: list[str] = []

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("tool_use", LlmStopReason.TOOL_USE),
        ("end_turn", LlmStopReason.END_TURN),
        ("stop_sequence", LlmStopReason.STOP_SEQUENCE),
        ("max_tokens", LlmStopReason.MAX_TOKENS),
    ],
)
def test_bedrock_stop_reason_mapping(reason: str, expected: LlmStopReason) -> None:
    llm = object.__new__(BedrockLLM)

    assert BedrockLLM._map_bedrock_stop_reason(llm, reason) is expected


@pytest.mark.unit
def test_bedrock_unknown_stop_reason_defaults_to_end_turn_with_warning() -> None:
    llm = object.__new__(BedrockLLM)
    logger = _Logger()
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
        ("stop", LlmStopReason.END_TURN),
    ],
)
def test_openai_stop_reason_mapping(finish_reason: str, expected: LlmStopReason) -> None:
    llm = object.__new__(OpenAILLM)
    llm.logger = _Logger()
    choice = SimpleNamespace(finish_reason=finish_reason)
    message = SimpleNamespace(tool_calls=None)

    result = asyncio.run(OpenAILLM._openai_stop_result(llm, choice, message))

    assert result.stop_reason is expected
    assert result.requested_tool_calls is None
