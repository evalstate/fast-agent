from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from fast_agent.llm.request_params import (
    STRUCTURED_TOOL_POLICY_VALUES,
    RequestParams,
    is_structured_tool_policy,
    response_mode_to_tool_result_mode,
    tool_result_mode_allows_response_mode,
    tool_result_mode_is_passthrough,
)


def test_is_structured_tool_policy_accepts_declared_values() -> None:
    for value in STRUCTURED_TOOL_POLICY_VALUES:
        assert is_structured_tool_policy(value)


def test_is_structured_tool_policy_rejects_unknown_values() -> None:
    assert not is_structured_tool_policy("sometimes")
    assert not is_structured_tool_policy(None)


def test_response_mode_values_map_to_tool_result_mode_overrides() -> None:
    assert response_mode_to_tool_result_mode("inherit") is None
    assert response_mode_to_tool_result_mode("postprocess") == "postprocess"
    assert response_mode_to_tool_result_mode("passthrough") == "passthrough"


def test_tool_result_mode_policy_helpers() -> None:
    assert not tool_result_mode_allows_response_mode("postprocess")
    assert not tool_result_mode_allows_response_mode("passthrough")
    assert tool_result_mode_allows_response_mode("selectable")
    assert not tool_result_mode_is_passthrough("postprocess")
    assert tool_result_mode_is_passthrough("passthrough")
    assert not tool_result_mode_is_passthrough("selectable")


def test_request_params_messages_default_is_not_shared() -> None:
    first = RequestParams()
    second = RequestParams()

    assert first.messages is not second.messages


def test_request_params_rejects_bool_numeric_values() -> None:
    numeric_fields = (
        "maxTokens",
        "max_iterations",
        "streaming_timeout",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    )

    for field_name in numeric_fields:
        payload: dict[str, Any] = {field_name: True}
        with pytest.raises(ValidationError, match="must not be boolean"):
            RequestParams(**payload)


def test_request_params_accepts_numeric_values() -> None:
    params = RequestParams(
        maxTokens=10,
        max_iterations=2,
        streaming_timeout=3.5,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        min_p=0.1,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        repetition_penalty=1.1,
    )

    assert params.maxTokens == 10
    assert params.max_iterations == 2
    assert params.streaming_timeout == 3.5
    assert params.temperature == 0.7
    assert params.top_p == 0.9
    assert params.top_k == 40
    assert params.min_p == 0.1
    assert params.presence_penalty == 0.2
    assert params.frequency_penalty == 0.3
    assert params.repetition_penalty == 1.1
