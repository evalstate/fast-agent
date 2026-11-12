import json

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.commands.history_summaries import build_history_turn_report
from fast_agent.constants import (
    ANTHROPIC_ASSISTANT_RAW_CONTENT,
    FAST_AGENT_TIMING,
    FAST_AGENT_TOOL_TIMING,
    FAST_AGENT_USAGE,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def _timing_payload(
    *,
    start_time: float,
    end_time: float,
    duration_ms: float,
    ttft_ms: float | None = None,
    time_to_response_ms: float | None = None,
) -> dict[str, float]:
    payload: dict[str, float] = {
        "start_time": start_time,
        "end_time": end_time,
        "duration_ms": duration_ms,
    }
    if ttft_ms is not None:
        payload["ttft_ms"] = ttft_ms
    if time_to_response_ms is not None:
        payload["time_to_response_ms"] = time_to_response_ms
    return payload


def _usage_payload(output_tokens: int) -> str:
    return json.dumps(
        {
            "schema": "fast-agent.usage/v2",
            "provider_attempts": [
                {
                    "provider": "openai",
                    "usage_schema": "openai-chat",
                    "model": "test",
                    "prompt": {"total": 10},
                    "completion": {"total": output_tokens},
                    "raw_usage": {},
                }
            ],
        }
    )


def test_build_history_turn_report_calculates_turn_metrics() -> None:
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Find the answer")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Checking...")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="lookup", arguments={}),
                )
            },
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            _timing_payload(
                                start_time=10.0,
                                end_time=10.4,
                                duration_ms=400,
                                ttft_ms=100,
                                time_to_response_ms=160,
                            )
                        ),
                    )
                ],
                FAST_AGENT_USAGE: [TextContent(type="text", text=_usage_payload(8))],
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="result")],
                    is_error=False,
                )
            },
            channels={
                FAST_AGENT_TOOL_TIMING: [
                    TextContent(
                        type="text",
                        text='{"call_1": {"timing_ms": 250, "transport_channel": "post-sse"}}',
                    )
                ]
            },
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Done")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            _timing_payload(
                                start_time=10.65,
                                end_time=11.15,
                                duration_ms=500,
                            )
                        ),
                    )
                ],
                FAST_AGENT_USAGE: [TextContent(type="text", text=_usage_payload(12))],
            },
        ),
    ]

    report = build_history_turn_report(messages)

    assert report.turn_count == 1
    assert report.total_tool_calls == 1
    assert report.total_tool_errors == 0
    assert report.total_llm_time_ms == 900
    assert report.total_tool_time_ms == 250
    assert report.total_turn_time_ms == 1150
    assert report.average_ttft_ms == 100
    assert report.average_response_ms == 160

    turn = report.turns[0]
    assert turn.user_snippet == "Find the answer"
    assert turn.assistant_snippet == "Done"
    assert turn.turn_time_ms == 1150
    assert turn.tool_time_ms == 250
    assert turn.ttft_ms == 100
    assert turn.response_ms == 160
    assert turn.output_tokens == 20
    assert turn.tps is not None
    assert round(turn.tps, 1) == 25.0


def test_build_history_turn_report_counts_provider_mcp_tools() -> None:
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Who am I?")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="You're evalstate.")],
            channels={
                ANTHROPIC_ASSISTANT_RAW_CONTENT: [
                    TextContent(
                        type="text",
                        text='{"type":"mcp_tool_use","id":"mcptoolu_1","name":"hf_whoami","server_name":"huggingface_mcp","input":{}}',
                    ),
                    TextContent(
                        type="text",
                        text='{"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","is_error":false,"content":[{"type":"text","text":"evalstate"}]}',
                    ),
                ]
            },
        ),
    ]

    report = build_history_turn_report(messages)

    assert report.turn_count == 1
    assert report.total_tool_calls == 1
    assert report.total_tool_errors == 0


def _measured_response(
    output_tokens: int | None,
    duration_ms: float | None,
    ttft_ms: float | None = None,
    response_ms: float | None = None,
) -> PromptMessageExtended:
    channels: dict[str, list[TextContent]] = {}
    if output_tokens is not None:
        channels[FAST_AGENT_USAGE] = [TextContent(type="text", text=_usage_payload(output_tokens))]
    if duration_ms is not None:
        channels[FAST_AGENT_TIMING] = [
            TextContent(
                type="text",
                text=json.dumps(
                    _timing_payload(
                        start_time=0,
                        end_time=duration_ms / 1000,
                        duration_ms=duration_ms,
                        ttft_ms=ttft_ms,
                        time_to_response_ms=response_ms,
                    )
                ),
            )
        ]
    return PromptMessageExtended(role="assistant", channels=channels)


@pytest.mark.parametrize("response_ms", [None, 1495.87, 31036.84, 37900])
def test_history_tps_includes_reasoning_generation_time(response_ms: float | None) -> None:
    report = build_history_turn_report(
        [
            PromptMessageExtended(role="user"),
            _measured_response(1296, 37930.12, 1495.87, response_ms),
        ]
    )
    assert report.turns[0].tps == pytest.approx(1296 / 36.43425)
    assert report.average_tps == report.turns[0].tps
    if response_ms is not None:
        assert report.turns[0].response_ms == response_ms


def test_history_tps_aggregates_matching_per_call_generation_windows() -> None:
    report = build_history_turn_report(
        [
            PromptMessageExtended(role="user"),
            _measured_response(10, 2000, 1000, 1500),
            _measured_response(90, 6000, 3000, 5000),
            # Unpaired telemetry must not inflate either side of the ratio.
            _measured_response(1000, None),
            _measured_response(None, 10000),
        ]
    )
    turn = report.turns[0]
    assert turn.tps == pytest.approx(100 / 4)
    assert turn.output_tokens == 1100
    assert turn.llm_time_ms == 18000
    assert turn.ttft_ms == 1000
    assert turn.response_ms == 1500


@pytest.mark.parametrize("ttft_ms", [None, 0, 1000, 2000])
def test_history_tps_falls_back_to_full_call_duration(ttft_ms: float | None) -> None:
    report = build_history_turn_report(
        [PromptMessageExtended(role="user"), _measured_response(20, 1000, ttft_ms, 900)]
    )
    assert report.turns[0].tps == 20


@pytest.mark.parametrize("tokens,duration", [(None, 1000), (20, None), (20, 0), (0, 1000)])
def test_history_tps_without_usable_measurement(tokens: int | None, duration: float | None) -> None:
    report = build_history_turn_report(
        [PromptMessageExtended(role="user"), _measured_response(tokens, duration)]
    )
    assert report.turns[0].tps is None
    assert report.average_tps is None
