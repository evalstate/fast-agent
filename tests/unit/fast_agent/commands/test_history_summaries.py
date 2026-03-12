import json

from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.commands.history_summaries import build_history_turn_report
from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_TOOL_TIMING, FAST_AGENT_USAGE
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
    return json.dumps({"turn": {"output_tokens": output_tokens}, "raw_usage": {}, "summary": {}})


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
                    isError=False,
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
    assert round(turn.tps, 1) == 27.0
