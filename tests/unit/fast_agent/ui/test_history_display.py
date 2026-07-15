import json
from types import SimpleNamespace
from typing import Any, cast

from mcp.types import CallToolResult, ImageContent, TextContent
from rich.console import Console

from fast_agent.constants import ANTHROPIC_SERVER_TOOLS_CHANNEL, FAST_AGENT_TIMING, FAST_AGENT_USAGE
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ui.history_display import (
    SUMMARY_COUNT,
    _build_history_bar,
    _build_history_rows,
    _extract_tool_result_summary,
    _message_role,
    _shade_block,
    display_history_show,
)
from fast_agent.ui.history_display_models import HistoryTimelineEntry


def test_history_overview_summary_window_shows_twelve_rows() -> None:
    assert SUMMARY_COUNT == 12


def test_extract_tool_result_summary_returns_named_fields_for_mixed_content() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="hello\nworld"),
            ImageContent(type="image", data="abc", mimeType="image/png"),
        ]
    )

    summary = _extract_tool_result_summary(result)

    assert summary.preview == "hello world"
    assert summary.chars == len("hello world")
    assert summary.non_text is True


def test_shade_block_uses_expected_threshold_markers() -> None:
    assert _shade_block(0, non_text=False, color="red").plain == "·"
    assert _shade_block(49, non_text=False, color="red").plain == "░"
    assert _shade_block(50, non_text=False, color="red").plain == "▒"
    assert _shade_block(500, non_text=False, color="red").plain == "▓"
    assert _shade_block(2000, non_text=False, color="red").plain == "█"
    assert _shade_block(1, non_text=True, color="red").plain == "^"


def test_build_history_bar_uses_singular_turn_count() -> None:
    bar = _build_history_bar(
        [HistoryTimelineEntry(role="user", chars=1, non_text=False, is_error=False)]
    )

    assert bar.detail.plain == "1 turn"


def test_display_history_show_includes_ttft_and_response_columns() -> None:
    history = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="world")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "start_time": 10.0,
                                "end_time": 10.4,
                                "duration_ms": 400,
                                "ttft_ms": 120,
                                "time_to_response_ms": 240,
                            }
                        ),
                    )
                ],
                FAST_AGENT_USAGE: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "schema": "fast-agent.usage/v2",
                                "provider_attempts": [
                                    {
                                        "provider": "openai",
                                        "usage_schema": "openai-chat",
                                        "model": "test",
                                        "prompt": {"total": 10},
                                        "completion": {"total": 8},
                                        "raw_usage": {},
                                    }
                                ],
                            }
                        ),
                    )
                ],
            },
        ),
    ]
    console = Console(record=True, width=120)

    display_history_show("test-agent", history, console=console)

    output = console.export_text()
    assert "Avg TTFT:" in output
    assert "Avg Resp:" in output
    assert "TTFT" in output
    assert "Resp" in output


def test_build_history_rows_places_provider_tool_activity_before_assistant_row() -> None:
    history = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="who am i?")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="You're evalstate.")],
            channels={
                ANTHROPIC_SERVER_TOOLS_CHANNEL: [
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

    rows = _build_history_rows(history)

    assert [row.role for row in rows] == ["user", "tool", "tool", "assistant"]
    assert rows[1].preview == "{}"
    assert rows[1].label == "remote tool call"
    assert rows[1].arrow == "◀"
    assert rows[2].preview == "evalstate"
    assert rows[2].label == "remote tool result"
    assert rows[3].preview == "You're evalstate."


def test_message_role_normalizes_role_case() -> None:
    message = cast("Any", SimpleNamespace(role="USER"))

    assert _message_role(message) == "user"
