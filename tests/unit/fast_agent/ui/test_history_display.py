import json

from mcp.types import TextContent
from rich.console import Console

from fast_agent.constants import ANTHROPIC_SERVER_TOOLS_CHANNEL, FAST_AGENT_TIMING, FAST_AGENT_USAGE
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ui.history_display import SUMMARY_COUNT, _build_history_rows, display_history_show


def test_history_overview_summary_window_shows_twelve_rows() -> None:
    assert SUMMARY_COUNT == 12


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
                            {"turn": {"output_tokens": 8}, "raw_usage": {}, "summary": {}}
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

    assert [row["role"] for row in rows] == ["user", "tool", "tool", "assistant"]
    assert rows[1]["preview"] == "{}"
    assert rows[1]["label"] == "remote tool call"
    assert rows[1]["arrow"] == "◀"
    assert rows[2]["preview"] == "evalstate"
    assert rows[2]["label"] == "remote tool result"
    assert rows[3]["preview"] == "You're evalstate."
