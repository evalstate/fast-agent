import json

from mcp.types import TextContent
from rich.console import Console

from fast_agent.constants import FAST_AGENT_TIMING, FAST_AGENT_USAGE
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ui.history_display import SUMMARY_COUNT, display_history_show


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
