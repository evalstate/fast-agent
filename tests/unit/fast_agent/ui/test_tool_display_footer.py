from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay


def test_tool_call_hides_redundant_single_tool_footer() -> None:
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="hf_hub_query_raw",
            tool_args={"code": "x = 1"},
            bottom_items=["hf_hub_query_raw"],
            highlight_index=0,
            name="hub_search",
        )

    rendered = capture.get()
    assert "tool call - hf_hub_query_raw" in rendered
    assert "\n▎ hf_hub_query_raw\n" not in rendered


def test_tool_call_keeps_footer_when_multiple_tools_are_available() -> None:
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        display.show_tool_call(
            tool_name="hf_hub_query_raw",
            tool_args={"code": "x = 1"},
            bottom_items=["hf_hub_query_raw", "hf_trending"],
            highlight_index=0,
            name="hub_search",
        )

    rendered = capture.get()
    assert "\n▎ hf_hub_query_raw • hf_trending\n" in rendered
