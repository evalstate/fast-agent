from __future__ import annotations

from mcp.types import CallToolResult, TextContent
from rich.text import Text

from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.display_suppression import suppress_interactive_display


def test_progress_only_display_suppresses_status_and_chat(capsys) -> None:
    display = ConsoleDisplay()

    with suppress_interactive_display():
        display.show_status_message(Text("hidden status"))
        display.show_user_message("hidden user")
        display.show_tool_call("echo", {"value": "hidden"})
        display.show_tool_result(
            CallToolResult(
                content=[TextContent(type="text", text="hidden result")],
                isError=False,
            ),
            tool_name="echo",
        )

    output = capsys.readouterr().out
    assert output == ""


def test_progress_only_display_suppresses_streaming_assistant_output(capsys) -> None:
    display = ConsoleDisplay()

    with suppress_interactive_display():
        with display.streaming_assistant_message(name="demo") as handle:
            handle.update("hidden chunk")
            handle.finalize("hidden final")

    output = capsys.readouterr().out
    assert output == ""
