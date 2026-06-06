from __future__ import annotations

from mcp.types import CallToolResult, TextContent
from rich.text import Text

from fast_agent.ui.console_display import ConsoleDisplay, ParallelAgentDisplayResult
from fast_agent.ui.display_suppression import (
    display_chat_enabled,
    display_status_enabled,
    display_tools_enabled,
    display_usage_enabled,
    suppress_interactive_display,
)


def test_display_predicates_share_interactive_transcript_state() -> None:
    assert display_chat_enabled()
    assert display_status_enabled()
    assert display_tools_enabled()
    assert display_usage_enabled()

    with suppress_interactive_display():
        assert not display_chat_enabled()
        assert not display_status_enabled()
        assert not display_tools_enabled()
        assert not display_usage_enabled()


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


def test_progress_only_display_suppresses_url_and_system_messages(capsys) -> None:
    display = ConsoleDisplay()

    with suppress_interactive_display():
        display.show_url_elicitation(
            message="hidden url request",
            url="https://example.com/login",
            server_name="hidden-server",
        )
        display.show_system_message(
            "hidden system prompt",
            agent_name="hidden-agent",
            server_count=1,
        )

    output = capsys.readouterr().out
    assert output == ""


def test_progress_only_display_suppresses_parallel_results(capsys, monkeypatch) -> None:
    display = ConsoleDisplay()
    monkeypatch.setattr(
        display,
        "_parallel_agent_results",
        lambda _parallel_agent: [
            ParallelAgentDisplayResult(
                name="hidden-agent",
                model="hidden-model",
                content="hidden parallel result",
                tokens=12,
                tool_calls=1,
            )
        ],
    )

    with suppress_interactive_display():
        display.show_parallel_results(object())

    output = capsys.readouterr().out
    assert output == ""
