from __future__ import annotations

from typing import Any

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, TextContent

from fast_agent.constants import ANTHROPIC_SERVER_TOOLS_CHANNEL, FAST_AGENT_TOOL_METADATA
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ui import history_actions


@pytest.mark.asyncio
async def test_display_history_turn_shows_provider_tools_before_assistant(monkeypatch) -> None:
    events: list[str] = []

    class _CaptureDisplay:
        def __init__(self, config=None) -> None:
            del config

        def show_user_message(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def show_assistant_message(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append("assistant")

        def show_tool_call(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append("tool_call")

        def show_tool_result(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append("tool_result")

    monkeypatch.setattr("fast_agent.ui.console_display.ConsoleDisplay", _CaptureDisplay)

    turn = [
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

    await history_actions.display_history_turn("demo", turn, config=None)

    assert events == ["tool_call", "tool_result", "assistant"]


@pytest.mark.asyncio
async def test_display_history_turn_skips_empty_assistant_for_tool_only_remote_turn(
    monkeypatch,
) -> None:
    events: list[str] = []

    class _CaptureDisplay:
        def __init__(self, config=None) -> None:
            del config

        def show_user_message(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def show_assistant_message(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append("assistant")

        def show_tool_call(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append("tool_call")

        def show_tool_result(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            events.append("tool_result")

    monkeypatch.setattr("fast_agent.ui.console_display.ConsoleDisplay", _CaptureDisplay)

    turn = [
        PromptMessageExtended(
            role="assistant",
            content=[],
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

    await history_actions.display_history_turn("demo", turn, config=None)

    assert events == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_display_history_turn_passes_stored_tool_metadata(monkeypatch) -> None:
    seen_metadata: list[dict[str, object] | None] = []

    class _CaptureDisplay:
        def __init__(self, config=None) -> None:
            del config

        def show_user_message(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def show_assistant_message(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def show_tool_call(self, *args: Any, **kwargs: Any) -> None:
            del args
            seen_metadata.append(kwargs.get("metadata"))

        def show_tool_result(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    monkeypatch.setattr("fast_agent.ui.console_display.ConsoleDisplay", _CaptureDisplay)

    turn = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="I'll run the query.")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="run_query",
                        arguments={"code": "print(1)"},
                    ),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={},
            channels={
                FAST_AGENT_TOOL_METADATA: [
                    TextContent(
                        type="text",
                        text='{"call_1":{"variant":"code","code_arg":"code","language":"python"}}',
                    )
                ]
            },
        ),
    ]

    await history_actions.display_history_turn("demo", turn, config=None)

    assert seen_metadata == [{"variant": "code", "code_arg": "code", "language": "python"}]
