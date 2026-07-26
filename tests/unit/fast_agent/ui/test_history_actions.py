from __future__ import annotations

from typing import Any

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.constants import (
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    FAST_AGENT_SUBAGENT_RESULT_METADATA,
    FAST_AGENT_TOOL_METADATA,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ui import history_actions


class _CaptureDisplay:
    events: list[str] = []
    tool_call_metadata: list[dict[str, object] | None] = []
    tool_calls: list[dict[str, object]] = []

    def __init__(self, config=None) -> None:
        del config

    def show_user_message(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    async def show_assistant_message(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("assistant")

    def show_tool_call(self, *args: Any, **kwargs: Any) -> None:
        del args
        self.events.append("tool_call")
        self.tool_call_metadata.append(kwargs.get("metadata"))
        self.tool_calls.append(
            {
                "tool_name": kwargs.get("tool_name"),
                "tool_args": kwargs.get("tool_args"),
                "tool_call_id": kwargs.get("tool_call_id"),
            }
        )

    def show_tool_result(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.events.append("tool_result")


class _SubagentHistoryDisplay:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def show_user_message(self, **kwargs: object) -> None:
        self.events.append(("user", kwargs))

    async def show_assistant_message(self, **kwargs: object) -> None:
        self.events.append(("assistant", kwargs))

    def show_tool_call(self, **kwargs: object) -> None:
        self.events.append(("tool_call", kwargs))

    def show_tool_result(self, **kwargs: object) -> None:
        self.events.append(("tool_result", kwargs))


def _capture_history_display(monkeypatch) -> type[_CaptureDisplay]:
    _CaptureDisplay.events = []
    _CaptureDisplay.tool_call_metadata = []
    _CaptureDisplay.tool_calls = []
    monkeypatch.setattr("fast_agent.ui.console_display.ConsoleDisplay", _CaptureDisplay)
    return _CaptureDisplay


@pytest.mark.asyncio
async def test_display_history_turn_shows_provider_tools_before_assistant(monkeypatch) -> None:
    display = _capture_history_display(monkeypatch)

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

    assert display.events == ["tool_call", "tool_result", "assistant"]


@pytest.mark.asyncio
async def test_display_history_turn_skips_empty_assistant_for_tool_only_remote_turn(
    monkeypatch,
) -> None:
    display = _capture_history_display(monkeypatch)

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

    assert display.events == ["tool_call", "tool_result"]


@pytest.mark.asyncio
async def test_display_history_turn_passes_stored_tool_metadata(monkeypatch) -> None:
    display = _capture_history_display(monkeypatch)

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

    assert display.tool_call_metadata == [
        {"variant": "code", "code_arg": "code", "language": "python"}
    ]
    assert display.tool_calls == [
        {
            "tool_name": "run_query",
            "tool_args": {"code": "print(1)"},
            "tool_call_id": "call_1",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "requested_label", "label", "request_name"),
    [
        (
            {"message": "research this", "label": "research-pal"},
            "research-pal",
            "research-pal",
            "demo → research-pal",
        ),
        (
            {"message": "research this"},
            None,
            "brisk-otter",
            "demo → subagent",
        ),
        (
            {"message": "research this", "label": "[bold]spoof[/bold]"},
            None,
            "brisk-otter",
            "demo → subagent",
        ),
    ],
)
async def test_history_subagent_replay_uses_chat_panels_and_result_metadata(
    arguments,
    requested_label,
    label,
    request_name,
) -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="child response")],
        _meta={
            FAST_AGENT_SUBAGENT_RESULT_METADATA: {
                "child_session_id": "child-123",
                "child_agent_name": f"demo[{label}]",
                "requested_label": requested_label,
                "label": label,
                "model_spec": "test-model",
                "provider": "fast-agent",
                "status": "completed",
                "usage": {"total_tokens": 5},
                "turn_count": 1,
            }
        },
    )
    turn = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="I'll delegate this.")],
            tool_calls={
                "call-1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="subagent",
                        arguments=arguments,
                    ),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={"call-1": result},
            channels={
                FAST_AGENT_TOOL_METADATA: [
                    TextContent(
                        type="text",
                        text='{"call-1":{"fast_agent":{"builtin":"subagent"}}}',
                    )
                ]
            },
        ),
    ]
    display = _SubagentHistoryDisplay()
    context = history_actions._HistoryTurnDisplayContext(
        display=display,
        agent_name="demo",
        turn_index=None,
        total_turns=None,
        tool_metadata_lookup=history_actions._stored_tool_metadata_from_turn(turn),
    )

    for message in turn:
        context.record_tool_calls(message)
        if not context.queue_user_message(message):
            await context.display_message(message)
    context.flush_user_group()

    assert not [event for event, _ in display.events if event.startswith("tool_")]
    user = next(payload for event, payload in display.events if event == "user")
    assert user["message"] == "research this"
    assert user["name"] == request_name
    child = [payload for event, payload in display.events if event == "assistant"][-1]
    assert child["message_text"] == "child response"
    assert child["name"] == f"subagent: {label}"
    assert child["model"] == "test-model"
    assert child["bottom_items"] == ["session child-123"]
