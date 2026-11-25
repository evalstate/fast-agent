from __future__ import annotations

from typing import Any, ClassVar, Literal

import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.config import LoggerSettings, Settings
from fast_agent.constants import (
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    BUILTIN_SUBAGENT_TOOL_NAME,
    FAST_AGENT_TOOL_METADATA,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ui import history_actions


class _CaptureDisplay:
    events: ClassVar[list[str]] = []
    assistant_calls: ClassVar[list[dict[str, object]]] = []
    tool_call_metadata: ClassVar[list[dict[str, object] | None]] = []
    tool_calls: ClassVar[list[dict[str, object]]] = []
    user_panels: ClassVar[list[tuple[object, object, object]]] = []

    def __init__(self, config=None) -> None:
        del config

    def show_user_message(self, *args: Any, **kwargs: Any) -> None:
        del args
        self.user_panels.append(
            (kwargs.get("name"), kwargs.get("turn_range"), kwargs.get("from_hc"))
        )

    async def show_assistant_message(self, *args: Any, **kwargs: Any) -> None:
        del args
        self.events.append("assistant")
        self.assistant_calls.append(
            {
                "name": kwargs.get("name"),
                "render_markdown": kwargs.get("render_markdown"),
            }
        )

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

    def show_history_tool_result(self, *args: Any, **kwargs: Any) -> None:
        self.show_tool_result(*args, **kwargs)


def _capture_history_display(monkeypatch) -> type[_CaptureDisplay]:
    _CaptureDisplay.events = []
    _CaptureDisplay.assistant_calls = []
    _CaptureDisplay.tool_call_metadata = []
    _CaptureDisplay.tool_calls = []
    _CaptureDisplay.user_panels = []
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
@pytest.mark.parametrize("commentary", [None, "Waiting for the checks."])
async def test_review_user_turn_preserves_process_activity_without_empty_bubble(
    monkeypatch, commentary: str | None
) -> None:
    from mcp_types import CallToolResult

    from fast_agent.commands.history_summaries import collect_user_turns
    from fast_agent.types.llm_stop_reason import LlmStopReason

    display = _capture_history_display(monkeypatch)
    history = [
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="Run checks")]),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=commentary)] if commentary else [],
            tool_calls={
                "wait_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="process", arguments={"action": "wait", "process_id": "process-1"}
                    ),
                )
            },
            stop_reason=LlmStopReason.TOOL_USE,
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "wait_1": CallToolResult(content=[TextContent(type="text", text="Checks passed")])
            },
        ),
        PromptMessageExtended(
            role="assistant", content=[TextContent(type="text", text="All checks passed.")]
        ),
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="What next?")]),
    ]
    turns = collect_user_turns(history)
    assert len(turns) == 2
    assert list(turns[0].messages) == history[:4]

    await history_actions.display_history_turn("demo", list(turns[0].messages), config=None)

    expected = ["tool_call", "tool_result", "assistant"]
    if commentary:
        expected.insert(0, "assistant")
    assert display.events == expected


@pytest.mark.asyncio
async def test_display_history_turn_headings_show_who_sent_each_message(monkeypatch) -> None:
    display = _capture_history_display(monkeypatch)

    def user(text: str) -> PromptMessageExtended:
        return PromptMessageExtended(role="user", content=[TextContent(type="text", text=text)])

    turn = [
        user("delegate the build"),
        PromptMessageExtended(role="assistant", content=[TextContent(type="text", text="ok")]),
        user('<hc from="t7">build is green</hc>'),
        PromptMessageExtended(role="assistant", content=[TextContent(type="text", text="done")]),
    ]

    await history_actions.display_history_turn(
        "dev", turn, config=None, turn_index=2, total_turns=3
    )

    assert display.user_panels == [("dev", (2, 2), False), ("hc · t7", None, True)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("streaming_mode", "expected_render_markdown"),
    [
        (None, True),
        ("markdown", True),
        ("none", None),
        ("plain", False),
    ],
)
async def test_display_history_turn_sets_render_markdown_from_streaming_mode(
    monkeypatch,
    streaming_mode: Literal["markdown", "plain", "none"] | None,
    expected_render_markdown: bool | None,
) -> None:
    import json

    display = _capture_history_display(monkeypatch)
    config = (
        None
        if streaming_mode is None
        else Settings(logger=LoggerSettings(streaming=streaming_mode))
    )
    turn = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Delegating now.")],
            tool_calls={
                "sub_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="subagent",
                        arguments={"label": "worker", "message": "check tokens"},
                    ),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "sub_1": CallToolResult(
                    content=[TextContent(type="text", text="Found `<|im_end|>` token.")]
                )
            },
            channels={
                FAST_AGENT_TOOL_METADATA: [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"sub_1": {"fast_agent": {"builtin": BUILTIN_SUBAGENT_TOOL_NAME}}}
                        ),
                    )
                ]
            },
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="**Summary**: done")],
        ),
    ]

    await history_actions.display_history_turn("demo", turn, config=config)

    assert len(display.assistant_calls) == 3
    assert display.assistant_calls[0]["render_markdown"] is expected_render_markdown
    assert display.assistant_calls[1]["render_markdown"] is None
    assert display.assistant_calls[2]["render_markdown"] is expected_render_markdown


@pytest.mark.asyncio
async def test_display_history_turn_renders_when_chat_and_tools_disabled() -> None:
    from fast_agent.ui import console

    config = Settings(logger=LoggerSettings(show_chat=False, show_tools=False))
    turn = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="please check status")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="demo_tool", arguments={"q": "status"}),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            content=[],
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="tool output payload")]
                )
            },
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Everything looks good.")],
        ),
    ]

    with console.console.capture() as capture:
        await history_actions.display_history_turn(
            "demo",
            turn,
            config=config,
            turn_index=1,
            total_turns=1,
        )

    output = capture.get()
    assert "please check status" in output
    assert "demo_tool" in output
    assert "tool output payload" in output
    assert "Everything looks good." in output
