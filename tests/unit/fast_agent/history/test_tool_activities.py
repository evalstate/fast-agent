from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.constants import ANTHROPIC_ASSISTANT_RAW_CONTENT, ANTHROPIC_SERVER_TOOLS_CHANNEL
from fast_agent.history.tool_activities import (
    message_tool_call_count,
    message_tool_error_count,
    remote_tool_activities,
    tool_activities_for_message,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def test_remote_tool_activities_prefer_raw_assistant_order() -> None:
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        channels={
            ANTHROPIC_ASSISTANT_RAW_CONTENT: [
                TextContent(
                    type="text",
                    text='{"type":"mcp_tool_use","id":"mcptoolu_1","name":"hf_hub_query","server_name":"huggingface_mcp","input":{"message":"top models"}}',
                ),
                TextContent(
                    type="text",
                    text='{"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","is_error":false,"content":[{"type":"text","text":"{\\"ok\\":true}"}]}',
                ),
            ],
            ANTHROPIC_SERVER_TOOLS_CHANNEL: [
                TextContent(type="text", text='{"type":"mcp_tool_result","tool_use_id":"ignored"}')
            ],
        },
    )

    events = remote_tool_activities(message)

    assert [event.kind for event in events] == ["call", "result"]
    assert events[0].tool_name == "huggingface_mcp/hf_hub_query"
    assert events[0].arguments == {"message": "top models"}
    assert events[1].result is not None
    assert get_text(events[1].result.content[0]) == '{"ok":true}'
    assert message_tool_call_count(message) == 1
    assert message_tool_error_count(message) == 0


def test_remote_tool_activities_fallback_to_server_tool_channel() -> None:
    message = PromptMessageExtended(
        role="assistant",
        content=[],
        channels={
            ANTHROPIC_SERVER_TOOLS_CHANNEL: [
                TextContent(
                    type="text",
                    text='{"type":"mcp_tool_use","id":"mcptoolu_2","name":"hf_whoami","server_name":"huggingface_mcp","input":{}}',
                ),
                TextContent(
                    type="text",
                    text='{"type":"mcp_tool_result","tool_use_id":"mcptoolu_2","is_error":true,"content":[{"type":"text","text":"forbidden"}]}',
                ),
            ]
        },
    )

    events = remote_tool_activities(message)

    assert [event.tool_name for event in events] == [
        "huggingface_mcp/hf_whoami",
        "huggingface_mcp/hf_whoami",
    ]
    assert events[1].is_error is True
    assert message_tool_call_count(message) == 1
    assert message_tool_error_count(message) == 1


def test_tool_activities_include_standard_tool_calls_and_results() -> None:
    assistant = PromptMessageExtended(
        role="assistant",
        tool_calls={
            "call_1": CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name="lookup", arguments={"q": "hello"}),
            )
        },
    )
    user = PromptMessageExtended(
        role="user",
        tool_results={
            "call_1": CallToolResult(
                content=[TextContent(type="text", text="world")],
                isError=False,
            )
        },
    )

    assistant_activities = tool_activities_for_message(assistant)
    user_activities = tool_activities_for_message(user, tool_name_lookup={"call_1": "lookup"})

    assert [(item.kind, item.tool_name, item.is_remote) for item in assistant_activities] == [
        ("call", "lookup", False)
    ]
    assert [(item.kind, item.tool_name, item.is_remote) for item in user_activities] == [
        ("result", "lookup", False)
    ]
