"""Tests for ConversationSummary"""

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.types import ConversationSummary, PromptMessageExtended


def test_empty_conversation():
    """Test ConversationSummary with no messages"""
    summary = ConversationSummary(messages=[])

    assert summary.message_count == 0
    assert summary.user_message_count == 0
    assert summary.assistant_message_count == 0
    assert summary.tool_calls == 0
    assert summary.tool_errors == 0
    assert summary.tool_successes == 0
    assert summary.tool_error_rate == 0.0
    assert summary.tool_call_map == {}
    assert summary.tool_error_map == {}
    assert summary.has_tool_calls is False
    assert summary.has_tool_errors is False


def test_simple_conversation():
    """Test ConversationSummary with basic user/assistant messages"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Hi there!")],
        ),
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="How are you?")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="I'm doing well!")],
        ),
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.message_count == 4
    assert summary.user_message_count == 2
    assert summary.assistant_message_count == 2
    assert summary.tool_calls == 0
    assert summary.has_tool_calls is False


def test_tool_calls():
    """Test ConversationSummary with tool calls"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="What's the weather?")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Let me check...")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="get_weather",
                        arguments={"city": "New York"},
                    ),
                ),
                "call_2": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="get_temperature",
                        arguments={"city": "New York"},
                    ),
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Sunny")],
                    isError=False,
                ),
                "call_2": CallToolResult(
                    content=[TextContent(type="text", text="72°F")],
                    isError=False,
                ),
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.message_count == 3
    assert summary.tool_calls == 2
    assert summary.tool_errors == 0
    assert summary.tool_successes == 2
    assert summary.tool_error_rate == 0.0
    assert summary.has_tool_calls is True
    assert summary.has_tool_errors is False
    assert summary.tool_call_map == {"get_weather": 1, "get_temperature": 1}
    assert summary.tool_error_map == {}


def test_tool_errors():
    """Test ConversationSummary with tool errors"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Run some tasks")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Running...")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="task_a", arguments={}),
                ),
                "call_2": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="task_b", arguments={}),
                ),
                "call_3": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="task_a", arguments={}),
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Success")],
                    isError=False,
                ),
                "call_2": CallToolResult(
                    content=[TextContent(type="text", text="Error: failed")],
                    isError=True,
                ),
                "call_3": CallToolResult(
                    content=[TextContent(type="text", text="Error: timeout")],
                    isError=True,
                ),
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.tool_calls == 3
    assert summary.tool_errors == 2
    assert summary.tool_successes == 1
    assert summary.tool_error_rate == pytest.approx(2 / 3)
    assert summary.has_tool_errors is True
    assert summary.tool_call_map == {"task_a": 2, "task_b": 1}
    assert summary.tool_error_map == {"task_a": 1, "task_b": 1}


def test_multiple_tool_call_rounds():
    """Test ConversationSummary with multiple rounds of tool calls"""
    messages = [
        # First round
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="fetch_data", arguments={}),
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Data received")],
                    isError=False,
                ),
            },
        ),
        # Second round
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call_2": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="process_data", arguments={}),
                ),
                "call_3": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="fetch_data", arguments={}),
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_2": CallToolResult(
                    content=[TextContent(type="text", text="Processed")],
                    isError=False,
                ),
                "call_3": CallToolResult(
                    content=[TextContent(type="text", text="Error")],
                    isError=True,
                ),
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.tool_calls == 3
    assert summary.tool_errors == 1
    assert summary.tool_successes == 2
    assert summary.tool_error_rate == pytest.approx(1 / 3)
    assert summary.tool_call_map == {"fetch_data": 2, "process_data": 1}
    assert summary.tool_error_map == {"fetch_data": 1}


def test_model_dump():
    """Test that computed fields are included in model_dump()"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Hi")],
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="test_tool", arguments={}),
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Done")],
                    isError=False,
                ),
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)
    data = summary.model_dump()

    # Check that all computed fields are present in the dump
    assert "message_count" in data
    assert "user_message_count" in data
    assert "assistant_message_count" in data
    assert "tool_calls" in data
    assert "tool_errors" in data
    assert "tool_successes" in data
    assert "tool_error_rate" in data
    assert "tool_call_map" in data
    assert "tool_error_map" in data
    assert "has_tool_calls" in data
    assert "has_tool_errors" in data

    # Verify values
    assert data["message_count"] == 3
    assert data["user_message_count"] == 2
    assert data["assistant_message_count"] == 1
    assert data["tool_calls"] == 1
    assert data["tool_errors"] == 0
    assert data["tool_successes"] == 1
    assert data["tool_call_map"] == {"test_tool": 1}


def test_unknown_tool_id_in_results():
    """Test handling of tool results without corresponding tool calls"""
    messages = [
        # Tool result without a preceding tool call (edge case)
        PromptMessageExtended(
            role="user",
            tool_results={
                "unknown_call": CallToolResult(
                    content=[TextContent(type="text", text="Error")],
                    isError=True,
                ),
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)

    # Should handle gracefully and map to "unknown"
    assert summary.tool_errors == 1
    assert summary.tool_error_map == {"unknown": 1}


def test_tool_call_without_result():
    """Test tool calls that don't have corresponding results"""
    messages = [
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="pending_tool", arguments={}),
                ),
            },
        ),
        # No tool result provided
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.tool_calls == 1
    assert summary.tool_errors == 0
    assert summary.tool_successes == 0
    assert summary.tool_call_map == {"pending_tool": 1}
    assert summary.tool_error_map == {}
