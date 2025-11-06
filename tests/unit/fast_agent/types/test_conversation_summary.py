"""Tests for ConversationSummary"""

import json

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.constants import FAST_AGENT_TIMING
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


def test_timing_data():
    """Test ConversationSummary with timing data in channels"""
    timing_data_1 = {"start_time": 100.0, "end_time": 102.5, "duration_ms": 2500.0}
    timing_data_2 = {"start_time": 105.0, "end_time": 106.2, "duration_ms": 1200.0}

    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Hi there!")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text=json.dumps(timing_data_1))
                ]
            },
        ),
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="How are you?")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="I'm doing well!")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text=json.dumps(timing_data_2))
                ]
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.total_elapsed_time_ms == 3700.0  # 2500 + 1200
    assert len(summary.assistant_message_timings) == 2
    assert summary.assistant_message_timings[0] == timing_data_1
    assert summary.assistant_message_timings[1] == timing_data_2
    assert summary.average_assistant_response_time_ms == pytest.approx(1850.0)  # (2500 + 1200) / 2


def test_timing_no_data():
    """Test ConversationSummary when there's no timing data"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Hello")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Hi there!")],
            # No timing channel
        ),
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.total_elapsed_time_ms == 0.0
    assert summary.assistant_message_timings == []
    assert summary.average_assistant_response_time_ms == 0.0


def test_timing_partial_data():
    """Test ConversationSummary with some messages having timing data"""
    timing_data = {"start_time": 100.0, "end_time": 102.5, "duration_ms": 2500.0}

    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="First response")],
            # No timing
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Second response")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text=json.dumps(timing_data))
                ]
            },
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Third response")],
            # No timing
        ),
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.total_elapsed_time_ms == 2500.0  # Only the one with timing
    assert len(summary.assistant_message_timings) == 1
    assert summary.average_assistant_response_time_ms == 2500.0


def test_timing_invalid_json():
    """Test ConversationSummary handles invalid timing JSON gracefully"""
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Response")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text="invalid json{}")
                ]
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)

    # Should handle gracefully - no errors, just no timing data
    assert summary.total_elapsed_time_ms == 0.0
    assert summary.assistant_message_timings == []
    assert summary.average_assistant_response_time_ms == 0.0


def test_timing_in_model_dump():
    """Test that timing properties are included in model_dump()"""
    timing_data = {"start_time": 100.0, "end_time": 102.5, "duration_ms": 2500.0}

    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Response")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text=json.dumps(timing_data))
                ]
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)
    data = summary.model_dump()

    # Check that timing fields are in the dump
    assert "total_elapsed_time_ms" in data
    assert "assistant_message_timings" in data
    assert "average_assistant_response_time_ms" in data

    # Verify values
    assert data["total_elapsed_time_ms"] == 2500.0
    assert data["average_assistant_response_time_ms"] == 2500.0
    assert len(data["assistant_message_timings"]) == 1


def test_conversation_span():
    """Test conversation_span_ms calculation"""
    timing_data_1 = {"start_time": 100.0, "end_time": 102.5, "duration_ms": 2500.0}
    timing_data_2 = {"start_time": 105.0, "end_time": 106.2, "duration_ms": 1200.0}
    timing_data_3 = {"start_time": 108.0, "end_time": 109.0, "duration_ms": 1000.0}

    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="First")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text=json.dumps(timing_data_1))
                ]
            },
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Second")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text=json.dumps(timing_data_2))
                ]
            },
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Third")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text=json.dumps(timing_data_3))
                ]
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)

    # Conversation span = last end - first start = 109.0 - 100.0 = 9.0 seconds = 9000ms
    assert summary.conversation_span_ms == 9000.0
    assert summary.first_llm_start_time == 100.0
    assert summary.last_llm_end_time == 109.0

    # Total elapsed time = sum of durations = 2500 + 1200 + 1000 = 4700ms
    assert summary.total_elapsed_time_ms == 4700.0

    # The difference shows time spent in tools/orchestration
    overhead_ms = summary.conversation_span_ms - summary.total_elapsed_time_ms
    assert overhead_ms == pytest.approx(4300.0)  # 9000 - 4700


def test_conversation_span_no_timing():
    """Test conversation_span_ms when there's no timing data"""
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Response")],
        ),
    ]

    summary = ConversationSummary(messages=messages)

    assert summary.conversation_span_ms == 0.0
    assert summary.first_llm_start_time is None
    assert summary.last_llm_end_time is None


def test_conversation_span_single_message():
    """Test conversation_span_ms with a single message"""
    timing_data = {"start_time": 100.0, "end_time": 102.5, "duration_ms": 2500.0}

    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Response")],
            channels={
                FAST_AGENT_TIMING: [
                    TextContent(type="text", text=json.dumps(timing_data))
                ]
            },
        ),
    ]

    summary = ConversationSummary(messages=messages)

    # With single message, span equals duration
    assert summary.conversation_span_ms == 2500.0
    assert summary.total_elapsed_time_ms == 2500.0
    assert summary.first_llm_start_time == 100.0
    assert summary.last_llm_end_time == 102.5
