"""Tests for message search utilities"""

import re

from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from fast_agent.types import PromptMessageExtended, extract_first, find_matches, search_messages


def test_search_messages_user_scope():
    """Test searching in user messages only"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Hello, I need help with errors")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="There was an error in processing")],
        ),
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Thank you for the help")],
        ),
    ]

    results = search_messages(messages, "error", scope="user")
    assert len(results) == 1
    assert results[0].role == "user"
    assert "errors" in results[0].content[0].text


def test_search_messages_assistant_scope():
    """Test searching in assistant messages only"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Hello, I need help with errors")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="There was an error in processing")],
        ),
    ]

    results = search_messages(messages, "error", scope="assistant")
    assert len(results) == 1
    assert results[0].role == "assistant"


def test_search_messages_tool_results_scope():
    """Test searching in tool results"""
    messages = [
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Job started: abc123def")],
                    isError=False,
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_2": CallToolResult(
                    content=[TextContent(type="text", text="Task completed successfully")],
                    isError=False,
                ),
            },
        ),
    ]

    results = search_messages(messages, r"Job started:", scope="tool_results")
    assert len(results) == 1
    assert results[0].tool_results["call_1"].content[0].text == "Job started: abc123def"


def test_search_messages_tool_calls_scope():
    """Test searching in tool calls"""
    messages = [
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call_1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="create_job", arguments={"job_type": "processing"}
                    ),
                ),
            },
        ),
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call_2": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="check_status", arguments={}),
                ),
            },
        ),
    ]

    # Search for tool name
    results = search_messages(messages, "create_job", scope="tool_calls")
    assert len(results) == 1
    assert "call_1" in results[0].tool_calls

    # Search in arguments
    results = search_messages(messages, "processing", scope="tool_calls")
    assert len(results) == 1


def test_search_messages_all_scope():
    """Test searching across all message types"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Find the error")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="Searching for error...")],
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="error: Not found")],
                    isError=True,
                ),
            },
        ),
    ]

    results = search_messages(messages, "error", scope="all")
    assert len(results) == 3  # All messages contain "error"


def test_search_messages_regex_pattern():
    """Test searching with regex patterns"""
    messages = [
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Job started: abc123")],
                    isError=False,
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_2": CallToolResult(
                    content=[TextContent(type="text", text="Job started: def456")],
                    isError=False,
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_3": CallToolResult(
                    content=[TextContent(type="text", text="Task completed")],
                    isError=False,
                ),
            },
        ),
    ]

    results = search_messages(messages, r"Job started: [a-z0-9]+", scope="tool_results")
    assert len(results) == 2


def test_search_messages_compiled_pattern():
    """Test searching with pre-compiled regex"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Error occurred")],
        ),
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="error in processing")],
        ),
    ]

    # Case insensitive search with compiled pattern
    pattern = re.compile(r"error", re.IGNORECASE)
    results = search_messages(messages, pattern, scope="user")
    assert len(results) == 2


def test_find_matches():
    """Test find_matches returns match objects"""
    messages = [
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Job started: abc123def")],
                    isError=False,
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_2": CallToolResult(
                    content=[TextContent(type="text", text="Job started: xyz789ghi")],
                    isError=False,
                ),
            },
        ),
    ]

    matches = find_matches(messages, r"Job started: ([a-z0-9]+)", scope="tool_results")
    assert len(matches) == 2

    # First match
    msg1, match1 = matches[0]
    assert msg1.tool_results is not None
    assert match1.group(0) == "Job started: abc123def"
    assert match1.group(1) == "abc123def"

    # Second match
    msg2, match2 = matches[1]
    assert match2.group(1) == "xyz789ghi"


def test_find_matches_multiple_in_same_message():
    """Test find_matches when a single message has multiple matches"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[
                TextContent(
                    type="text", text="Found error at line 10. Another error at line 20."
                )
            ],
        ),
    ]

    matches = find_matches(messages, r"error at line \d+", scope="user")
    assert len(matches) == 2
    assert matches[0][1].group(0) == "error at line 10"
    assert matches[1][1].group(0) == "error at line 20"


def test_extract_first_basic():
    """Test extract_first with basic pattern"""
    messages = [
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Job started: abc123def")],
                    isError=False,
                ),
            },
        ),
    ]

    # Extract whole match (group 0)
    result = extract_first(messages, r"Job started: [a-z0-9]+", scope="tool_results")
    assert result == "Job started: abc123def"


def test_extract_first_with_capture_group():
    """Test extract_first with capture groups"""
    messages = [
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Job started: abc123def")],
                    isError=False,
                ),
            },
        ),
    ]

    # Extract first capture group
    result = extract_first(messages, r"Job started: ([a-z0-9]+)", scope="tool_results", group=1)
    assert result == "abc123def"


def test_extract_first_no_match():
    """Test extract_first when no match is found"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Hello world")],
        ),
    ]

    result = extract_first(messages, r"Job started:", scope="user")
    assert result is None


def test_extract_first_multiple_messages():
    """Test extract_first returns first match across multiple messages"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="No match here")],
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_1": CallToolResult(
                    content=[TextContent(type="text", text="Job started: first123")],
                    isError=False,
                ),
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={
                "call_2": CallToolResult(
                    content=[TextContent(type="text", text="Job started: second456")],
                    isError=False,
                ),
            },
        ),
    ]

    result = extract_first(messages, r"Job started: ([a-z0-9]+)", scope="tool_results", group=1)
    assert result == "first123"  # Returns first match


def test_empty_messages_list():
    """Test search functions with empty message list"""
    messages = []

    assert search_messages(messages, "test") == []
    assert find_matches(messages, "test") == []
    assert extract_first(messages, "test") is None


def test_search_messages_no_matches():
    """Test search_messages when pattern doesn't match"""
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Hello world")],
        ),
    ]

    results = search_messages(messages, "goodbye", scope="user")
    assert results == []
