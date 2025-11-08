"""
Utilities for searching and extracting content from message histories.

This module provides functions to search through PromptMessageExtended lists
for content matching patterns, with filtering by message role and content type.

Search Scopes:
--------------
- "user": Searches in user message content blocks (text content only)
- "assistant": Searches in assistant message content blocks (text content only)
- "tool_calls": Searches in tool call names AND stringified arguments
- "tool_results": Searches in tool result content blocks (text content)
- "all": Searches all of the above (default)

Note: The search looks at text content extracted with get_text(), not raw ContentBlock objects.
"""

import re
from typing import List, Literal, Tuple

from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

SearchScope = Literal["user", "assistant", "tool_calls", "tool_results", "all"]


def search_messages(
    messages: List[PromptMessageExtended],
    pattern: str | re.Pattern,
    scope: SearchScope = "all",
) -> List[PromptMessageExtended]:
    """
    Find messages containing content that matches a pattern.

    Args:
        messages: List of messages to search
        pattern: String or compiled regex pattern to search for
        scope: Where to search - "user", "assistant", "tool_calls", "tool_results", or "all"

    Returns:
        List of messages that contain at least one match

    Example:
        ```python
        # Find messages with error content
        error_messages = search_messages(
            agent.message_history,
            r"error|failed",
            scope="tool_results"
        )
        ```
    """
    compiled_pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
    matching_messages = []

    for msg in messages:
        if _message_contains_pattern(msg, compiled_pattern, scope):
            matching_messages.append(msg)

    return matching_messages


def find_matches(
    messages: List[PromptMessageExtended],
    pattern: str | re.Pattern,
    scope: SearchScope = "all",
) -> List[Tuple[PromptMessageExtended, re.Match]]:
    """
    Find all pattern matches in messages, returning match objects.

    This is useful when you need access to match groups or match positions.

    Args:
        messages: List of messages to search
        pattern: String or compiled regex pattern to search for
        scope: Where to search - "user", "assistant", "tool_calls", "tool_results", or "all"

    Returns:
        List of (message, match) tuples for each match found

    Example:
        ```python
        # Extract job IDs with capture groups
        matches = find_matches(
            agent.message_history,
            r"Job started: ([a-f0-9]+)",
            scope="tool_results"
        )
        for msg, match in matches:
            job_id = match.group(1)
            print(f"Found job: {job_id}")
        ```
    """
    compiled_pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
    results = []

    for msg in messages:
        matches = _find_in_message(msg, compiled_pattern, scope)
        for match in matches:
            results.append((msg, match))

    return results


def extract_first(
    messages: List[PromptMessageExtended],
    pattern: str | re.Pattern,
    scope: SearchScope = "all",
    group: int = 0,
) -> str | None:
    """
    Extract the first match from messages.

    This is a convenience function for the common case of extracting a single value.

    Args:
        messages: List of messages to search
        pattern: String or compiled regex pattern to search for
        scope: Where to search - "user", "assistant", "tool_calls", "tool_results", or "all"
        group: Regex group to extract (0 = whole match, 1+ = capture groups)

    Returns:
        Extracted string or None if no match found

    Example:
        ```python
        # Extract job ID in one line
        job_id = extract_first(
            agent.message_history,
            r"Job started: ([a-f0-9]+)",
            scope="tool_results",
            group=1
        )
        ```
    """
    matches = find_matches(messages, pattern, scope)
    if not matches:
        return None

    _, match = matches[0]
    return match.group(group)


def extract_last(
    messages: List[PromptMessageExtended],
    pattern: str | re.Pattern,
    scope: SearchScope = "all",
    group: int = 0,
) -> str | None:
    """
    Extract the last match from messages.

    This is useful when you want the most recent occurrence of a pattern,
    such as the final status update or most recent job ID.

    Args:
        messages: List of messages to search
        pattern: String or compiled regex pattern to search for
        scope: Where to search - "user", "assistant", "tool_calls", "tool_results", or "all"
        group: Regex group to extract (0 = whole match, 1+ = capture groups)

    Returns:
        Extracted string or None if no match found

    Example:
        ```python
        # Extract the most recent status update
        final_status = extract_last(
            agent.message_history,
            r"Status: (\\w+)",
            scope="tool_results",
            group=1
        )
        ```
    """
    matches = find_matches(messages, pattern, scope)
    if not matches:
        return None

    _, match = matches[-1]
    return match.group(group)


def _message_contains_pattern(
    msg: PromptMessageExtended,
    pattern: re.Pattern,
    scope: SearchScope,
) -> bool:
    """Check if a message contains the pattern in the specified scope."""
    texts = _extract_searchable_text(msg, scope)
    for text in texts:
        if pattern.search(text):
            return True
    return False


def _find_in_message(
    msg: PromptMessageExtended,
    pattern: re.Pattern,
    scope: SearchScope,
) -> List[re.Match]:
    """Find all matches of pattern in a message."""
    texts = _extract_searchable_text(msg, scope)
    matches = []
    for text in texts:
        for match in pattern.finditer(text):
            matches.append(match)
    return matches


def _extract_searchable_text(
    msg: PromptMessageExtended,
    scope: SearchScope,
) -> List[str]:
    """Extract text from message based on scope."""
    texts = []

    # User content
    if scope in ("user", "all") and msg.role == "user":
        for content in msg.content:
            text = get_text(content)
            if text:
                texts.append(text)

    # Assistant content
    if scope in ("assistant", "all") and msg.role == "assistant":
        for content in msg.content:
            text = get_text(content)
            if text:
                texts.append(text)

    # Tool calls (search in tool names and serialized arguments)
    if scope in ("tool_calls", "all") and msg.tool_calls:
        for tool_call in msg.tool_calls.values():
            # Add tool name
            texts.append(tool_call.params.name)
            # Add stringified arguments
            if tool_call.params.arguments:
                texts.append(str(tool_call.params.arguments))

    # Tool results
    if scope in ("tool_results", "all") and msg.tool_results:
        for tool_result in msg.tool_results.values():
            for content in tool_result.content:
                text = get_text(content)
                if text:
                    texts.append(text)

    return texts
