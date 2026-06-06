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

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final, Literal

from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

SearchScope = Literal["user", "assistant", "tool_calls", "tool_results", "all"]
SEARCH_SCOPES: Final[tuple[SearchScope, ...]] = (
    "user",
    "assistant",
    "tool_calls",
    "tool_results",
    "all",
)


def _validate_scope(scope: SearchScope) -> SearchScope:
    if scope not in SEARCH_SCOPES:
        raise ValueError(f"Unsupported message search scope: {scope}")
    return scope


def _compile_pattern(pattern: str | re.Pattern) -> re.Pattern:
    return re.compile(pattern) if isinstance(pattern, str) else pattern


def search_messages(
    messages: list[PromptMessageExtended],
    pattern: str | re.Pattern,
    scope: SearchScope = "all",
) -> list[PromptMessageExtended]:
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
    scope = _validate_scope(scope)
    compiled_pattern = _compile_pattern(pattern)
    return [
        msg
        for msg in messages
        if _message_contains_pattern(msg, compiled_pattern, scope)
    ]


def find_matches(
    messages: list[PromptMessageExtended],
    pattern: str | re.Pattern,
    scope: SearchScope = "all",
) -> list[tuple[PromptMessageExtended, re.Match]]:
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
    scope = _validate_scope(scope)
    return list(_iter_matches(messages, _compile_pattern(pattern), scope))


def extract_first(
    messages: list[PromptMessageExtended],
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
    scope = _validate_scope(scope)
    for _message, match in _iter_matches(messages, _compile_pattern(pattern), scope):
        return match.group(group)
    return None


def extract_last(
    messages: list[PromptMessageExtended],
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
    scope = _validate_scope(scope)
    last_match: re.Match | None = None
    for _message, match in _iter_matches(messages, _compile_pattern(pattern), scope):
        last_match = match
    if last_match is None:
        return None
    return last_match.group(group)


def _message_contains_pattern(
    msg: PromptMessageExtended,
    pattern: re.Pattern,
    scope: SearchScope,
) -> bool:
    """Check if a message contains the pattern in the specified scope."""
    return any(pattern.search(text) for text in _extract_searchable_text(msg, scope))


def _find_in_message(
    msg: PromptMessageExtended,
    pattern: re.Pattern,
    scope: SearchScope,
) -> list[re.Match]:
    """Find all matches of pattern in a message."""
    return [
        match
        for text in _extract_searchable_text(msg, scope)
        for match in pattern.finditer(text)
    ]


def _iter_matches(
    messages: list[PromptMessageExtended],
    pattern: re.Pattern,
    scope: SearchScope,
):
    for msg in messages:
        for match in _find_in_message(msg, pattern, scope):
            yield msg, match


def _content_texts(contents) -> list[str]:
    return [text for content in contents if (text := get_text(content))]


def _extract_searchable_text(
    msg: PromptMessageExtended,
    scope: SearchScope,
) -> list[str]:
    """Extract text from message based on scope."""
    texts = []

    # User content
    if scope in ("user", "all") and msg.role == "user":
        texts.extend(_content_texts(msg.content))

    # Assistant content
    if scope in ("assistant", "all") and msg.role == "assistant":
        texts.extend(_content_texts(msg.content))

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
            texts.extend(_content_texts(tool_result.content))

    return texts
