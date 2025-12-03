"""
Type definitions for context compaction.
"""

from enum import StrEnum


class ContextCompactionMode(StrEnum):
    """
    Enumeration of supported context compaction strategies.

    NONE: No compaction - let the context grow until it hits provider limits
    TRUNCATE: Remove older messages to stay within the limit
    SUMMARIZE: Use LLM to create a summary of older messages
    """

    NONE = "none"
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
