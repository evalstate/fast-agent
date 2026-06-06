"""Shared text utilities for command handlers."""

from fast_agent.utils.text import strip_to_none


def truncate_description(description: str, char_limit: int = 240) -> str:
    """Truncate a description intelligently at sentence or word boundaries.

    Args:
        description: The text to truncate.
        char_limit: Maximum character length (default 240).

    Returns:
        The truncated description with "..." appended if truncated.
    """
    if char_limit <= 0:
        return ""
    description = strip_to_none(description) or ""
    if len(description) <= char_limit:
        return description

    truncate_pos = char_limit
    sentence_break = description.rfind(". ", 0, char_limit)
    if sentence_break != -1 and sentence_break > char_limit - 50:
        truncate_pos = sentence_break + 1
    else:
        word_break = description.rfind(" ", 0, char_limit)
        if word_break != -1 and word_break > char_limit - 30:
            truncate_pos = word_break

    return description[:truncate_pos].rstrip() + "..."
