"""Shared text utilities for command handlers."""


def truncate_description(description: str, char_limit: int = 240) -> str:
    """Truncate a description intelligently at sentence or word boundaries.

    Args:
        description: The text to truncate.
        char_limit: Maximum character length (default 240).

    Returns:
        The truncated description with "..." appended if truncated.
    """
    description = description.strip()
    if len(description) <= char_limit:
        return description

    truncate_pos = char_limit
    sentence_break = description.rfind(". ", 0, char_limit + 20)
    if sentence_break > char_limit - 50:
        truncate_pos = sentence_break + 1
    else:
        word_break = description.rfind(" ", 0, char_limit + 10)
        if word_break > char_limit - 30:
            truncate_pos = word_break

    return description[:truncate_pos].rstrip() + "..."
