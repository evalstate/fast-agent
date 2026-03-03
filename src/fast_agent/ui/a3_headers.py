from __future__ import annotations

from rich.text import Text


def build_a3_section_header(
    title: str,
    *,
    color: str = "blue",
    include_dot: bool = False,
) -> Text:
    """Build a compact A3-style section header line.

    Args:
        title: Header text.
        color: Primary color for bar/title.
        include_dot: Whether to include the dim middle dot after the bar.
    """
    header = Text()
    header.append("▎", style=color)
    if include_dot:
        header.append("●", style=f"dim {color}")
    header.append(" ")
    header.append(title, style=color)
    return header


__all__ = ["build_a3_section_header"]

