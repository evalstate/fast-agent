from __future__ import annotations

from typing import Protocol

from rich.text import Text


class MessageStyle(Protocol):
    name: str
    header_spacing_after: int
    shell_exit_spacing_after: int
    bottom_metadata_requires_highlight: bool

    def header_line(self, left: str, right: str, width: int) -> Text: ...

    def bottom_metadata_line(
        self,
        items: list[str] | None,
        highlight_index: int | None,
        highlight_color: str,
        max_item_length: int | None,
        width: int,
    ) -> Text | None: ...

    def metadata_line(self, content: Text, width: int) -> Text: ...

    def shell_exit_line(self, exit_code: int, width: int) -> Text: ...

    def tool_update_line(self, width: int) -> Text: ...


def resolve_message_style(style_name: str | None) -> MessageStyle:
    if style_name and style_name.lower() == "classic":
        return ClassicMessageStyle()
    return A3MessageStyle()


def _shorten_items(items: list[str], max_length: int) -> list[str]:
    return [item[: max_length - 1] + "…" if len(item) > max_length else item for item in items]


# Indicator shown when jumping to a non-visible highlighted item
_JUMP_INDICATOR = " ▶ "


def _format_bottom_metadata_compact(
    items: list[str],
    highlight_index: int | None,
    highlight_color: str,
    max_width: int | None = None,
) -> Text:
    """Format items with ' • ' separator, showing highlighted item even if truncated."""
    separator = " • "
    default_style = "white dim"
    return _format_items_with_highlight_jump(
        items=items,
        highlight_index=highlight_index,
        highlight_color=highlight_color,
        default_style=default_style,
        separator=separator,
        max_width=max_width,
    )


def _format_bottom_metadata(
    items: list[str],
    highlight_index: int | None,
    highlight_color: str,
    max_width: int | None = None,
) -> Text:
    """Format items with ' | ' separator, showing highlighted item even if truncated."""
    separator = " | "
    default_style = "dim"
    return _format_items_with_highlight_jump(
        items=items,
        highlight_index=highlight_index,
        highlight_color=highlight_color,
        default_style=default_style,
        separator=separator,
        max_width=max_width,
    )


def _format_items_with_highlight_jump(
    items: list[str],
    highlight_index: int | None,
    highlight_color: str,
    default_style: str,
    separator: str,
    max_width: int | None,
) -> Text:
    """
    Format a list of items, ensuring the highlighted item is always visible.

    If the highlighted item would be truncated in normal rendering, we:
    1. Truncate the visible items earlier to leave space
    2. Add a dim "▶" jump indicator
    3. Show the highlighted item at the end
    """
    if not items:
        return Text()

    highlight_idx: int | None = None
    if highlight_index is not None and 0 <= highlight_index < len(items):
        highlight_idx = highlight_index

    if highlight_idx is None:
        return _render_items_normal(
            items=items,
            highlight_index=None,
            highlight_color=highlight_color,
            default_style=default_style,
            separator=separator,
            max_width=max_width,
        )

    separator_width = Text(separator).cell_len if separator else 0
    visible_count = 0
    current_len = 0

    for i, item in enumerate(items):
        sep_len = separator_width if i > 0 else 0
        item_len = Text(item).cell_len
        segment_len = sep_len + item_len

        if max_width is not None and current_len + segment_len > max_width:
            break

        current_len += segment_len
        visible_count += 1

    highlight_visible = highlight_idx < visible_count

    if highlight_visible:
        # Normal rendering - highlight is visible or no highlight
        return _render_items_normal(
            items=items,
            highlight_index=highlight_idx,
            highlight_color=highlight_color,
            default_style=default_style,
            separator=separator,
            max_width=max_width,
        )
    else:
        # Jump rendering - highlight would be truncated
        return _render_items_with_jump(
            items=items,
            highlight_index=highlight_idx,
            highlight_color=highlight_color,
            default_style=default_style,
            separator=separator,
            max_width=max_width,
        )


def _render_items_normal(
    items: list[str],
    highlight_index: int | None,
    highlight_color: str,
    default_style: str,
    separator: str,
    max_width: int | None,
) -> Text:
    """Render items normally, truncating with ellipsis when space runs out."""
    formatted = Text()

    for i, item in enumerate(items):
        sep = Text(separator, style="dim") if i > 0 else Text("")
        should_highlight = highlight_index is not None and i == highlight_index
        item_style = highlight_color if should_highlight else default_style
        item_text = Text(item, style=item_style)

        if max_width is not None:
            if formatted.cell_len + sep.cell_len + item_text.cell_len > max_width:
                # Doesn't fit - add ellipsis and stop
                if formatted.cell_len == 0 and max_width > 1:
                    formatted.append("…", style="dim")
                elif formatted.cell_len < max_width:
                    formatted.append(" …", style="dim")
                break

        if sep.plain:
            formatted.append_text(sep)
        formatted.append_text(item_text)

    return formatted


def _render_items_with_jump(
    items: list[str],
    highlight_index: int,
    highlight_color: str,
    default_style: str,
    separator: str,
    max_width: int | None,
) -> Text:
    """
    Render items with a jump to the highlighted item.

    Truncates visible items early, shows a dim "▶" indicator, then the highlighted item.
    """
    formatted = Text()

    # Build the highlighted item segment
    highlight_item = items[highlight_index]
    highlight_text = Text(highlight_item, style=highlight_color)
    use_jump_indicator = highlight_index > 0
    jump_indicator = Text(_JUMP_INDICATOR, style="dim") if use_jump_indicator else Text("")

    # Calculate space needed for jump indicator + highlighted item
    reserved_space = highlight_text.cell_len + (jump_indicator.cell_len if use_jump_indicator else 0)

    if max_width is not None:
        available_for_prefix = max_width - reserved_space

        if available_for_prefix <= 0:
            # No space for prefix items, just show jump + highlight (or just highlight)
            if max_width >= highlight_text.cell_len:
                if use_jump_indicator and max_width >= reserved_space:
                    formatted.append_text(jump_indicator)
                formatted.append_text(highlight_text)
            else:
                # Not even enough for the highlight item itself
                formatted.append("…", style="dim")
            return formatted

        # Render prefix items that fit within available_for_prefix
        for i, item in enumerate(items):
            if i >= highlight_index:
                break

            sep = Text(separator, style="dim") if i > 0 else Text("")
            item_text = Text(item, style=default_style)

            if formatted.cell_len + sep.cell_len + item_text.cell_len > available_for_prefix:
                break

            if sep.plain:
                formatted.append_text(sep)
            formatted.append_text(item_text)

    else:
        # No width limit - just render all items before highlight
        for i, item in enumerate(items):
            if i >= highlight_index:
                break

            sep = Text(separator, style="dim") if i > 0 else Text("")
            item_text = Text(item, style=default_style)

            if sep.plain:
                formatted.append_text(sep)
            formatted.append_text(item_text)

    # Add jump indicator and highlighted item
    if use_jump_indicator:
        formatted.append_text(jump_indicator)
    formatted.append_text(highlight_text)

    return formatted


class A3MessageStyle:
    name = "a3"
    header_spacing_after = 0
    shell_exit_spacing_after = 1
    bottom_metadata_requires_highlight = True

    def header_line(self, left: str, right: str, width: int) -> Text:  # noqa: ARG002
        left_text = Text.from_markup(left)
        right_content = right.strip()
        combined = Text()
        combined.append_text(left_text)
        if right_content:
            right_text = Text.from_markup(right_content)
            right_text.stylize("dim")
            combined.append(" ", style="default")
            combined.append_text(right_text)
        return combined

    def metadata_line(self, content: Text, width: int) -> Text:  # noqa: ARG002
        line = Text()
        line.append("▎• ", style="dim")
        line.append_text(content)
        return line

    def bottom_metadata_line(
        self,
        items: list[str] | None,
        highlight_index: int | None,
        highlight_color: str,
        max_item_length: int | None,
        width: int,
    ) -> Text | None:
        if not items:
            return None

        display_items = items
        if max_item_length:
            display_items = _shorten_items(items, max_item_length)

        prefix = Text("▎• ", style="dim")
        available = max(0, width - prefix.cell_len)

        metadata_text = _format_bottom_metadata_compact(
            display_items,
            highlight_index,
            highlight_color,
            max_width=available,
        )

        line = Text()
        line.append_text(prefix)
        line.append_text(metadata_text)
        return line

    def shell_exit_line(self, exit_code: int, width: int) -> Text:  # noqa: ARG002
        if exit_code == 0:
            exit_code_style = "white reverse dim"
        elif exit_code == 1:
            exit_code_style = "red reverse dim"
        else:
            exit_code_style = "red reverse bold"

        exit_code_text = f" exit code {exit_code} "
        line = Text()
        line.append("▎• ", style="dim")
        line.append(exit_code_text, style=exit_code_style)
        return line

    def tool_update_line(self, width: int) -> Text:  # noqa: ARG002
        note = Text("tool update", style="dim")
        return self.metadata_line(note, width)


class ClassicMessageStyle:
    name = "classic"
    header_spacing_after = 1
    shell_exit_spacing_after = 0
    bottom_metadata_requires_highlight = False

    def header_line(self, left: str, right: str, width: int) -> Text:
        left_text = Text.from_markup(left)
        right_content = right.strip()

        if right_content:
            right_text = Text()
            right_text.append("[", style="dim")
            right_text.append_text(Text.from_markup(right_content))
            right_text.append("]", style="dim")
            separator_count = width - left_text.cell_len - right_text.cell_len
            if separator_count < 1:
                separator_count = 1
        else:
            right_text = Text("")
            separator_count = width - left_text.cell_len

        combined = Text()
        combined.append_text(left_text)
        combined.append(" ", style="default")
        combined.append("─" * (separator_count - 1), style="dim")
        combined.append_text(right_text)
        return combined

    def metadata_line(self, content: Text, width: int) -> Text:
        prefix = Text("─| ")
        prefix.stylize("dim")
        suffix = Text(" |")
        suffix.stylize("dim")

        line = Text()
        line.append_text(prefix)
        line.append_text(content)
        line.append_text(suffix)

        remaining = width - line.cell_len
        if remaining > 0:
            line.append("─" * remaining, style="dim")
        return line

    def bottom_metadata_line(
        self,
        items: list[str] | None,
        highlight_index: int | None,
        highlight_color: str,
        max_item_length: int | None,
        width: int,
    ) -> Text | None:
        if not items:
            return Text("─" * width, style="dim")

        display_items = items
        if max_item_length:
            display_items = _shorten_items(items, max_item_length)

        prefix = Text("─| ")
        prefix.stylize("dim")
        suffix = Text(" |")
        suffix.stylize("dim")
        available = max(0, width - prefix.cell_len - suffix.cell_len)

        metadata_text = _format_bottom_metadata(
            display_items,
            highlight_index,
            highlight_color,
            max_width=available,
        )

        line = Text()
        line.append_text(prefix)
        line.append_text(metadata_text)
        line.append_text(suffix)

        remaining = width - line.cell_len
        if remaining > 0:
            line.append("─" * remaining, style="dim")
        return line

    def tool_update_line(self, width: int) -> Text:
        return Text("─" * width, style="dim")

    def shell_exit_line(self, exit_code: int, width: int) -> Text:
        if exit_code == 0:
            exit_code_style = "white reverse dim"
        elif exit_code == 1:
            exit_code_style = "red reverse dim"
        else:
            exit_code_style = "red reverse bold"

        exit_code_text = f" exit code {exit_code} "
        exit_text = Text(exit_code_text, style=exit_code_style)
        return self.metadata_line(exit_text, width)
