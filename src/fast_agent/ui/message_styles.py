from __future__ import annotations

from rich.text import Text


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
    stream_reprint_banner_label = "FINAL RESPONSE"

    def header_line(
        self,
        left: str,
        right: str,
        width: int,
        *,
        rule_fill: bool = False,
    ) -> Text:
        left_text = Text.from_markup(left)
        right_content = right.strip()
        if rule_fill:
            return self._header_line_with_rule_fill(left_text, right_content, width)

        combined = Text()
        combined.append_text(left_text)
        if right_content:
            right_text = Text.from_markup(right_content)
            right_text.stylize("dim")
            combined.append(" ", style="default")
            combined.append_text(right_text)
        return combined

    def _header_line_with_rule_fill(self, left_text: Text, right_content: str, width: int) -> Text:
        combined = Text()
        combined.append_text(left_text)

        if right_content:
            right_text = Text.from_markup(right_content)
            right_text.stylize("dim")
            remaining = width - left_text.cell_len - right_text.cell_len
            if remaining >= 3:
                combined.append(" ", style="default")
                combined.append("─" * (remaining - 2), style="dim")
                combined.append(" ", style="default")
                combined.append_text(right_text)
                return combined
            if remaining >= 1:
                combined.append(" ", style="default")
            combined.append_text(right_text)
            return combined

        remaining = width - left_text.cell_len
        if remaining >= 2:
            combined.append(" ", style="default")
            combined.append("─" * (remaining - 1), style="dim")
        elif remaining == 1:
            combined.append(" ", style="default")
        return combined

    def metadata_line(self, content: Text, width: int) -> Text:  # noqa: ARG002
        line = Text()
        line.append("  ", style="dim")
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

        prefix = Text("▎ ", style="dim")
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

    def shell_exit_line(
        self,
        exit_code: int,
        width: int,
        detail: str | None = None,
    ) -> Text:  # noqa: ARG002
        if exit_code == 0:
            exit_code_style = "white reverse dim"
        elif exit_code == 1:
            exit_code_style = "red reverse dim"
        else:
            exit_code_style = "red reverse bold"

        exit_code_text = f" exit code {exit_code} "
        line = Text()
        line.append("▎", style="dim")
        line.append(exit_code_text, style=exit_code_style)
        if detail:
            line.append(detail, style="dim")
        return line

    def tool_update_line(self, width: int) -> Text:  # noqa: ARG002
        note = Text("tool update", style="dim")
        return self.metadata_line(note, width)

    def stream_reprint_banner(
        self,
        width: int,
        *,
        label: str | None = None,
    ) -> list[Text]:
        banner_width = max(1, width)
        banner_style = "bright_green bold"
        banner_label = (label or self.stream_reprint_banner_label).strip()
        if not banner_label:
            banner_label = self.stream_reprint_banner_label

        top = Text("━" * banner_width, style=banner_style)
        bottom = Text("━" * banner_width, style=banner_style)

        middle = Text()
        label_text = Text(f" {banner_label} ", style="black on bright_green bold")
        if label_text.cell_len >= banner_width:
            label_text.truncate(banner_width)
            middle.append_text(label_text)
            return [top, middle, bottom]

        fill_width = banner_width - label_text.cell_len
        left_fill = fill_width // 2
        right_fill = fill_width - left_fill
        if left_fill:
            middle.append("━" * left_fill, style=banner_style)
        middle.append_text(label_text)
        if right_fill:
            middle.append("━" * right_fill, style=banner_style)
        return [top, middle, bottom]
