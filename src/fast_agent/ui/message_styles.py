from __future__ import annotations

from typing import Final

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from fast_agent.utils.tool_names import normalize_tool_name


def _shorten_items(items: list[str], max_length: int) -> list[str]:
    if max_length <= 0:
        return ["" for _item in items]
    if max_length == 1:
        return ["…" if item else "" for item in items]
    return [item[: max_length - 1] + "…" if len(item) > max_length else item for item in items]


# Indicator shown when jumping to a non-visible highlighted item
_JUMP_INDICATOR = " ▶ "
_SHELL_EXIT_CODE_STYLES = {
    0: "white dim",
    1: "red reverse dim",
}
_SHELL_EXIT_CODE_ERROR_STYLE = "red reverse bold"
SHORT_RULE_WIDTH: Final[int] = 36
STREAM_REPRINT_BANNER_WIDTH: Final[int] = 60
CODE_RULE_EDGE_STYLE: Final[Style] = Style(color="bright_black")
CODE_RULE_PILL_STYLE: Final[Style] = Style(color="bright_black", reverse=True)


def format_code_tool_pill_label(tool_name: str, language: str | None) -> str:
    """Format the inverse pill label for code-variant tool calls."""
    normalized = normalize_tool_name(tool_name)
    base_name = tool_name.rsplit("__", 1)[-1] if "__" in tool_name else tool_name
    clean_lang = (language or "").strip()
    if normalized == "exec_sandbox":
        return f"{clean_lang} (sandbox)" if clean_lang and clean_lang != "text" else "sandbox"
    if clean_lang and clean_lang != "text":
        return f"{clean_lang} ({base_name})"
    return base_name


def build_code_rule(
    label: str = "",
    *,
    max_width: int | None = None,
    prefix: Text | None = None,
    suffix: Text | None = None,
) -> Text:
    """Build a resize-safe short rule line with optional prefix, inverse pill, and metadata."""
    target = SHORT_RULE_WIDTH if max_width is None else min(max_width, SHORT_RULE_WIDTH)
    clean_label = label.strip()
    line = Text()
    used = 0
    if prefix is not None:
        line.append_text(prefix)
        used += prefix.cell_len
    elif clean_label:
        line.append("──", style=CODE_RULE_EDGE_STYLE)
        used += 2

    if clean_label:
        pill = f" {clean_label} "
        line.append(pill, style=CODE_RULE_PILL_STYLE)
        used += cell_len(pill)

    has_suffix = suffix is not None and bool(suffix.plain)
    if has_suffix and suffix is not None:
        if clean_label:
            line.append(" ")
            used += 1
        line.append_text(suffix)
        used += suffix.cell_len

    remaining = target - used
    if has_suffix:
        if remaining >= 3:
            line.append(" ")
            line.append("─" * (remaining - 1), style=CODE_RULE_EDGE_STYLE)
    elif remaining >= 2:
        line.append("─" * remaining, style=CODE_RULE_EDGE_STYLE)
    elif clean_label and (max_width is None or used + 2 <= max_width):
        line.append("──", style=CODE_RULE_EDGE_STYLE)
    elif prefix is None and not clean_label and target > 0:
        line.append("─" * target, style=CODE_RULE_EDGE_STYLE)

    return line


def _format_bottom_metadata_compact(
    items: list[str],
    highlight_indexes: list[int],
    highlight_color: str,
    max_width: int | None = None,
) -> Text:
    """Format items with ' • ' separator, showing highlighted item even if truncated."""
    separator = " • "
    default_style = "white dim"
    return _format_items_with_highlight_jump(
        items=items,
        highlight_indexes=highlight_indexes,
        highlight_color=highlight_color,
        default_style=default_style,
        separator=separator,
        max_width=max_width,
    )


def _format_items_with_highlight_jump(
    items: list[str],
    highlight_indexes: list[int],
    highlight_color: str,
    default_style: str,
    separator: str,
    max_width: int | None,
) -> Text:
    """
    Format a list of items, ensuring highlighted items are visible when space permits.

    If a highlighted item would be truncated in normal rendering, we:
    1. Truncate the visible items earlier to leave space
    2. Add a dim "▶" jump indicator
    3. Show the highlighted item at the end
    """
    if not items:
        return Text()

    valid_highlights = sorted({index for index in highlight_indexes if 0 <= index < len(items)})

    if not valid_highlights:
        return _render_items_normal(
            items=items,
            highlight_indexes=[],
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

    hidden_highlights = [index for index in valid_highlights if index >= visible_count]

    if not hidden_highlights:
        return _render_items_normal(
            items=items,
            highlight_indexes=valid_highlights,
            highlight_color=highlight_color,
            default_style=default_style,
            separator=separator,
            max_width=max_width,
        )

    return _render_items_with_jump(
        items=items,
        highlight_indexes=valid_highlights,
        jump_index=hidden_highlights[0],
        highlight_color=highlight_color,
        default_style=default_style,
        separator=separator,
        max_width=max_width,
    )


def _render_items_normal(
    items: list[str],
    highlight_indexes: list[int],
    highlight_color: str,
    default_style: str,
    separator: str,
    max_width: int | None,
) -> Text:
    """Render items normally, truncating with ellipsis when space runs out."""
    formatted = Text()

    for i, item in enumerate(items):
        sep = Text(separator, style="dim") if i > 0 else Text("")
        should_highlight = i in highlight_indexes
        item_style = highlight_color if should_highlight else default_style
        item_text = Text(item, style=item_style)

        if (
            max_width is not None
            and formatted.cell_len + sep.cell_len + item_text.cell_len > max_width
        ):
            # Doesn't fit - add ellipsis and stop
            if formatted.cell_len == 0 and max_width > 1:
                formatted.append("…", style="dim")
            elif formatted.cell_len < max_width:
                formatted.append(" …", style="dim")
            break

        _append_item_segment(formatted, sep, item_text)

    return formatted


def _append_item_segment(formatted: Text, separator: Text, item_text: Text) -> None:
    if separator.plain:
        formatted.append_text(separator)
    formatted.append_text(item_text)


def _render_items_with_jump(
    items: list[str],
    highlight_indexes: list[int],
    jump_index: int,
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
    highlight_item = items[jump_index]
    highlight_text = Text(highlight_item, style=highlight_color)
    use_jump_indicator = jump_index > 0
    jump_indicator = Text(_JUMP_INDICATOR, style="dim") if use_jump_indicator else Text("")

    # Calculate space needed for jump indicator + highlighted item
    reserved_space = highlight_text.cell_len + (
        jump_indicator.cell_len if use_jump_indicator else 0
    )

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

        _append_prefix_items(
            formatted,
            items[:jump_index],
            highlight_indexes=highlight_indexes,
            highlight_color=highlight_color,
            default_style=default_style,
            separator=separator,
            max_width=available_for_prefix,
        )

    else:
        _append_prefix_items(
            formatted,
            items[:jump_index],
            highlight_indexes=highlight_indexes,
            highlight_color=highlight_color,
            default_style=default_style,
            separator=separator,
            max_width=None,
        )

    # Add jump indicator and highlighted item
    if use_jump_indicator:
        formatted.append_text(jump_indicator)
    formatted.append_text(highlight_text)

    return formatted


def _append_prefix_items(
    formatted: Text,
    items: list[str],
    *,
    highlight_indexes: list[int],
    highlight_color: str,
    default_style: str,
    separator: str,
    max_width: int | None,
) -> None:
    for index, item in enumerate(items):
        sep = Text(separator, style="dim") if index > 0 else Text("")
        item_style = highlight_color if index in highlight_indexes else default_style
        item_text = Text(item, style=item_style)
        if (
            max_width is not None
            and formatted.cell_len + sep.cell_len + item_text.cell_len > max_width
        ):
            return
        _append_item_segment(formatted, sep, item_text)


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
        target_width = min(width, SHORT_RULE_WIDTH)

        if right_content:
            right_text = Text.from_markup(right_content)
            right_text.stylize("dim")
            remaining = target_width - left_text.cell_len - right_text.cell_len
            if remaining >= 3:
                combined.append(" ", style="default")
                combined.append("─" * (remaining - 2), style=CODE_RULE_EDGE_STYLE)
                combined.append(" ", style="default")
                combined.append_text(right_text)
                return combined
            if width - left_text.cell_len - right_text.cell_len >= 1:
                combined.append(" ", style="default")
            combined.append_text(right_text)
            return combined

        remaining = target_width - left_text.cell_len
        if remaining >= 2:
            combined.append(" ", style="default")
            combined.append("─" * (remaining - 1), style=CODE_RULE_EDGE_STYLE)
        return combined

    def metadata_line(self, content: Text) -> Text:
        line = Text()
        line.append("  ", style="dim")
        line.append_text(content)
        return line

    def bottom_metadata_line(
        self,
        items: list[str] | None,
        highlight_indexes: list[int],
        highlight_color: str,
        max_item_length: int | None,
        width: int,
    ) -> Text | None:
        if not items:
            return None

        display_items = items
        if max_item_length is not None:
            display_items = _shorten_items(items, max_item_length)

        prefix = Text("▎ ", style="dim")
        available = max(0, width - prefix.cell_len)

        metadata_text = _format_bottom_metadata_compact(
            display_items,
            highlight_indexes,
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
        detail: str | None = None,
    ) -> Text:
        exit_code_style = _SHELL_EXIT_CODE_STYLES.get(exit_code, _SHELL_EXIT_CODE_ERROR_STYLE)

        line = Text()
        line.append("▎· ", style="dim")
        if exit_code == 0:
            line.append(f"exit code {exit_code}", style=exit_code_style)
            if detail and (clean_detail := detail.lstrip()):
                line.append(f" · {clean_detail}", style="dim")
        else:
            line.append(f" exit code {exit_code} ", style=exit_code_style)
            if detail:
                line.append(detail, style="dim")
        return line

    def tool_update_line(self) -> Text:
        note = Text("tool update", style="dim")
        return self.metadata_line(note)

    def stream_reprint_banner(
        self,
        width: int,
        *,
        label: str | None = None,
    ) -> list[Text]:
        banner_width = max(1, min(width, STREAM_REPRINT_BANNER_WIDTH))
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
