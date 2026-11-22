"""Interactive picker for PTC function definitions (``/tools select``).

The picker lists the functions callable from the PTC code tool, deferred (searchable) ones first,
and returns the Python names the user chose; the caller turns them into ``^tool:`` mentions.
Selection logic lives in ``ToolPickerState`` so it can be tested without a terminal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.widgets import Frame

from fast_agent.commands.tool_summaries import format_token_estimate
from fast_agent.mcp.common import SEP
from fast_agent.ptc.catalog import LOCAL_SOURCE
from fast_agent.ptc.mentions import tool_mention_token
from fast_agent.ui.picker_theme import build_picker_style
from fast_agent.utils.count_display import format_count

if TYPE_CHECKING:
    from prompt_toolkit.formatted_text import StyleAndTextTuples
    from prompt_toolkit.key_binding import KeyPressEvent

    from fast_agent.tools.tool_surface import ToolSurface

ALL_SOURCES_LABEL = "all"
"""Label of the source group that shows every function (``None``); always the first group."""
_PAGE = 10
_MIN_ROWS = 10
_MAX_ROWS = 18
_CHROME_ROWS = 7
"""Rows used by the header, frame borders and footer around the list."""
_KEY_HINTS = " Type to search · [-] already seen · Space toggle · Tab source · Enter add · Esc"
_WORD_SEPARATORS = re.compile(r"[\s_/-]+")


@dataclass(frozen=True, slots=True)
class ToolPickerItem:
    name: str
    """Python name, as used in ``^tool:<name>`` mentions."""
    tool_name: str
    source: str
    description: str
    stub: str
    tokens: int
    deferred: bool
    seen: bool

    @property
    def row_name(self) -> str:
        """Show the tool's local name when its source already identifies the namespace."""
        prefix = f"{self.source}{SEP}"
        return (
            self.tool_name.removeprefix(prefix) if self.tool_name.startswith(prefix) else self.name
        )

    def matches(self, query: str) -> bool:
        """Whether the filter ``query`` keeps this item.

        A substring of any field matches; otherwise every term of the query (split on
        whitespace, ``_``, ``-`` and ``/``) must start a word, so ``repo/files`` and
        ``repo_fi`` find ``list_repo_files``.
        """
        needle = query.casefold()
        fields = tuple(
            value.casefold() for value in (self.name, self.tool_name, self.source, self.description)
        )
        if any(needle in value for value in fields):
            return True
        terms = [term for term in _WORD_SEPARATORS.split(needle) if term]
        words = [word for value in fields for word in _WORD_SEPARATORS.split(value) if word]
        return bool(terms) and all(any(word.startswith(term) for word in words) for term in terms)


def tool_picker_items(surface: ToolSurface) -> tuple[ToolPickerItem, ...]:
    """Programmatic functions of ``surface``, deferred first, then grouped by source."""
    items = [
        ToolPickerItem(
            name=function.python_name or function.tool.name,
            tool_name=function.tool.name,
            source=function.source or LOCAL_SOURCE,
            description=(function.tool.description or "").strip(),
            stub=function.stub or function.signature or function.tool.name,
            tokens=function.stub_tokens,
            deferred=function.deferred,
            seen=function.seen,
        )
        for function in surface.programmatic_functions()
    ]
    return tuple(sorted(items, key=lambda item: (not item.deferred, item.source.casefold())))


@dataclass(slots=True)
class ToolPickerState:
    """Filter text, source group, highlighted row and multi-selection."""

    items: tuple[ToolPickerItem, ...]
    query: str = ""
    source_index: int = 0
    cursor: int = 0
    selected: dict[str, ToolPickerItem] = field(default_factory=dict)
    """Chosen items by name, in the order they were chosen."""

    @property
    def sources(self) -> tuple[str | None, ...]:
        """Source groups; ``None`` (all sources) comes first."""
        return (None, *dict.fromkeys(item.source for item in self.items))

    @property
    def source(self) -> str | None:
        return self.sources[self.source_index]

    @property
    def visible(self) -> tuple[ToolPickerItem, ...]:
        source = self.source
        return tuple(
            item
            for item in self.items
            if (source is None or item.source == source)
            and (not self.query or item.matches(self.query))
        )

    @property
    def current(self) -> ToolPickerItem | None:
        visible = self.visible
        return visible[self.cursor] if self.cursor < len(visible) else None

    @property
    def selected_tokens(self) -> int:
        return sum(item.tokens for item in self.selected.values())

    def type_text(self, text: str) -> None:
        self.query += text
        self.cursor = 0

    def backspace(self) -> None:
        self.query = self.query[:-1]
        self.cursor = 0

    def clear_query(self) -> None:
        self.query = ""
        self.cursor = 0

    def cycle_source(self, delta: int) -> None:
        self.source_index = (self.source_index + delta) % len(self.sources)
        self.cursor = 0

    def move(self, delta: int) -> None:
        count = len(self.visible)
        self.cursor = min(max(self.cursor + delta, 0), max(count - 1, 0))

    def move_to(self, index: int) -> None:
        self.cursor = 0
        self.move(index)

    def toggle(self) -> None:
        item = self.current
        if item is None:
            return
        if self.selected.pop(item.name, None) is None:
            self.selected[item.name] = item

    def chosen_names(self) -> tuple[str, ...]:
        """Names to add on Enter: the selection, or the highlighted item when none is chosen."""
        if self.selected:
            return tuple(self.selected)
        current = self.current
        return (current.name,) if current is not None else ()

    def footer_text(self) -> str:
        count = format_count(len(self.selected), "function")
        if not self.selected:
            return f"Selected: {count}"
        return f"Selected: {count} ({format_token_estimate(self.selected_tokens)})"


class _ToolPicker:
    def __init__(self, items: tuple[ToolPickerItem, ...], *, tool_name: str) -> None:
        self.state = ToolPickerState(items)
        self.tool_name = tool_name
        self.list_control = FormattedTextControl(
            self._render_rows,
            focusable=True,
            show_cursor=False,
            get_cursor_position=lambda: Point(x=0, y=self.state.cursor),
        )
        self.list_window = Window(
            self.list_control,
            wrap_lines=False,
            height=self._list_height,
            dont_extend_height=True,
            ignore_content_width=True,
            always_hide_cursor=True,
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )
        preview_window = Window(
            FormattedTextControl(self._render_preview),
            wrap_lines=True,
            height=self._list_height,
            dont_extend_height=True,
            ignore_content_width=True,
        )
        body = HSplit(
            [
                Window(FormattedTextControl(self._render_header), height=1),
                VSplit(
                    [
                        Frame(self.list_window, title=self._list_title, width=Dimension(weight=2)),
                        Frame(preview_window, title="Definition", width=Dimension(weight=3)),
                    ]
                ),
                Window(FormattedTextControl(self._render_footer), height=2),
            ]
        )
        self.app: Application[tuple[str, ...] | None] = Application(
            layout=Layout(body, focused_element=self.list_window),
            key_bindings=self._create_key_bindings(),
            style=build_picker_style(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    def _list_height(self) -> Dimension:
        """Fit the items, leaving room to preview a stub, within the terminal."""
        available = self.app.output.get_size().rows - _CHROME_ROWS
        rows = min(max(len(self.state.items), _MIN_ROWS), _MAX_ROWS, available)
        return Dimension.exact(max(rows, 3))

    def _list_title(self) -> str:
        return f"{self.tool_name} functions ({len(self.state.visible)})"

    def _list_width(self) -> int:
        info = self.list_window.render_info
        return info.window_width if info is not None else 40

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        state = self.state

        @kb.add("up")
        def _up(event: KeyPressEvent) -> None:
            state.move(-1)

        @kb.add("down")
        def _down(event: KeyPressEvent) -> None:
            state.move(1)

        @kb.add("pageup")
        def _page_up(event: KeyPressEvent) -> None:
            state.move(-_PAGE)

        @kb.add("pagedown")
        def _page_down(event: KeyPressEvent) -> None:
            state.move(_PAGE)

        @kb.add("home")
        def _home(event: KeyPressEvent) -> None:
            state.move_to(0)

        @kb.add("end")
        def _end(event: KeyPressEvent) -> None:
            state.move_to(len(state.visible) - 1)

        @kb.add("tab")
        @kb.add("right")
        def _next_source(event: KeyPressEvent) -> None:
            state.cycle_source(1)

        @kb.add("s-tab")
        @kb.add("left")
        def _previous_source(event: KeyPressEvent) -> None:
            state.cycle_source(-1)

        @kb.add("space")
        def _toggle(event: KeyPressEvent) -> None:
            state.toggle()

        @kb.add("backspace")
        def _backspace(event: KeyPressEvent) -> None:
            state.backspace()

        @kb.add("c-u")
        def _clear(event: KeyPressEvent) -> None:
            state.clear_query()

        @kb.add("enter")
        def _accept(event: KeyPressEvent) -> None:
            if names := state.chosen_names():
                event.app.exit(result=names)

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(event: KeyPressEvent) -> None:
            event.app.exit(result=None)

        @kb.add("<any>")
        def _type(event: KeyPressEvent) -> None:
            if event.data.isprintable() and not event.data.isspace():
                state.type_text(event.data)

        return kb

    def _render_header(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = [
            ("class:focus", " Search: "),
            ("", self.state.query),
            ("class:muted", "▏   "),
        ]
        for index, source in enumerate(self.state.sources):
            style = "class:selected" if index == self.state.source_index else "class:muted"
            label = ALL_SOURCES_LABEL if source is None else source
            fragments.extend([(style, f" {label} "), ("", " ")])
        return fragments

    def _render_rows(self) -> StyleAndTextTuples:
        visible = self.state.visible
        if not visible:
            return [("class:muted", "  No matching functions")]
        width = self._list_width()
        source_width = (
            max(len(item.source) for item in self.state.items) if self.state.source is None else 0
        )
        fragments: StyleAndTextTuples = []
        for index, item in enumerate(visible):
            highlighted = index == self.state.cursor
            chosen = item.name in self.state.selected
            tokens = format_token_estimate(item.tokens).removesuffix(" tokens")
            check = "x" if chosen else "-" if item.seen or not item.deferred else " "
            source = f"{item.source.ljust(source_width)} " if source_width else ""
            meta = f"  {source}{tokens.rjust(6)}"
            marker = f"{'❯' if highlighted else ' '} [{check}] "
            name_width = max(width - len(marker) - len(meta), 2)
            label = _truncate(item.row_name, name_width)
            row_style = "class:selected" if highlighted else ""
            label_style = f"{row_style} class:active".strip() if chosen else row_style
            fragments.extend(
                [
                    (label_style, marker),
                    (label_style, label.ljust(name_width)),
                    (f"{row_style} class:muted".strip(), f"{meta}\n"),
                ]
            )
        return fragments

    def _render_preview(self) -> StyleAndTextTuples:
        item = self.state.current
        if item is None:
            return [("class:muted", "Nothing to preview.")]
        if item.seen:
            cost = f"{format_token_estimate(item.tokens)} on demand · seen in this conversation"
        elif item.deferred:
            cost = f"{format_token_estimate(item.tokens)} on demand (found by search_tools)"
        else:
            cost = (
                f"{format_token_estimate(item.tokens)}, already in the {self.tool_name} description"
            )
        fragments: StyleAndTextTuples = [
            ("class:focus", tool_mention_token(item.name)),
            ("class:muted", f"  {item.source} · {cost}\n"),
        ]
        if item.tool_name != item.name:
            fragments.append(("class:muted", f"tool: {item.tool_name}\n"))
        fragments.append(("", f"\n{item.stub}"))
        return fragments

    def _render_footer(self) -> StyleAndTextTuples:
        return [
            ("class:attention" if self.state.selected else "", f" {self.state.footer_text()}\n"),
            ("class:muted", _KEY_HINTS),
        ]


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else f"{text[: width - 1]}…"


async def run_tool_picker_async(surface: ToolSurface) -> tuple[str, ...] | None:
    """Pick PTC functions; returns their Python names, or None when cancelled."""
    items = tool_picker_items(surface)
    if not items or surface.programmatic is None:
        return None
    return await _ToolPicker(items, tool_name=surface.programmatic.tool_name).app.run_async()
