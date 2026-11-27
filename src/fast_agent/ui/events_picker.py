"""Interactive picker for MCP event subscriptions (``/events select``).

Selection, grouping, filtering, and detail formatting live in ``EventsPickerState``
so they can be tested without a terminal; ``_EventsPicker`` wraps it in a thin
prompt_toolkit application using the shared picker theme and layout helpers.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import textwrap
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.widgets import Frame

from fast_agent.commands.handlers.events import EventsPickerResult
from fast_agent.mcp.events_subscriptions import format_poll_interval
from fast_agent.ui.picker_theme import build_picker_style
from fast_agent.ui.single_list_picker_layout import picker_terminal_size, truncate_picker_text

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from prompt_toolkit.formatted_text import StyleAndTextTuples
    from prompt_toolkit.key_binding import KeyPressEvent

    from fast_agent.mcp.events_extension import EventDescriptor
    from fast_agent.mcp.events_subscriptions import EventSubscription, SubscriptionOrigin

SUBSCRIBED_GROUP = "subscribed"
"""Tab group showing active and ended event subscriptions."""

_PAGE = 10
_MIN_ROWS = 10
_MAX_ROWS = 18
_CHROME_ROWS = 7
_KEY_HINTS = (
    " Type to search · Tab group · Space toggle · w wake · p poll · x/Del remove · Enter · Esc"
)
_WORD_SEPARATORS = re.compile(r"[\s_/.:-]+")


class EventsPickerManager(Protocol):
    """Subset of ``EventSubscriptionManager`` needed by the interactive picker."""

    def subscriptions(self) -> list[EventSubscription]: ...

    async def subscribe(
        self,
        server_name: str,
        name: str,
        arguments: dict[str, object] | None = None,
        *,
        wake: bool = False,
        once: bool = False,
        min_poll_ms: int | None = None,
        origin: SubscriptionOrigin = "user",
    ) -> EventSubscription: ...

    async def unsubscribe(self, key: str) -> bool: ...

    def poll_now(self, key: str) -> bool: ...


@runtime_checkable
class _DisplayServerNameManager(Protocol):
    def display_server_name(self, server_name: str) -> str: ...


def _identity_server_name(server_name: str) -> str:
    return server_name


def _required_argument_names(descriptor: EventDescriptor | None) -> tuple[str, ...]:
    if descriptor is None or not isinstance(descriptor.input_schema, Mapping):
        return ()
    properties = descriptor.input_schema.get("properties")
    raw_required = descriptor.input_schema.get("required")
    if not isinstance(raw_required, list):
        return ()
    if isinstance(properties, Mapping):
        return tuple(name for name in raw_required if isinstance(name, str) and name in properties)
    return tuple(name for name in raw_required if isinstance(name, str))


def _format_arguments_inline(arguments: Mapping[str, object]) -> str:
    if not arguments:
        return ""
    parts: list[str] = []
    for key, value in sorted(arguments.items()):
        rendered = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class EventsPickerItem:
    """One row in the events picker (either a subscription or an offered event type)."""

    kind: Literal["subscribed", "available"]
    server_name: str
    name: str
    descriptor: EventDescriptor | None = None
    subscription: EventSubscription | None = None

    @property
    def is_subscribed(self) -> bool:
        return self.kind == "subscribed" and self.subscription is not None

    @property
    def is_active(self) -> bool:
        return self.subscription is not None and self.subscription.status == "active"

    @property
    def is_pollable(self) -> bool:
        return self.descriptor is not None and self.descriptor.supports("poll")

    @property
    def is_selectable(self) -> bool:
        return self.is_subscribed or self.is_pollable

    @property
    def is_dimmed(self) -> bool:
        if self.is_subscribed:
            return not self.is_active
        return not self.is_pollable

    @property
    def required_args(self) -> tuple[str, ...]:
        return _required_argument_names(self.descriptor)

    def prefill_command(self) -> str | None:
        """Draft ``/events subscribe`` command for an available pollable event with required args."""
        if self.kind != "available" or not self.is_pollable or not self.required_args:
            return None
        arg_placeholders = " ".join(f"{arg}=" for arg in self.required_args)
        return (
            f"/events subscribe {shlex.quote(self.server_name)} "
            f"{shlex.quote(self.name)} {arg_placeholders}"
        )

    def format_row(self, *, subscribed: bool = False) -> str:
        """Render the single-line list row label."""
        if self.kind == "subscribed" and self.subscription is not None:
            sub = self.subscription
            args_text = _format_arguments_inline(sub.arguments)
            head = f"● {self.server_name} {self.name}"
            if args_text:
                head = f"{head} {args_text}"
            mode = "wake" if sub.wake else "passive"
            return f"{head} · {mode} · {sub.delivered} delivered · {sub.status}"

        bullet = "●" if subscribed else "○"
        req = self.required_args
        req_text = f" ({', '.join(f'{arg}*' for arg in req)})" if req else ""
        delivery = " ".join(self.descriptor.delivery) if self.descriptor is not None else ""
        suffix = f" {delivery}" if delivery else ""
        return f"{bullet} {self.server_name} {self.name}{req_text}{suffix}"

    def matches(self, query: str) -> bool:
        """Whether ``query`` matches this row by substring or word-prefix terms."""
        needle = query.casefold()
        fields: list[str] = [self.server_name, self.name]
        if self.descriptor is not None and self.descriptor.description:
            fields.append(self.descriptor.description)
        if self.subscription is not None:
            fields.append(self.subscription.key)
            if self.subscription.arguments:
                fields.append(_format_arguments_inline(self.subscription.arguments))
        lowered = tuple(value.casefold() for value in fields)
        if any(needle in value for value in lowered):
            return True
        terms = [term for term in _WORD_SEPARATORS.split(needle) if term]
        words = [word for value in lowered for word in _WORD_SEPARATORS.split(value) if word]
        return bool(terms) and all(any(word.startswith(term) for word in words) for term in terms)


def format_event_detail_fragments(item: EventsPickerItem | None) -> StyleAndTextTuples:
    """Build styled detail-pane fragments for the highlighted picker item."""
    if item is None:
        return [("class:muted", "Nothing selected.")]

    fragments: StyleAndTextTuples = [
        ("class:focus", f"{item.server_name} / {item.name}\n"),
    ]

    if item.subscription is not None:
        sub = item.subscription
        status = f"{sub.status}: {sub.last_error}" if sub.last_error else sub.status
        mode = "wake" if sub.wake else "passive"
        meta_parts = [f"id: {sub.key}", mode, f"{sub.delivered} delivered", status]
        if sub.once:
            meta_parts.append("once")
        if sub.min_poll_ms is not None:
            meta_parts.append(f"every {format_poll_interval(sub.min_poll_ms)}")
        fragments.append(("class:muted", " · ".join(meta_parts) + "\n"))
        if sub.arguments:
            fragments.append(("", f"Args: {_format_arguments_inline(sub.arguments)}\n"))

    descriptor = item.descriptor
    if descriptor is not None:
        delivery = ", ".join(descriptor.delivery) or "none"
        if not descriptor.supports("poll"):
            fragments.append(
                ("class:attention", f"Delivery: {delivery} (not subscribable via poll)\n")
            )
        else:
            fragments.append(("class:muted", f"Delivery: {delivery}\n"))

        if descriptor.description:
            fragments.append(("", f"\n{descriptor.description.strip()}\n"))

        if isinstance(descriptor.input_schema, Mapping):
            properties = descriptor.input_schema.get("properties")
            required_set = set(_required_argument_names(descriptor))
            if isinstance(properties, Mapping) and properties:
                fragments.append(("class:focus", "\nArguments:\n"))
                for key, raw_prop in properties.items():
                    if not isinstance(key, str):
                        continue
                    mark = "*" if key in required_set else ""
                    prop = raw_prop if isinstance(raw_prop, Mapping) else {}
                    extras: list[str] = []
                    prop_type = prop.get("type")
                    if isinstance(prop_type, str):
                        extras.append(prop_type)
                    enum_values = prop.get("enum")
                    if isinstance(enum_values, list) and enum_values:
                        rendered_enum = ", ".join(str(v) for v in enum_values[:6])
                        if len(enum_values) > 6:
                            rendered_enum += ", …"
                        extras.append(f"enum: {rendered_enum}")
                    if "default" in prop:
                        extras.append(f"default={prop['default']!r}")
                    extra_text = f" ({'; '.join(extras)})" if extras else ""
                    desc = prop.get("description")
                    desc_text = (
                        f" — {desc.strip()}" if isinstance(desc, str) and desc.strip() else ""
                    )
                    fragments.append(("", f"  {key}{mark}{extra_text}{desc_text}\n"))

        if isinstance(descriptor.payload_schema, Mapping):
            payload_props = descriptor.payload_schema.get("properties")
            if isinstance(payload_props, Mapping) and payload_props:
                keys = [k for k in payload_props if isinstance(k, str)]
                if keys:
                    fragments.append(("class:focus", "\nPayload keys:\n"))
                    fragments.append(("class:muted", f"  {', '.join(keys)}\n"))

    return fragments


def format_event_detail_text(item: EventsPickerItem | None) -> str:
    """Plain-text rendering of ``format_event_detail_fragments`` for tests."""
    return "".join(fragment[1] for fragment in format_event_detail_fragments(item))


@dataclass(slots=True)
class EventsPickerState:
    """Pure-Python state for the interactive ``/events select`` picker."""

    subscriptions: tuple[EventsPickerItem, ...] = ()
    available_by_server: dict[str, tuple[EventsPickerItem, ...]] = field(default_factory=dict)
    query: str = ""
    group_index: int = 0
    cursor: int = 0
    status_message: str = ""
    display_server_name: Callable[[str], str] = field(
        default=_identity_server_name, repr=False, compare=False
    )

    @classmethod
    def from_data(
        cls,
        *,
        subscriptions: Sequence[EventSubscription],
        descriptors_by_server: Mapping[str, Sequence[EventDescriptor]],
        display_server_name: Callable[[str], str] = _identity_server_name,
    ) -> EventsPickerState:
        descriptor_lookup: dict[tuple[str, str], EventDescriptor] = {}
        available_by_server: dict[str, tuple[EventsPickerItem, ...]] = {}
        for raw_server_name, descriptors in descriptors_by_server.items():
            server_name = display_server_name(raw_server_name)
            for descriptor in descriptors:
                descriptor_lookup[(server_name, descriptor.name)] = descriptor
            available_by_server[server_name] = tuple(
                EventsPickerItem(
                    kind="available",
                    server_name=server_name,
                    name=descriptor.name,
                    descriptor=descriptor,
                )
                for descriptor in descriptors
            )

        sub_items = tuple(
            EventsPickerItem(
                kind="subscribed",
                server_name=display_server_name(sub.server_name),
                name=sub.name,
                descriptor=descriptor_lookup.get((display_server_name(sub.server_name), sub.name)),
                subscription=sub,
            )
            for sub in subscriptions
        )
        initial_group = 0 if sub_items else (1 if available_by_server else 0)
        return cls(
            subscriptions=sub_items,
            available_by_server=available_by_server,
            group_index=initial_group,
            display_server_name=display_server_name,
        )

    def refresh_subscriptions(self, subscriptions: Sequence[EventSubscription]) -> None:
        """Replace subscription rows while preserving descriptors, group, and cursor bounds."""
        descriptor_lookup: dict[tuple[str, str], EventDescriptor] = {}
        for server_name, items in self.available_by_server.items():
            for item in items:
                if item.descriptor is not None:
                    descriptor_lookup[(server_name, item.name)] = item.descriptor
        self.subscriptions = tuple(
            EventsPickerItem(
                kind="subscribed",
                server_name=self.display_server_name(sub.server_name),
                name=sub.name,
                descriptor=descriptor_lookup.get(
                    (self.display_server_name(sub.server_name), sub.name)
                ),
                subscription=sub,
            )
            for sub in subscriptions
        )
        self._clamp_cursor()

    @property
    def groups(self) -> tuple[str, ...]:
        return (SUBSCRIBED_GROUP, *self.available_by_server.keys())

    @property
    def group(self) -> str:
        groups = self.groups
        return groups[self.group_index % len(groups)]

    @property
    def visible(self) -> tuple[EventsPickerItem, ...]:
        group = self.group
        items = (
            self.subscriptions
            if group == SUBSCRIBED_GROUP
            else self.available_by_server.get(group, ())
        )
        if not self.query:
            return items
        return tuple(item for item in items if item.matches(self.query))

    @property
    def current(self) -> EventsPickerItem | None:
        visible = self.visible
        return visible[self.cursor] if 0 <= self.cursor < len(visible) else None

    def _clamp_cursor(self) -> None:
        count = len(self.visible)
        self.cursor = min(max(self.cursor, 0), max(count - 1, 0))

    def type_text(self, text: str) -> None:
        self.query += text
        self.cursor = 0

    def backspace(self) -> None:
        self.query = self.query[:-1]
        self.cursor = 0

    def clear_query(self) -> None:
        self.query = ""
        self.cursor = 0

    def cycle_group(self, delta: int) -> None:
        self.group_index = (self.group_index + delta) % len(self.groups)
        self.cursor = 0

    def move(self, delta: int) -> None:
        count = len(self.visible)
        self.cursor = min(max(self.cursor + delta, 0), max(count - 1, 0))

    def move_to(self, index: int) -> None:
        self.cursor = 0
        self.move(index)

    def is_item_subscribed(self, item: EventsPickerItem) -> bool:
        """Whether an available item has an active subscription (showing ● on the server tab)."""
        if item.kind == "subscribed":
            return item.is_active
        return any(
            sub_item.is_active
            and sub_item.server_name == item.server_name
            and sub_item.name == item.name
            for sub_item in self.subscriptions
        )

    def detail_fragments(self) -> StyleAndTextTuples:
        return format_event_detail_fragments(self.current)

    def detail_text(self) -> str:
        return format_event_detail_text(self.current)

    def toggle_wake_current(self) -> EventSubscription | None:
        """Toggle ``wake`` on the currently highlighted active subscription."""
        item = self.current
        if item is None or not item.is_active or item.subscription is None:
            return None
        item.subscription.wake = not item.subscription.wake
        mode = "wake" if item.subscription.wake else "passive"
        self.status_message = f"{item.subscription.key}: {mode}"
        return item.subscription

    def enter_action(self) -> tuple[Literal["close", "subscribe", "prefill", "noop"], str | None]:
        """Determine what pressing Enter does on the current row."""
        item = self.current
        if item is None:
            return ("close", None)
        if item.is_subscribed:
            return ("close", None)
        if not item.is_pollable:
            self.status_message = f"{item.name} is not subscribable via poll"
            return ("noop", None)
        prefill = item.prefill_command()
        if prefill is not None:
            return ("prefill", prefill)
        return ("subscribe", None)


class _EventsPicker:
    def __init__(
        self,
        *,
        manager: EventsPickerManager,
        descriptors_by_server: Mapping[str, Sequence[EventDescriptor]],
    ) -> None:
        self.manager = manager
        display_fn = (
            manager.display_server_name
            if isinstance(manager, _DisplayServerNameManager)
            else _identity_server_name
        )
        self.state = EventsPickerState.from_data(
            subscriptions=manager.subscriptions(),
            descriptors_by_server=descriptors_by_server,
            display_server_name=display_fn,
        )
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._mutation_in_flight = False
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
        detail_window = Window(
            FormattedTextControl(self._render_detail),
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
                        Frame(self.list_window, title=self._list_title, width=Dimension(weight=3)),
                        Frame(detail_window, title="Details", width=Dimension(weight=2)),
                    ]
                ),
                Window(FormattedTextControl(self._render_footer), height=2),
            ]
        )
        self.app: Application[EventsPickerResult | None] = Application(
            layout=Layout(body, focused_element=self.list_window),
            key_bindings=self._create_key_bindings(),
            style=build_picker_style(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    def _list_height(self) -> Dimension:
        available = picker_terminal_size().rows - _CHROME_ROWS
        total = max(
            len(self.state.subscriptions),
            max((len(items) for items in self.state.available_by_server.values()), default=0),
        )
        rows = min(max(total, _MIN_ROWS), _MAX_ROWS, available)
        return Dimension.exact(max(rows, 3))

    def _list_title(self) -> str:
        return f"Events: {self.state.group} ({len(self.state.visible)})"

    def _list_width(self) -> int:
        info = self.list_window.render_info
        return info.window_width if info is not None else 50

    def _spawn(self, coro: Coroutine[object, object, None]) -> None:
        if self._mutation_in_flight:
            coro.close()
            return
        self._mutation_in_flight = True

        async def _guarded() -> None:
            try:
                await coro
            finally:
                self._mutation_in_flight = False

        task: asyncio.Task[None] = asyncio.create_task(_guarded())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _cancel_tasks(self) -> None:
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._background_tasks.clear()
        self._mutation_in_flight = False

    def _invalidate_if_running(self) -> None:
        if self.app.is_running:
            self.app.invalidate()

    async def _subscribe_item(self, item: EventsPickerItem, *, exit_after: bool) -> None:
        try:
            sub = await self.manager.subscribe(
                item.server_name, item.name, {}, wake=True, origin="user"
            )
        except Exception as exc:
            self.state.status_message = f"Subscribe failed: {exc}"
            self._invalidate_if_running()
            return
        self.state.refresh_subscriptions(self.manager.subscriptions())
        display_server = self.state.display_server_name(sub.server_name)
        message = f"Subscribed to `{sub.name}` on `{display_server}` as `{sub.key}`."
        self.state.status_message = f"Subscribed to {display_server}/{sub.name}"
        if exit_after:
            if self.app.is_running:
                self.app.exit(result=EventsPickerResult(message=message))
        else:
            self._invalidate_if_running()

    async def _toggle_wake_item(self, sub: EventSubscription) -> None:
        try:
            updated = await self.manager.subscribe(
                sub.server_name,
                sub.name,
                dict(sub.arguments),
                wake=not sub.wake,
                once=sub.once,
                min_poll_ms=sub.min_poll_ms,
                origin="user",
            )
        except Exception as exc:
            self.state.status_message = f"Wake toggle failed: {exc}"
            self._invalidate_if_running()
            return
        self.state.refresh_subscriptions(self.manager.subscriptions())
        mode = "wake" if updated.wake else "passive"
        self.state.status_message = f"{updated.key}: {mode}"
        self._invalidate_if_running()

    async def _unsubscribe_item(self, key: str) -> None:
        try:
            removed = await self.manager.unsubscribe(key)
        except Exception as exc:
            self.state.status_message = f"Unsubscribe failed: {exc}"
            self._invalidate_if_running()
            return
        self.state.refresh_subscriptions(self.manager.subscriptions())
        self.state.status_message = (
            f"Removed subscription {key}" if removed else f"Subscription {key} not found"
        )
        self._invalidate_if_running()

    async def run_async(self) -> EventsPickerResult | None:
        try:
            return await self.app.run_async()
        finally:
            await self._cancel_tasks()

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        state = self.state

        @kb.add("up")
        def _up(event: KeyPressEvent) -> None:
            del event
            state.move(-1)

        @kb.add("down")
        def _down(event: KeyPressEvent) -> None:
            del event
            state.move(1)

        @kb.add("pageup")
        def _page_up(event: KeyPressEvent) -> None:
            del event
            state.move(-_PAGE)

        @kb.add("pagedown")
        def _page_down(event: KeyPressEvent) -> None:
            del event
            state.move(_PAGE)

        @kb.add("home")
        def _home(event: KeyPressEvent) -> None:
            del event
            state.move_to(0)

        @kb.add("end")
        def _end(event: KeyPressEvent) -> None:
            del event
            state.move_to(len(state.visible) - 1)

        @kb.add("tab")
        @kb.add("right")
        def _next_group(event: KeyPressEvent) -> None:
            del event
            state.cycle_group(1)

        @kb.add("s-tab")
        @kb.add("left")
        def _previous_group(event: KeyPressEvent) -> None:
            del event
            state.cycle_group(-1)

        @kb.add("space")
        def _toggle(event: KeyPressEvent) -> None:
            del event
            item = state.current
            if item is None:
                return
            if item.is_active and item.subscription is not None:
                self._spawn(self._unsubscribe_item(item.subscription.key))
                return
            if item.kind == "available":
                if not item.is_pollable:
                    state.status_message = f"{item.name} is not subscribable via poll"
                    return
                if item.required_args:
                    state.status_message = "Press Enter to fill required arguments"
                    return
                self._spawn(self._subscribe_item(item, exit_after=False))

        @kb.add("delete")
        def _delete(event: KeyPressEvent) -> None:
            del event
            item = state.current
            if item is not None and item.is_subscribed and item.subscription is not None:
                self._spawn(self._unsubscribe_item(item.subscription.key))

        @kb.add("backspace")
        def _backspace(event: KeyPressEvent) -> None:
            del event
            state.backspace()

        @kb.add("c-u")
        def _clear(event: KeyPressEvent) -> None:
            del event
            state.clear_query()

        @kb.add("enter")
        def _accept(event: KeyPressEvent) -> None:
            action, payload = state.enter_action()
            if action == "close":
                event.app.exit(result=EventsPickerResult())
            elif action == "prefill" and payload is not None:
                event.app.exit(result=EventsPickerResult(buffer_prefill=payload))
            elif action == "subscribe" and state.current is not None:
                self._spawn(self._subscribe_item(state.current, exit_after=True))

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(event: KeyPressEvent) -> None:
            event.app.exit(result=None)

        @kb.add("<any>")
        def _type(event: KeyPressEvent) -> None:
            data = event.data
            if (
                not state.query
                and state.current is not None
                and state.current.is_subscribed
                and state.current.subscription is not None
            ):
                sub = state.current.subscription
                if data == "x":
                    self._spawn(self._unsubscribe_item(sub.key))
                    return
                if state.current.is_active:
                    if data == "w":
                        self._spawn(self._toggle_wake_item(sub))
                        return
                    if data == "p":
                        ok = self.manager.poll_now(sub.key)
                        state.status_message = (
                            f"Polling {sub.key} now" if ok else f"Cannot poll {sub.key}"
                        )
                        return
            if data.isprintable() and not data.isspace():
                state.type_text(data)

        return kb

    def _render_header(self) -> StyleAndTextTuples:
        fragments: StyleAndTextTuples = [
            ("class:focus", " Search: "),
            ("", self.state.query),
            ("class:muted", "▏   "),
        ]
        for index, group in enumerate(self.state.groups):
            style = "class:selected" if index == self.state.group_index else "class:muted"
            fragments.extend([(style, f" {group} "), ("", " ")])
        return fragments

    def _render_rows(self) -> StyleAndTextTuples:
        visible = self.state.visible
        if not visible:
            if self.state.group == SUBSCRIBED_GROUP and not self.state.query:
                return [("class:muted", "  No event subscriptions (Tab to browse servers)")]
            return [("class:muted", "  No matching events")]
        width = self._list_width()
        fragments: StyleAndTextTuples = []
        for index, item in enumerate(visible):
            highlighted = index == self.state.cursor
            marker = "❯ " if highlighted else "  "
            text = truncate_picker_text(
                item.format_row(subscribed=self.state.is_item_subscribed(item)),
                max(width - len(marker), 4),
            )
            if highlighted and item.is_dimmed:
                style = "class:selected class:muted"
            elif highlighted:
                style = "class:selected"
            elif item.is_dimmed:
                style = "class:muted"
            elif item.is_active:
                style = "class:active"
            else:
                style = ""
            fragments.append((style, f"{marker}{text}\n"))
        return fragments

    def _render_detail(self) -> StyleAndTextTuples:
        width = max(self._detail_width(), 24)
        wrapped: StyleAndTextTuples = []
        for fragment in self.state.detail_fragments():
            style, text = fragment[0], fragment[1]
            if not text:
                continue
            lines = text.split("\n")
            for idx, line in enumerate(lines):
                has_newline = idx < len(lines) - 1
                if len(line) > width:
                    wrapped_line = textwrap.fill(
                        line,
                        width=width,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                else:
                    wrapped_line = line
                wrapped.append((style, wrapped_line + ("\n" if has_newline else "")))
        return wrapped

    def _detail_width(self) -> int:
        try:
            cols = self.app.output.get_size().columns
        except Exception:
            return 72
        # Match single_list_picker_layout's right pane (~half width minus borders/padding).
        return max((cols // 2) - 6, 24)

    def _render_footer(self) -> StyleAndTextTuples:
        status = self.state.status_message or (f"{len(self.state.subscriptions)} subscription(s)")
        return [
            ("class:focus", f" {status}\n"),
            ("class:muted", _KEY_HINTS),
        ]


async def run_events_picker_async(
    *,
    manager: EventsPickerManager,
    descriptors_by_server: Mapping[str, Sequence[EventDescriptor]],
) -> EventsPickerResult | None:
    """Open the interactive events picker and return its result, or None if cancelled."""
    return await _EventsPicker(
        manager=manager,
        descriptors_by_server=descriptors_by_server,
    ).run_async()
