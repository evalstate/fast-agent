from __future__ import annotations

import asyncio
from io import StringIO
from typing import TYPE_CHECKING

import pytest
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.text import Text

from fast_agent.ui.progress.display import RichProgressDisplay
from fast_agent.ui.prompt import turn_input
from fast_agent.ui.prompt.turn_input import (
    TurnInputReader,
    add_turn_input_listener,
)
from fast_agent.ui.prompt.turn_input_dock import (
    TurnInputSnapshot,
    render_turn_input_dock,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

termios = pytest.importorskip("termios", reason="POSIX terminal modes")


@pytest.fixture
def pipe() -> Iterator[PipeInput]:
    with create_pipe_input() as input:
        yield input


def _render_dock(
    snapshot: TurnInputSnapshot,
    *,
    width: int = 80,
    max_height: int | None = None,
) -> tuple[list[str], int]:
    buf = StringIO()
    console = Console(file=buf, width=width, force_terminal=False, color_system=None)
    rendered_dock = render_turn_input_dock(snapshot, width=width, max_height=max_height)
    rendered = console.render_lines(
        rendered_dock.renderable, options=console.options.update(width=width), pad=False
    )
    return ["".join(segment.text for segment in line) for line in rendered], rendered_dock.height


def test_dock_renders_placeholder_when_empty() -> None:
    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="",
        cursor_position=0,
        queued_messages=(),
        multiline=False,
    )
    lines, height = _render_dock(snap, width=80)
    assert len(lines) == height == 2
    assert lines[0].startswith("─")
    assert "❯" in lines[1]
    assert "Type to queue" in lines[1]


def test_dock_renders_queued_messages_and_overflow() -> None:
    snap = TurnInputSnapshot(
        prompt_prefix="dev ❯",
        draft_text="",
        cursor_position=0,
        queued_messages=("first task", "second task", "third task", "fourth task"),
        multiline=False,
    )
    lines, height = _render_dock(snap, width=80)
    assert len(lines) == height == 5
    assert "↳ first task" in lines[0]
    assert "↳ second task" in lines[1]
    assert "+2 more queued" in lines[2]
    assert "dev ❯" in lines[4]


@pytest.mark.parametrize(
    ("draft", "width"),
    [
        ("aaaa " + "b" * 20 + " " + "c" * 20, 30),
        ("x\tyy\t\t\t\t\tzz", 30),
        ("界" * 21, 30),
        ("a" * 22, 30),
        ("alpha beta gamma delta epsilon zeta\nsecond line here\nthird line", 24),
    ],
)
def test_dock_height_matches_rendered_lines_for_word_wrap_tabs_and_cjk(
    draft: str, width: int
) -> None:
    snap = TurnInputSnapshot(
        prompt_prefix="agent ❯",
        draft_text=draft,
        cursor_position=len(draft),
        queued_messages=("queued one",),
        multiline=True,
    )
    lines, height = _render_dock(snap, width=width)
    assert len(lines) == height
    # Exact-width boundary cursor must remain visible.
    assert "▏" in "".join(lines)


def test_dock_caps_long_single_line_draft_by_physical_rows() -> None:
    draft = "x" * 2000 + "TAIL_MARKER"
    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text=draft,
        cursor_position=len(draft),
        queued_messages=(),
        multiline=False,
    )
    lines, height = _render_dock(snap, width=40)
    assert len(lines) == height <= 7  # 1 divider + at most 6 physical draft rows
    assert "TAIL_MARKER" in "\n".join(lines)


def test_dock_strips_pasted_ansi_controls_from_display() -> None:
    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="safe \x1b[2J\x1b[10Atext",
        cursor_position=len("safe \x1b[2J\x1b[10Atext"),
        queued_messages=("queued \x1b[31mred\x1b[0m",),
        multiline=False,
    )
    buf = StringIO()
    console = Console(file=buf, width=80, force_terminal=True, color_system=None)
    dock = render_turn_input_dock(snap, width=80)
    console.print(dock.renderable)
    output = buf.getvalue()
    assert "\x1b[2J" not in output
    assert "\x1b[10A" not in output
    assert "\x1b[31m" not in output
    assert "safe text" in output
    assert "queued red" in output


def test_dock_windows_long_multiline_draft_around_cursor() -> None:
    draft = "\n".join(f"line-{i}" for i in range(12))
    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text=draft,
        cursor_position=len(draft),
        queued_messages=(),
        multiline=True,
    )
    lines, height = _render_dock(snap, width=80)
    assert len(lines) == height
    joined = "\n".join(lines)
    assert "line-11" in joined
    assert "line-0" not in joined


def test_turn_input_buffer_editing_and_cursor_navigation(pipe: PipeInput) -> None:
    reader = TurnInputReader(input=pipe, write=lambda _s: None)
    # Type "hello world", move left 5, insert "brave " -> "hello brave world"
    pipe.send_text("hello world" + "\x1b[D" * 5 + "brave ")
    reader._readable()
    assert reader.draft_text == "hello brave world"
    assert reader.cursor_position == len("hello brave ")

    # Ctrl+A to start, Delete 'h', insert 'H'
    pipe.send_text("\x01\x1b[3~H")
    reader._readable()
    assert reader.draft_text == "Hello brave world"

    # Ctrl+E to end, Ctrl+W deletes last word ("world")
    pipe.send_text("\x05\x17")
    reader._readable()
    assert reader.draft_text == "Hello brave "

    # Ctrl+U clears to start of line
    pipe.send_text("\x15new text")
    reader._readable()
    assert reader.draft_text == "new text"


def test_up_arrow_pops_last_queued_message_into_draft(pipe: PipeInput) -> None:
    queued_counts: list[int] = []
    reader = TurnInputReader(
        input=pipe,
        on_queued=queued_counts.append,
        write=lambda _s: None,
    )
    pipe.send_text("first\rsecond\r")
    reader._readable()
    assert reader.queued_messages == ("first", "second")
    assert queued_counts == [1, 2]

    # Up arrow on empty draft pops "second" back into the draft for editing
    pipe.send_text("\x1b[A")
    reader._readable()
    assert reader.queued_messages == ("first",)
    assert reader.draft_text == "second"
    assert queued_counts == [1, 2, 1]

    # Edit and re-submit
    pipe.send_text(" edited\r")
    reader._readable()
    assert reader.queued_messages == ("first", "second edited")
    assert reader.draft_text == ""
    assert queued_counts == [1, 2, 1, 2]


@pytest.mark.asyncio
async def test_cursor_position_replays_into_next_prompt_single_and_multiline(
    pipe: PipeInput,
) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.output import DummyOutput

    from fast_agent.ui.prompt.keybindings import create_keybindings

    # Single-line cursor replay
    reader = TurnInputReader(input=pipe, write=lambda _s: None)
    pipe.send_text("hello world" + "\x1b[D" * 6)
    reader._readable()
    assert reader.cursor_position == 5
    store_typeahead(pipe, reader.stop())
    session = PromptSession(input=pipe, output=DummyOutput(), key_bindings=create_keybindings())
    result = await asyncio.wait_for(
        session.prompt_async(pre_run=lambda: pipe.send_text("!\r")),
        timeout=1,
    )
    assert result == "hello! world"

    # Multiline cursor replay: cursor at "ab|c\ndef"
    reader2 = TurnInputReader(input=pipe, write=lambda _s: None)
    pipe.send_text("abc\ndef\x1b[A\x1b[D")
    reader2._readable()
    assert reader2.draft_text == "abc\ndef"
    assert reader2.cursor_position == 2
    store_typeahead(pipe, reader2.stop())
    result2 = await asyncio.wait_for(
        session.prompt_async(pre_run=lambda: pipe.send_text("!\r")),
        timeout=1,
    )
    assert result2 == "ab!c\ndef"


@pytest.mark.asyncio
async def test_pasted_image_with_surrounding_text_converts_to_attachment_on_replay(
    pipe: PipeInput, tmp_path: Path
) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input.typeahead import store_typeahead
    from prompt_toolkit.output import DummyOutput

    from fast_agent.ui.prompt.keybindings import create_keybindings

    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    reader = TurnInputReader(input=pipe, on_queued=lambda _n: None, write=lambda _s: None)
    pipe.send_text(f"look \x1b[200~{image}\x1b[201~ please\r")
    assert reader.take_messages() == []
    store_typeahead(pipe, reader.stop())

    session = PromptSession(
        input=pipe,
        output=DummyOutput(),
        key_bindings=create_keybindings(enable_clipboard_image_paste=True),
    )
    result = await asyncio.wait_for(session.prompt_async(), timeout=1)
    assert "^file:" in result
    assert "look" in result
    assert "please" in result


def test_listener_notified_on_edit_and_snapshot_hides_while_suspended(
    monkeypatch, pipe: PipeInput
) -> None:
    events: list[TurnInputSnapshot | None] = []
    reader = TurnInputReader(input=pipe, prompt_prefix="agent ❯", write=lambda _s: None)
    monkeypatch.setattr(turn_input, "_active", reader)

    unsubscribe = add_turn_input_listener(
        lambda: events.append(turn_input.active_turn_input_snapshot())
    )
    try:
        pipe.send_text("hi")
        reader._readable()
        assert events[-1] is not None
        assert events[-1].draft_text == "hi"
        assert events[-1].prompt_prefix == "agent ❯"

        with reader.suspended():
            assert turn_input.active_turn_input_snapshot() is None
            assert events[-1] is None

        assert turn_input.active_turn_input_snapshot() is not None
        assert events[-1] is not None
    finally:
        unsubscribe()


def test_progress_display_listener_lifecycle_and_dock_visibility(
    monkeypatch, pipe: PipeInput
) -> None:
    buf = StringIO()
    test_console = Console(file=buf, width=80, force_terminal=False, color_system=None)
    listeners_before = len(turn_input._listeners)
    display = RichProgressDisplay(console=test_console)
    # Unstarted instance must not leak a listener.
    assert len(turn_input._listeners) == listeners_before

    reader = TurnInputReader(
        input=pipe,
        prompt_prefix="❯",
        on_queued=lambda _n: None,
        write=lambda _s: None,
    )
    monkeypatch.setattr(turn_input, "_active", reader)
    pipe.send_text("queued msg\rtyping now")
    reader._readable()

    def render_progress_text() -> str:
        parts = []
        for renderable in display._progress.get_renderables():
            with test_console.capture() as cap:
                test_console.print(renderable)
            parts.append(cap.get())
        return "".join(parts)

    try:
        display.start()
        assert len(turn_input._listeners) == listeners_before + 1
        active_out = render_progress_text()
        assert "↳ queued msg" in active_out
        assert "typing now" in active_out

        # Pausing must hide the dock so it is never left in scrollback
        display.pause()
        paused_out = render_progress_text()
        assert "typing now" not in paused_out
        assert "↳ queued msg" not in paused_out

        # Stopping unsubscribes; restarting re-subscribes
        display.stop()
        assert len(turn_input._listeners) == listeners_before
        display.start()
        assert len(turn_input._listeners) == listeners_before + 1
    finally:
        display.stop()
        assert len(turn_input._listeners) == listeners_before


def test_streaming_handle_includes_dock_while_live_and_omits_on_finalize(
    monkeypatch, pipe: PipeInput
) -> None:
    from fast_agent.config import Settings
    from fast_agent.ui.console_display import ConsoleDisplay
    from fast_agent.ui.streaming.display import StreamingMessageHandle

    updates: list[object] = []

    class _FakeLive:
        def __init__(self) -> None:
            self.transient = True

        def __enter__(self) -> "_FakeLive":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def update(self, renderable: object, *, refresh: bool = False) -> None:
            del refresh
            updates.append(renderable)

    reader = TurnInputReader(
        input=pipe,
        prompt_prefix="❯",
        on_queued=lambda _n: None,
        write=lambda _s: None,
    )
    monkeypatch.setattr(turn_input, "_active", reader)
    pipe.send_text("next question\rdrafting")
    reader._readable()

    handle = StreamingMessageHandle(
        display=ConsoleDisplay(Settings()),
        header_right="[dim]test[/dim]",
    )
    handle._live = _FakeLive()
    handle._live_started = True
    handle._async_mode = False

    def render_update(renderable: object) -> str:
        buf = StringIO()
        c = Console(file=buf, width=80, force_terminal=False, color_system=None)
        c.print(renderable)
        return buf.getvalue()

    handle._handle_chunk("Streaming response text")
    handle._render_current_buffer()
    assert updates
    live_text = render_update(updates[-1])
    assert "Streaming response text" in live_text
    assert "↳ next question" in live_text
    assert "drafting" in live_text
    # Header should not duplicate the queued count while the dock is visible.
    assert "⧗ 1 queued" not in live_text

    # Typing another character updates the cached frame immediately
    pipe.send_text("!")
    reader._readable()
    handle._render_current_buffer()
    live_text_updated = render_update(updates[-1])
    assert "drafting!" in live_text_updated

    # Finalizing must omit the dock so it is never preserved in scrollback
    handle._preserve_final_frame = True
    handle.finalize("Streaming response text")
    final_text = render_update(updates[-1])
    assert "Streaming response text" in final_text
    assert "drafting!" not in final_text
    assert "↳ next question" not in final_text


def test_dock_cursor_stays_visible_on_wrap_boundary_space() -> None:
    draft = "a" * 18 + " bbbbb"
    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text=draft,
        cursor_position=18,
        queued_messages=(),
        multiline=False,
    )
    rendered_dock = render_turn_input_dock(snap, width=20)
    rows = list(rendered_dock.renderable.renderables)
    has_reverse = any(
        any(span.style == "reverse" for span in row.spans) for row in rows if isinstance(row, Text)
    )
    assert has_reverse


def test_dock_strips_ansi_even_when_cursor_is_inside_escape_sequence() -> None:
    draft = "ok \x1b[31mred"
    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text=draft,
        cursor_position=5,  # inside \x1b[31m
        queued_messages=(),
        multiline=False,
    )
    lines, height = _render_dock(snap, width=40)
    assert len(lines) == height
    joined = "\n".join(lines)
    assert "[31m" not in joined
    assert "ok red" in joined


def test_dock_enforces_strict_max_height_with_queued_messages() -> None:
    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="hello",
        cursor_position=5,
        queued_messages=("one", "two", "three"),
        multiline=False,
    )
    lines, height = _render_dock(snap, width=40, max_height=3)
    assert len(lines) == height == 3
    assert "3 queued" in lines[0]


def test_deleted_image_paste_does_not_defer_subsequent_plain_message(
    pipe: PipeInput, tmp_path: Path
) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    reader = TurnInputReader(input=pipe, on_queued=lambda _n: None, write=lambda _s: None)
    # Paste image, erase line with Ctrl+U, then submit plain text
    pipe.send_text(f"\x1b[200~{image}\x1b[201~\x15plain message\rsecond\r")
    assert reader.take_messages() == ["plain message", "second"]


def test_out_of_order_image_pastes_both_preserve_boundaries_on_replay(
    pipe: PipeInput, tmp_path: Path
) -> None:
    img1 = tmp_path / "first.png"
    img2 = tmp_path / "second.png"
    img1.write_bytes(b"\x89PNG\r\n\x1a\n")
    img2.write_bytes(b"\x89PNG\r\n\x1a\n")
    reader = TurnInputReader(input=pipe, on_queued=lambda _n: None, write=lambda _s: None)
    # Paste img1, move to start with Ctrl+A, paste img2 before img1
    pipe.send_text(f"\x1b[200~{img1}\x1b[201~\x01\x1b[200~{img2}\x1b[201~ \r")
    reader._readable()
    keys = reader.stop()
    pasted_values = [k.data for k in keys if k.key == Keys.BracketedPaste]
    assert img2.as_posix() in pasted_values
    assert img1.as_posix() in pasted_values


def test_dock_divider_uses_resize_safe_short_rule_width() -> None:
    from fast_agent.ui.message_styles import SHORT_RULE_WIDTH

    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="",
        cursor_position=0,
        queued_messages=(),
        multiline=False,
    )
    lines, _ = _render_dock(snap, width=120)
    assert lines[0] == "─" * SHORT_RULE_WIDTH


def test_progress_live_render_rewinds_reflowed_height_on_width_shrink() -> None:
    from rich.segment import ControlType

    from fast_agent.ui.progress.display import _ResizeAwareLiveRender

    buf = StringIO()
    test_console = Console(file=buf, width=100, height=24, force_terminal=False, color_system=None)
    # 90-char top line + 36-char divider + 30-char input line = 3 logical lines at width 100
    renderable = Text("x" * 90 + "\n" + "─" * 36 + "\n" + "❯ " + "y" * 28)
    live_render = _ResizeAwareLiveRender(renderable, console=test_console)
    list(test_console.render(live_render))

    # Shrink terminal to width 50: 90-char line reflows to 2 rows -> 4 physical rows total
    test_console.width = 50
    control = live_render.position_cursor()
    cursor_ups = [
        code for code in control.segment.control or () if code[0] == ControlType.CURSOR_UP
    ]
    assert len(cursor_ups) == 3  # 4 physical rows -> 3 CURSOR_UP moves


def test_dock_renders_completion_popup_within_max_height() -> None:
    from fast_agent.ui.prompt.turn_input_dock import TurnInputCompletion

    completions = tuple(
        TurnInputCompletion(
            text=f"cmd{i}",
            display=f"/cmd{i}",
            display_meta=f"Description {i}",
            start_position=-1,
        )
        for i in range(6)
    )
    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="/c",
        cursor_position=2,
        queued_messages=("queued one", "queued two"),
        multiline=False,
        completions=completions,
        selected_completion=1,
    )

    lines, _ = _render_dock(snap, width=80, max_height=6)
    assert len(lines) <= 6
    joined = "\n".join(lines)
    assert "/cmd1" in joined
    assert "▸" in joined

    tight_lines, _ = _render_dock(snap, width=80, max_height=3)
    assert len(tight_lines) <= 3


def test_turn_input_reader_tab_completion_navigation_and_esc() -> None:
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.key_binding import KeyPress
    from prompt_toolkit.keys import Keys

    from fast_agent.ui.prompt.turn_input import TurnInputReader

    class _FakeCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            for cmd in ("/usage", "/update", "/status"):
                if cmd.startswith(text):
                    yield Completion(
                        cmd, start_position=-len(text), display=cmd, display_meta="cmd"
                    )

    cancelled = 0

    def on_cancel() -> None:
        nonlocal cancelled
        cancelled += 1

    with create_pipe_input() as pipe:
        reader = TurnInputReader(
            input=pipe,
            completer=_FakeCompleter(),
            on_cancel=on_cancel,
            on_queued=lambda _n: None,
            write=lambda _s: None,
        )

        def press(*keys: KeyPress) -> None:
            for k in keys:
                reader._on_key(k)

        # Typing "/u" auto-populates completions (/usage, /update)
        press(KeyPress("/", "/"), KeyPress("u", "u"))
        snap = reader.snapshot()
        assert [c.display for c in snap.completions] == ["/usage", "/update"]
        assert snap.selected_completion == 0

        # Tab cycles selection forward, Shift+Tab cycles backward
        press(KeyPress(Keys.ControlI, "\t"))
        assert reader.snapshot().selected_completion == 1
        press(KeyPress(Keys.BackTab, ""))
        assert reader.snapshot().selected_completion == 0

        # Down selects /update, Enter accepts completion into buffer without submitting yet
        press(KeyPress(Keys.Down, ""))
        assert reader.snapshot().selected_completion == 1
        press(KeyPress(Keys.ControlM, "\r"))
        assert reader.snapshot().draft_text == "/update"
        assert reader.snapshot().completions == ()
        assert reader.queued_messages == ()

        # Esc closes an open completion menu without cancelling the turn
        press(KeyPress(Keys.ControlU, ""), KeyPress("/", "/"), KeyPress("u", "u"))
        assert len(reader.snapshot().completions) == 2
        reader._escape_pending = True
        reader.flush()
        assert reader.snapshot().completions == ()
        assert cancelled == 0

        # Second Esc with menu closed cancels the turn
        reader._escape_pending = True
        reader.flush()
        assert cancelled == 1

        # Tab with single match auto-accepts immediately
        press(KeyPress(Keys.ControlU, ""), KeyPress("/", "/"), KeyPress("s", "s"))
        press(KeyPress(Keys.ControlI, "\t"))
        assert reader.snapshot().draft_text == "/status"
        assert reader.snapshot().completions == ()


def test_turn_input_reader_on_submit_intercepts_concurrent_commands() -> None:
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.key_binding import KeyPress
    from prompt_toolkit.keys import Keys

    from fast_agent.ui.prompt.turn_input import TurnInputReader

    intercepted: list[str] = []

    def on_submit(text: str) -> bool:
        if text.startswith("/usage"):
            intercepted.append(text)
            return True
        return False

    with create_pipe_input() as pipe:
        reader = TurnInputReader(
            input=pipe,
            on_submit=on_submit,
            on_cancel=lambda: None,
            on_queued=lambda _n: None,
            write=lambda _s: None,
        )
        # Concurrent command is intercepted and NOT queued
        for k in [KeyPress(ch, ch) for ch in "/usage"] + [KeyPress(Keys.ControlM, "\r")]:
            reader._on_key(k)
        assert intercepted == ["/usage"]
        assert reader.queued_messages == ()

        # Normal message is NOT intercepted and IS queued
        for k in [KeyPress(ch, ch) for ch in "hello"] + [KeyPress(Keys.ControlM, "\r")]:
            reader._on_key(k)
        assert intercepted == ["/usage"]
        assert reader.queued_messages == ("hello",)


def test_progress_display_paused_keep_dock_hides_tasks_and_keeps_dock_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.event_progress import ProgressAction, ProgressEvent
    from fast_agent.ui.progress import display as progress_module
    from fast_agent.ui.progress.display import RichProgressDisplay

    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="draft during shell",
        cursor_position=18,
        queued_messages=(),
        multiline=False,
    )
    monkeypatch.setattr(progress_module, "active_turn_input_snapshot", lambda: snap)

    buf = StringIO()
    test_console = Console(file=buf, width=80, height=24, force_terminal=False, color_system=None)
    display = RichProgressDisplay(console=test_console)
    display.start()
    try:
        display.update(
            ProgressEvent(
                action=ProgressAction.CALLING_TOOL,
                target="agent",
                details="shell",
                agent_name="agent",
            )
        )
        assert any(task.visible for task in display._progress.tasks)

        with display.paused(keep_dock=True):
            assert display._paused is True
            assert display._progress.dock_active is True
            assert display._live_started() is True
            assert all(not task.visible for task in display._progress.tasks)

            # A nested full pause() must stop Live and hide the dock, then
            # restore dock-only mode when it exits.
            with display.paused():
                assert display._progress.dock_active is False
                assert display._live_started() is False

            assert display._paused is True
            assert display._progress.dock_active is True
            assert display._live_started() is True
            assert all(not task.visible for task in display._progress.tasks)

        assert display._paused is False
        assert display._progress.dock_active is True
        assert display._live_started() is True
        assert any(task.visible for task in display._progress.tasks)
    finally:
        display.stop()


def test_progress_display_paused_keep_dock_promotes_deferred_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.ui.progress import display as progress_module
    from fast_agent.ui.progress.display import RichProgressDisplay

    snap = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="",
        cursor_position=0,
        queued_messages=(),
        multiline=False,
    )
    monkeypatch.setattr(progress_module, "active_turn_input_snapshot", lambda: snap)

    buf = StringIO()
    test_console = Console(file=buf, width=80, height=24, force_terminal=False, color_system=None)
    display = RichProgressDisplay(console=test_console)
    display.start()
    try:
        # Simulate streaming ending with a debounced progress resume before shell starts
        display.pause()
        display.resume(debounce_seconds=10.0)
        assert display._paused is True
        assert display._live_started() is False

        with display.paused(keep_dock=True):
            assert display._paused is True
            assert display._progress.dock_active is True
            assert display._live_started() is True

        assert display._paused is False
        assert display._progress.dock_active is True
        assert display._live_started() is True
    finally:
        display.stop()


def test_progress_display_paused_keep_dock_stops_live_without_turn_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.ui.progress import display as progress_module
    from fast_agent.ui.progress.display import RichProgressDisplay

    monkeypatch.setattr(progress_module, "active_turn_input_snapshot", lambda: None)

    buf = StringIO()
    test_console = Console(file=buf, width=80, height=24, force_terminal=False, color_system=None)
    display = RichProgressDisplay(console=test_console)
    display.start()
    try:
        with display.paused(keep_dock=True):
            assert display._paused is True
            assert display._progress.dock_active is False
            assert display._live_started() is False

        assert display._paused is False
        assert display._live_started() is True
    finally:
        display.stop()


def test_dock_highlights_prompt_arrow_red_for_shell_draft() -> None:
    snap = TurnInputSnapshot(
        prompt_prefix="agent ❯",
        draft_text="!git status",
        cursor_position=11,
        queued_messages=(),
        multiline=False,
    )
    rendered = render_turn_input_dock(snap, width=80, terminal_height=24)
    input_row = rendered.renderable.renderables[-1]
    assert isinstance(input_row, Text)
    spans = [(input_row.plain[s.start : s.end], s.style) for s in input_row.spans]
    assert ("agent ", "bold cyan") in spans
    assert ("❯ ", "bold red") in spans


def test_turn_input_reader_ctrl_t_toggles_multiline_mid_turn(pipe: PipeInput) -> None:
    toggled: list[bool] = []
    reader = TurnInputReader(
        input=pipe,
        multiline=False,
        on_toggle_multiline=toggled.append,
        on_queued=lambda _n: None,
        write=lambda _s: None,
    )
    # Toggle multiline on with Ctrl+T: Enter inserts newline, Ctrl+J submits
    pipe.send_text("\x14line1\rline2\n")
    assert reader.take_messages() == ["line1\nline2"]
    assert toggled == [True]
    assert reader.snapshot().multiline is True

    # Toggle multiline back off with Ctrl+T: Enter submits again
    pipe.send_text("\x14single\r")
    assert reader.take_messages() == ["single"]
    assert toggled == [True, False]
    assert reader.snapshot().multiline is False

    # Alt+Enter inserts newline in normal mode, and submits in multiline mode
    pipe.send_text("a\x1b\rb\r")
    assert reader.take_messages() == ["a\nb"]
    pipe.send_text("\x14c\rd\x1b\r")
    assert reader.take_messages() == ["c\nd"]


def test_dock_placeholder_hints_switch_for_multiline() -> None:
    normal = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="",
        cursor_position=0,
        queued_messages=(),
        multiline=False,
    )
    multiline = TurnInputSnapshot(
        prompt_prefix="❯",
        draft_text="",
        cursor_position=0,
        queued_messages=(),
        multiline=True,
    )
    normal_lines, _ = _render_dock(normal, width=140)
    multiline_lines, _ = _render_dock(multiline, width=140)
    assert "Esc to interrupt, Alt+Enter new line, Ctrl+T toggle multiline" in normal_lines[1]
    assert "Esc to interrupt, Alt+Enter submit, Ctrl+T toggle multiline" in multiline_lines[1]
