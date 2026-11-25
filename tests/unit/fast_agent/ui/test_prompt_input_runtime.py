from __future__ import annotations

from typing import TYPE_CHECKING, TextIO, cast

import pytest

from fast_agent.ui.command_payloads import EOFCommand
from fast_agent.ui.prompt import input_runtime
from fast_agent.ui.prompt.keybindings import PromptInputInterrupt

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession


class _FakeStream:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty
        self.writes: list[str] = []
        self.flush_calls = 0

    def isatty(self) -> bool:
        return self._tty

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        self.flush_calls += 1


def test_clear_prompt_echo_line_erases_regular_input() -> None:
    stream = _FakeStream(tty=True)

    input_runtime._clear_prompt_echo_line("good morning", stream=cast("TextIO", stream))

    assert stream.writes == [input_runtime._ERASE_PREVIOUS_LINE_SEQ]
    assert stream.flush_calls == 1


def test_clear_prompt_echo_line_keeps_slash_and_shell_commands_visible() -> None:
    stream = _FakeStream(tty=True)

    input_runtime._clear_prompt_echo_line("/help", stream=cast("TextIO", stream))
    input_runtime._clear_prompt_echo_line("!pwd", stream=cast("TextIO", stream))

    assert stream.writes == []
    assert stream.flush_calls == 0


def test_clear_prompt_echo_line_skips_multiline_and_non_tty() -> None:
    tty_stream = _FakeStream(tty=True)
    non_tty_stream = _FakeStream(tty=False)

    input_runtime._clear_prompt_echo_line("line1\nline2", stream=cast("TextIO", tty_stream))
    input_runtime._clear_prompt_echo_line("hello", stream=cast("TextIO", non_tty_stream))

    assert tty_stream.writes == []
    assert tty_stream.flush_calls == 0
    assert non_tty_stream.writes == []
    assert non_tty_stream.flush_calls == 0


def test_format_prompt_prefix_omits_default_agent_name() -> None:
    assert input_runtime._format_prompt_prefix("dev", default_agent_name="dev") == "❯"
    assert input_runtime._format_prompt_prefix("default") == "default ❯"
    assert input_runtime._format_prompt_prefix("dev") == "dev ❯"


def test_echo_visible_command_escapes_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []
    monkeypatch.setattr(input_runtime, "rich_print", printed.append)

    input_runtime._echo_visible_command(
        stripped="/skill install [draft]",
        agent_name="agent [dev]",
        default_agent_name=None,
    )

    assert printed == [r"[dim]agent \[dev] ❯ /skill install \[draft][/dim]"]


@pytest.mark.asyncio
async def test_run_prompt_once_converts_prompt_input_interrupt_to_interrupt_command() -> None:
    class _Buffer:
        def __init__(self) -> None:
            self.accept_handler = None

    class _Session:
        def __init__(self) -> None:
            self.default_buffer = _Buffer()

        async def prompt_async(self, *_args, **_kwargs):
            raise PromptInputInterrupt()

    result = await input_runtime.run_prompt_once(
        session=cast("PromptSession", _Session()),
        agent_name="agent",
        default_agent_name="agent",
        default_buffer="",
        resolve_prompt_text=lambda: "❯ ",
        parse_special_input=lambda value: value,
    )

    assert type(result).__name__ == "InterruptCommand"


@pytest.mark.asyncio
async def test_run_prompt_once_converts_eof_to_eof_command() -> None:
    class _Buffer:
        def __init__(self) -> None:
            self.accept_handler = None

    class _Session:
        def __init__(self) -> None:
            self.default_buffer = _Buffer()

        async def prompt_async(self, *_args, **_kwargs):
            raise EOFError()

    result = await input_runtime.run_prompt_once(
        session=cast("PromptSession", _Session()),
        agent_name="agent",
        default_agent_name="agent",
        default_buffer="",
        resolve_prompt_text=lambda: "❯ ",
        parse_special_input=lambda value: value,
    )

    assert isinstance(result, EOFCommand)


@pytest.mark.asyncio
async def test_inline_toolbar_renders_compactly_and_clears_below() -> None:
    import io

    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.buffer import CompletionState
    from prompt_toolkit.completion import Completion, WordCompleter
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.document import Document
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.renderer import CPR_Support

    buf = io.StringIO()
    out = Vt100_Output(buf, get_size=lambda: Size(rows=40, columns=100))
    session = input_runtime.create_prompt_session(
        history=InMemoryHistory(),
        completer=WordCompleter(["/help", "/clear"]),
        lexer=None,
        multiline_filter=False,
        toolbar=lambda: " NORMAL | model ",
        style=input_runtime.build_prompt_style(),
    )
    session.app.renderer.output = out
    session.message = "❯ "
    session.reserve_space_for_menu = 6

    assert session.app.renderer.cpr_support == CPR_Support.NOT_SUPPORTED

    with set_app(session.app):
        session.app.renderer.render(session.app, session.app.layout)
        raw = buf.getvalue()
        assert session.app.renderer._last_screen is not None
        assert session.app.renderer._last_screen.height == 2
        assert "\x1b[6n" not in raw
        assert "\x1b[?2026h" in raw
        assert "\x1b[?2026l" in raw
        assert "\x1b[J" in raw
        assert "NORMAL | model" in raw

        # Empty CompletionState (in-flight threaded completion) must not expand height.
        buf.seek(0)
        buf.truncate(0)
        session.default_buffer.text = "hello"
        session.default_buffer.complete_state = CompletionState(Document("hello"), [])
        session.app.renderer.render(session.app, session.app.layout)
        assert session.app.renderer._last_screen.height == 2
        assert "\x1b[J" in buf.getvalue()

        # Non-empty completions expand to reserve menu space + toolbar.
        session.default_buffer.text = "/"
        session.default_buffer.complete_state = CompletionState(
            Document("/"), [Completion("/help")]
        )
        session.app.renderer.render(session.app, session.app.layout)
        assert session.app.renderer._last_screen.height == 7

        # Closing completions shrinks back to 2 and erases below.
        buf.seek(0)
        buf.truncate(0)
        session.default_buffer.complete_state = None
        session.app.renderer.render(session.app, session.app.layout)
        assert session.app.renderer._last_screen.height == 2
        assert "\x1b[J" in buf.getvalue()


@pytest.mark.asyncio
async def test_resilient_resize_handler_debounces_bursts() -> None:
    import asyncio
    from types import SimpleNamespace

    erases = 0
    redraws = 0

    class _Renderer:
        _waiting_for_cpr_futures = None
        _min_available_height = 20

        def erase(self, *, leave_alternate_screen: bool = True) -> None:
            nonlocal erases
            assert leave_alternate_screen is False
            erases += 1

    class _App:
        is_running = True
        renderer = _Renderer()

        def _on_resize(self) -> None:
            nonlocal redraws
            redraws += 1

    session = SimpleNamespace(app=_App())
    input_runtime._install_resilient_resize_handler(
        cast("PromptSession", session),
        debounce_seconds=0.02,
    )

    session.app._on_resize()
    session.app._on_resize()
    session.app._on_resize()
    assert erases == 1
    assert redraws == 0

    await asyncio.sleep(0.05)
    assert redraws == 1
    assert session.app.renderer._min_available_height == 0
