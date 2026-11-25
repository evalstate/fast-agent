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


class _FakeVt100Stdout:
    encoding = "utf-8"

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, data: str) -> int:
        self._chunks.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return "".join(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()


@pytest.mark.asyncio
async def test_resilient_resize_recovers_after_dropped_cpr_without_submit_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import time

    from prompt_toolkit import PromptSession
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.renderer import CPR_Support

    buf = _FakeVt100Stdout()
    output = Vt100_Output(cast("TextIO", buf), lambda: Size(rows=24, columns=80), enable_cpr=True)

    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(
            bottom_toolbar="TOOLBAR",
            erase_when_done=True,
            input=pipe,
            output=output,
        )
        input_runtime._install_resilient_resize_handler(session, debounce_seconds=0.0)
        renderer = session.app.renderer
        monkeypatch.setattr(renderer, "CPR_TIMEOUT", 0.05)

        async def _drive() -> float:
            try:
                # Let initial CPR time out so cpr_support latches to NOT_SUPPORTED
                # with a stale pending future.
                await asyncio.sleep(0.1)
                assert renderer.cpr_support == CPR_Support.NOT_SUPPORTED
                assert len(renderer._waiting_for_cpr_futures) == 1

                # A resize that also times out must not spam repeated CPR warnings.
                buf.clear()
                session.app._on_resize()
                assert "\x1b[6n" in buf.getvalue()
                await asyncio.sleep(0.1)
                assert renderer.cpr_support == CPR_Support.NOT_SUPPORTED
                assert "WARNING:" not in buf.getvalue()

                # Next resize re-probes; reply arrives -> toolbar renders and queue drains.
                buf.clear()
                session.app._on_resize()
                assert "\x1b[6n" in buf.getvalue()
                pipe.send_text("\x1b[5;1R")
                await asyncio.sleep(0.05)
                assert renderer.cpr_support == CPR_Support.SUPPORTED
                assert not renderer._waiting_for_cpr_futures
                assert "TOOLBAR" in buf.getvalue()

                # Once SUPPORTED, rapid resizes preserve in-flight CPR replies until
                # both arrive instead of draining prematurely.
                session.app._on_resize()
                session.app._on_resize()
                pipe.send_text("\x1b[5;1R\x1b[5;1R")
                await asyncio.sleep(0.05)
                assert not renderer._waiting_for_cpr_futures

                return time.perf_counter()
            finally:
                pipe.send_text("\r")

        driver = asyncio.create_task(_drive())
        try:
            result = await asyncio.wait_for(session.prompt_async("> "), timeout=5.0)
        finally:
            submitted_at = await driver
        elapsed = time.perf_counter() - submitted_at

    assert result == ""
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_resilient_resize_coalesces_rapid_resize_burst() -> None:
    import asyncio

    from prompt_toolkit import PromptSession
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    buf = _FakeVt100Stdout()
    output = Vt100_Output(cast("TextIO", buf), lambda: Size(rows=24, columns=80), enable_cpr=True)

    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(
            bottom_toolbar="TOOLBAR",
            erase_when_done=True,
            input=pipe,
            output=output,
        )
        input_runtime._install_resilient_resize_handler(session, debounce_seconds=0.03)
        renderer = session.app.renderer

        async def _drive() -> None:
            try:
                await asyncio.sleep(0.03)
                pipe.send_text("\x1b[5;1R")
                await asyncio.sleep(0.03)
                buf.clear()

                # Burst of 5 rapid resizes should coalesce into a single redraw/CPR probe.
                for _ in range(5):
                    session.app._on_resize()
                assert buf.getvalue() == ""

                await asyncio.sleep(0.06)
                assert buf.getvalue().count("\x1b[6n") == 1
                assert len(renderer._waiting_for_cpr_futures) == 1

                pipe.send_text("\x1b[5;1R")
                await asyncio.sleep(0.03)
                assert not renderer._waiting_for_cpr_futures
            finally:
                pipe.send_text("\r")

        driver = asyncio.create_task(_drive())
        try:
            result = await asyncio.wait_for(session.prompt_async("> "), timeout=5.0)
        finally:
            await driver

    assert result == ""
