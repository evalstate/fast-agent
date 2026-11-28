from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completion
from prompt_toolkit.keys import Keys

from fast_agent.ui.prompt.keybindings import (
    PromptInputInterrupt,
    _accept_completion,
    _cycle_completion,
    create_keybindings,
    paste_clipboard_image_attachment_into_buffer,
)


def _buffer_with_completions() -> Buffer:
    buffer = Buffer()
    buffer._set_completions(
        [
            Completion("first"),
            Completion("second"),
        ]
    )
    return buffer


def test_cycle_completion_forward_wraps() -> None:
    buffer = _buffer_with_completions()

    assert _cycle_completion(buffer, backwards=False) is True
    assert buffer.complete_state is not None
    assert buffer.complete_state.complete_index == 0
    assert buffer.text == "first"

    assert _cycle_completion(buffer, backwards=False) is True
    assert buffer.complete_state is not None
    assert buffer.complete_state.complete_index == 1
    assert buffer.text == "second"

    assert _cycle_completion(buffer, backwards=False) is True
    assert buffer.complete_state is not None
    assert buffer.complete_state.complete_index == 0
    assert buffer.text == "first"


def test_cycle_completion_backward_wraps() -> None:
    buffer = _buffer_with_completions()

    assert _cycle_completion(buffer, backwards=True) is True
    assert buffer.complete_state is not None
    assert buffer.complete_state.complete_index == 1
    assert buffer.text == "second"

    assert _cycle_completion(buffer, backwards=True) is True
    assert buffer.complete_state is not None
    assert buffer.complete_state.complete_index == 0
    assert buffer.text == "first"


def test_accept_completion_selects_first_when_none_selected() -> None:
    buffer = _buffer_with_completions()

    assert _accept_completion(buffer) is True
    assert buffer.complete_state is None
    assert buffer.text == "first"


def test_accept_completion_preserves_selected_item() -> None:
    buffer = _buffer_with_completions()
    assert buffer.complete_state is not None

    buffer.go_to_completion(1)

    assert _accept_completion(buffer) is True
    assert buffer.complete_state is None
    assert buffer.text == "second"


def test_completion_helpers_return_false_without_active_menu() -> None:
    buffer = Buffer()

    assert _cycle_completion(buffer, backwards=False) is False
    assert _accept_completion(buffer) is False


def _binding_for(kb: Any, *keys: Keys | str) -> Any:
    return next(binding for binding in kb.bindings if binding.keys == keys)


def test_shift_tab_cycles_service_tier_when_completion_menu_closed() -> None:
    cycles: list[str] = []
    invalidations: list[str] = []

    class _App:
        def invalidate(self) -> None:
            invalidations.append("invalidate")

    kb = create_keybindings(on_cycle_service_tier=lambda: cycles.append("cycled"))
    binding = _binding_for(kb, Keys.BackTab)

    binding.handler(SimpleNamespace(current_buffer=Buffer(), app=_App()))

    assert cycles == ["cycled"]
    assert invalidations == ["invalidate"]


def test_shift_tab_keeps_completion_navigation_priority() -> None:
    cycles: list[str] = []
    buffer = _buffer_with_completions()
    kb = create_keybindings(on_cycle_service_tier=lambda: cycles.append("cycled"))
    binding = _binding_for(kb, Keys.BackTab)

    binding.handler(SimpleNamespace(current_buffer=buffer, app=None))

    assert cycles == []
    assert buffer.complete_state is not None
    assert buffer.complete_state.complete_index == 1
    assert buffer.text == "second"


def test_function_key_callbacks_fire_when_configured() -> None:
    events: list[str] = []

    class _App:
        def invalidate(self) -> None:
            events.append("invalidate")

    kb = create_keybindings(
        on_cycle_agent_mode=lambda: events.append("agent_mode"),
        on_cycle_reasoning=lambda: events.append("reasoning"),
        on_cycle_verbosity=lambda: events.append("verbosity"),
        on_cycle_web_search=lambda: events.append("web_search"),
        on_cycle_web_fetch=lambda: events.append("web_fetch"),
    )

    for key, label in (
        (Keys.F5, "agent_mode"),
        (Keys.F6, "reasoning"),
        (Keys.F7, "verbosity"),
        (Keys.F8, "web_search"),
        (Keys.F9, "web_fetch"),
    ):
        binding = _binding_for(kb, key)
        binding.handler(SimpleNamespace(current_buffer=Buffer(), app=_App()))
        assert label in events


def test_ctrl_c_binding_exits_with_prompt_input_interrupt() -> None:
    class _App:
        def __init__(self) -> None:
            self.exception: BaseException | None = None

        def exit(self, *, exception: BaseException | None = None) -> None:
            self.exception = exception

    app = _App()
    kb = create_keybindings()
    binding = _binding_for(kb, Keys.ControlC)

    binding.handler(SimpleNamespace(current_buffer=Buffer(), app=app))

    assert isinstance(app.exception, PromptInputInterrupt)


@pytest.mark.asyncio
async def test_alt_v_pastes_clipboard_image_as_attachment(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "clipboard.png"
    image_path.write_bytes(b"png")
    buffer = Buffer()
    buffer.text = "describe this"

    class _PastedImage:
        path = image_path
        width = 12
        height = 34

    async def fake_to_thread(*_args: Any, **_kwargs: Any) -> _PastedImage:
        return _PastedImage()

    monkeypatch.setattr("fast_agent.ui.prompt.keybindings.run_in_thread", fake_to_thread)
    await paste_clipboard_image_attachment_into_buffer(buffer)

    assert buffer.text.startswith("describe this ^file:")
    assert str(image_path) in buffer.text
    assert buffer.text.endswith(" ")
    assert buffer.cursor_position == len(buffer.text)


@pytest.mark.asyncio
async def test_clipboard_image_paste_error_prints_bracketed_text_literally(
    monkeypatch,
    capsys,
) -> None:
    invalidations: list[str] = []

    class _App:
        def invalidate(self) -> None:
            invalidations.append("invalidate")

    async def fake_to_thread(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("[draft] image unavailable")

    monkeypatch.setattr("fast_agent.ui.prompt.keybindings.run_in_thread", fake_to_thread)

    buffer = Buffer()
    buffer.text = "describe this"
    await paste_clipboard_image_attachment_into_buffer(buffer, app_ref=_App())

    assert buffer.text == "describe this"
    assert invalidations == ["invalidate"]
    assert "Failed to paste clipboard image: [draft] image unavailable" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_clipboard_no_image_uses_toolbar_notice_without_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fast_agent.ui.prompt import input as prompt_input
    from fast_agent.ui.prompt.clipboard_image import ClipboardNoImageError

    invalidations: list[str] = []

    class _App:
        def invalidate(self) -> None:
            invalidations.append("invalidate")

    async def fake_to_thread(*_args: Any, **_kwargs: Any) -> None:
        raise ClipboardNoImageError("no image on clipboard")

    monkeypatch.setattr("fast_agent.ui.prompt.keybindings.run_in_thread", fake_to_thread)
    monkeypatch.setattr(prompt_input, "_copy_notice", None)
    monkeypatch.setattr(prompt_input, "_copy_notice_until", 0.0)

    buffer = Buffer()
    buffer.text = "describe this"
    await paste_clipboard_image_attachment_into_buffer(buffer, app_ref=_App())

    assert buffer.text == "describe this"
    assert invalidations == ["invalidate"]
    assert prompt_input._copy_notice == "NO IMAGE"
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_clipboard_paste_error_uses_run_in_terminal_when_app_is_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _App:
        def invalidate(self) -> None:
            raise AssertionError("run_in_terminal handles redraw")

    app = _App()
    ran_in_terminal: list[bool] = []

    async def fake_to_thread(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("clipboard backend crashed")

    async def fake_run_in_terminal(func: Any) -> None:
        ran_in_terminal.append(True)
        func()

    monkeypatch.setattr("fast_agent.ui.prompt.keybindings.run_in_thread", fake_to_thread)
    monkeypatch.setattr("fast_agent.ui.prompt.keybindings.get_app_or_none", lambda: app)
    monkeypatch.setattr("fast_agent.ui.prompt.keybindings.run_in_terminal", fake_run_in_terminal)

    buffer = Buffer()
    await paste_clipboard_image_attachment_into_buffer(buffer, app_ref=app)

    assert ran_in_terminal == [True]
    assert "Failed to paste clipboard image: clipboard backend crashed" in capsys.readouterr().out


def test_clipboard_image_bindings_are_registered() -> None:
    kb = create_keybindings(enable_clipboard_image_paste=True)

    assert any(item.keys == (Keys.Escape, Keys.ControlV) for item in kb.bindings)
    assert any(item.keys == (Keys.Escape, "v") for item in kb.bindings)


def test_clipboard_image_bindings_are_not_registered_without_vision_support() -> None:
    kb = create_keybindings(enable_clipboard_image_paste=False)

    assert not any(item.keys == (Keys.Escape, Keys.ControlV) for item in kb.bindings)
    assert not any(item.keys == (Keys.Escape, "v") for item in kb.bindings)


@pytest.mark.asyncio
async def test_ctrl_y_writes_osc52_inside_herdr_without_requiring_pyperclip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64

    import pyperclip

    from fast_agent.ui.prompt import input as prompt_input

    class _TtyBuffer:
        def __init__(self) -> None:
            self.writes: list[str] = []

        def isatty(self) -> bool:
            return True

        def write(self, data: str) -> int:
            self.writes.append(data)
            return len(data)

        def flush(self) -> None:
            pass

    tty = _TtyBuffer()
    invalidations: list[str] = []

    class _App:
        def invalidate(self) -> None:
            invalidations.append("invalidate")

    def broken_copy(_text: str) -> None:
        raise FileNotFoundError("clip.exe")

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("HC_TERMINAL_ID", "t1m2")
    monkeypatch.setenv("HC_TERMINAL_SOCKET", "/run/hc.sock")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr("sys.stdout", tty)
    monkeypatch.setattr(pyperclip, "copy", broken_copy)
    monkeypatch.setattr(prompt_input, "_last_copyable_output", "hello from herdr")
    monkeypatch.setattr(prompt_input, "_copy_notice", None)

    kb = create_keybindings()
    binding = _binding_for(kb, Keys.ControlY)
    await binding.handler(SimpleNamespace(current_buffer=Buffer(), app=_App()))

    expected_b64 = base64.b64encode(b"hello from herdr").decode("ascii")
    assert tty.writes == [f"\x1b]52;c;{expected_b64}\x07"]
    assert prompt_input._copy_notice == "COPIED"
    assert invalidations == ["invalidate"]

    tty.writes.clear()
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    await binding.handler(SimpleNamespace(current_buffer=Buffer(), app=_App()))
    assert tty.writes == [f"\x1bPtmux;\x1b\x1b]52;c;{expected_b64}\x07\x1b\\"]


@pytest.mark.asyncio
async def test_ctrl_y_skips_tool_only_assistant_turns_when_falling_back_to_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyperclip

    from fast_agent.ui.prompt import input as prompt_input

    copied: list[str] = []

    class _Msg:
        def __init__(self, role: str, text: str | None) -> None:
            self.role = role
            self._text = text

        def last_text(self) -> str | None:
            return self._text

    class _Provider:
        def _agent(self, _name: str) -> Any:
            return SimpleNamespace(
                message_history=[
                    _Msg("assistant", "earlier answer"),
                    _Msg("assistant", None),
                ]
            )

    class _NonTty:
        def isatty(self) -> bool:
            return False

    class _App:
        def invalidate(self) -> None:
            pass

    monkeypatch.setattr("sys.stdout", _NonTty())
    monkeypatch.setattr(pyperclip, "copy", lambda text: copied.append(text))
    monkeypatch.setattr(prompt_input, "_last_copyable_output", None)

    kb = create_keybindings(agent_provider=cast("AgentApp", _Provider()), agent_name="main")
    binding = _binding_for(kb, Keys.ControlY)
    await binding.handler(SimpleNamespace(current_buffer=Buffer(), app=_App()))

    assert copied == ["earlier answer"]


@pytest.mark.asyncio
async def test_clear_terminal_screen_resets_scroll_region_and_drains_cpr() -> None:
    import asyncio
    import io
    from collections import deque

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.output.vt100 import Vt100_Output

    from fast_agent.ui.prompt.keybindings import _clear_terminal_screen

    pending = asyncio.get_running_loop().create_future()
    buf = io.StringIO()
    vt_output = Vt100_Output(buf, lambda: Size(rows=24, columns=80))
    invalidated = 0
    erased = 0

    class _VtRenderer:
        def __init__(self) -> None:
            self._waiting_for_cpr_futures = deque([pending])
            self._min_available_height = 30
            self.output = vt_output

        def erase(self, *, leave_alternate_screen: bool = True) -> None:
            nonlocal erased
            assert leave_alternate_screen is False
            erased += 1

    class _VtApp:
        renderer = _VtRenderer()

        def invalidate(self) -> None:
            nonlocal invalidated
            invalidated += 1

    vt_app = _VtApp()
    _clear_terminal_screen(SimpleNamespace(app=vt_app), None)

    raw = buf.getvalue()
    assert pending.done()
    assert erased == 1
    assert "\x1b[r\x1b[0m" in raw
    assert "\x1b[24;1H" + "\n" * 24 in raw
    assert "\x1b[2J" in raw
    assert "\x1b[6n" not in raw
    assert vt_app.renderer._min_available_height == 0
    assert invalidated == 1

    # Non-VT100 outputs delegate to renderer.clear() without emitting raw ANSI.
    cleared = 0
    dummy_invalidated = 0

    class _DummyRenderer:
        def __init__(self) -> None:
            self._waiting_for_cpr_futures: deque[asyncio.Future[None]] = deque()
            self._min_available_height = 12
            self.output = DummyOutput()

        def clear(self) -> None:
            nonlocal cleared
            cleared += 1

    class _DummyApp:
        renderer = _DummyRenderer()

        def invalidate(self) -> None:
            nonlocal dummy_invalidated
            dummy_invalidated += 1

    dummy_app = _DummyApp()
    _clear_terminal_screen(SimpleNamespace(app=dummy_app), None)
    assert cleared == 1
    assert dummy_app.renderer._min_available_height == 0
    assert dummy_invalidated == 1


@pytest.mark.asyncio
async def test_alt_r_clears_screen_and_reviews_latest_history_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import io
    from collections import deque

    from mcp.types import TextContent
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.output.vt100 import Vt100_Output

    from fast_agent.types import PromptMessageExtended
    from fast_agent.ui.prompt import keybindings as keybindings_module
    from fast_agent.ui.prompt.keybindings import create_keybindings

    turn_1_user = PromptMessageExtended(
        role="user", content=[TextContent(type="text", text="first question")]
    )
    turn_1_assistant = PromptMessageExtended(
        role="assistant", content=[TextContent(type="text", text="first answer")]
    )
    turn_2_user = PromptMessageExtended(
        role="user", content=[TextContent(type="text", text="second question")]
    )
    turn_2_assistant = PromptMessageExtended(
        role="assistant", content=[TextContent(type="text", text="second answer")]
    )

    class _Provider:
        def _agent(self, _name: str) -> Any:
            return SimpleNamespace(
                name="test-agent",
                context=None,
                message_history=[turn_1_user, turn_1_assistant, turn_2_user, turn_2_assistant],
            )

    calls: list[dict[str, object]] = []

    async def _fake_display_history_turn(
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        config: object,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        calls.append(
            {
                "agent_name": agent_name,
                "turn": turn,
                "config": config,
                "turn_index": turn_index,
                "total_turns": total_turns,
            }
        )

    monkeypatch.setattr(keybindings_module, "display_history_turn", _fake_display_history_turn)

    pending = asyncio.get_running_loop().create_future()
    buf = io.StringIO()
    vt_output = Vt100_Output(buf, lambda: Size(rows=24, columns=80))
    invalidated = 0

    class _VtRenderer:
        def __init__(self) -> None:
            self._waiting_for_cpr_futures = deque([pending])
            self._min_available_height = 20
            self.output = vt_output

    class _VtApp:
        renderer = _VtRenderer()

        def invalidate(self) -> None:
            nonlocal invalidated
            invalidated += 1

    kb = create_keybindings(agent_provider=cast("AgentApp", _Provider()), agent_name="test-agent")
    binding = _binding_for(kb, Keys.Escape, "r")

    vt_app = _VtApp()
    await binding.handler(SimpleNamespace(current_buffer=Buffer(), app=vt_app))

    raw = buf.getvalue()
    assert "[r[0m" in raw
    assert "[2J" in raw
    assert len(calls) == 1
    assert calls[0]["agent_name"] == "test-agent"
    assert calls[0]["turn"] == [turn_2_user, turn_2_assistant]
    assert calls[0]["turn_index"] == 2
    assert calls[0]["total_turns"] == 2
    assert pending.done()
    assert vt_app.renderer._min_available_height == 0
    assert invalidated == 1

    # Non-VT100 outputs (e.g. Windows10_Output / Win32Output) still clear the viewport.
    fallback_ops: list[str] = []

    class _FallbackOutput:
        def erase_screen(self) -> None:
            fallback_ops.append("erase_screen")

        def cursor_goto(self, row: int = 0, column: int = 0) -> None:
            fallback_ops.append(f"cursor_goto:{row},{column}")

        def flush(self) -> None:
            fallback_ops.append("flush")

    class _FallbackRenderer:
        def __init__(self) -> None:
            self._waiting_for_cpr_futures: deque[asyncio.Future[None]] = deque()
            self._min_available_height = 10
            self.output = _FallbackOutput()

    class _FallbackApp:
        renderer = _FallbackRenderer()

        def invalidate(self) -> None:
            pass

    fallback_app = _FallbackApp()
    await binding.handler(SimpleNamespace(current_buffer=Buffer(), app=fallback_app))
    assert fallback_ops == ["erase_screen", "cursor_goto:0,0", "flush"]
    assert fallback_app.renderer._min_available_height == 0


@pytest.mark.asyncio
async def test_alt_r_with_empty_history_shows_notice_without_clearing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.output.vt100 import Vt100_Output

    from fast_agent.ui.prompt import input as prompt_input
    from fast_agent.ui.prompt import keybindings as keybindings_module
    from fast_agent.ui.prompt.keybindings import create_keybindings

    class _Provider:
        def _agent(self, _name: str) -> Any:
            return SimpleNamespace(name="test-agent", context=None, message_history=[])

    called = False

    async def _fake_display_history_turn(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(keybindings_module, "display_history_turn", _fake_display_history_turn)

    buf = io.StringIO()
    vt_output = Vt100_Output(buf, lambda: Size(rows=24, columns=80))
    invalidated = 0

    class _VtRenderer:
        output = vt_output

    class _VtApp:
        renderer = _VtRenderer()

        def invalidate(self) -> None:
            nonlocal invalidated
            invalidated += 1

    kb = create_keybindings(agent_provider=cast("AgentApp", _Provider()), agent_name="test-agent")
    binding = _binding_for(kb, Keys.Escape, "r")

    await binding.handler(SimpleNamespace(current_buffer=Buffer(), app=_VtApp()))

    assert called is False
    assert buf.getvalue() == ""
    assert prompt_input._copy_notice == "NO HISTORY"
    assert invalidated == 1
