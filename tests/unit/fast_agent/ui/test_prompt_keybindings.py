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


def _binding_for(kb: Any, key: Keys) -> Any:
    return next(binding for binding in kb.bindings if binding.keys == (key,))


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
