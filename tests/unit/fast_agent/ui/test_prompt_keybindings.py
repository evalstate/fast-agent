from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
        on_cycle_reasoning=lambda: events.append("reasoning"),
        on_cycle_verbosity=lambda: events.append("verbosity"),
        on_cycle_web_search=lambda: events.append("web_search"),
        on_cycle_web_fetch=lambda: events.append("web_fetch"),
    )

    for key, label in (
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
