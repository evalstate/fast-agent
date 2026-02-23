from __future__ import annotations

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completion

from fast_agent.ui.prompt.keybindings import _accept_completion, _cycle_completion


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
