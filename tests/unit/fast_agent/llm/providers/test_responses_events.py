from __future__ import annotations

from fast_agent.llm.provider.openai.responses_events import (
    is_responses_failure_event,
    is_responses_reasoning_delta_event,
    is_responses_terminal_event,
    is_responses_text_delta_event,
)


def test_responses_terminal_events_are_explicit() -> None:
    assert is_responses_terminal_event("response.completed") is True
    assert is_responses_terminal_event("response.done") is True
    assert is_responses_terminal_event("response.incomplete") is True
    assert is_responses_terminal_event("response.failed") is False
    assert is_responses_terminal_event(None) is False


def test_responses_failure_events_are_explicit() -> None:
    assert is_responses_failure_event("error") is True
    assert is_responses_failure_event("response.failed") is True
    assert is_responses_failure_event("response.completed") is False
    assert is_responses_failure_event(None) is False


def test_responses_delta_events_are_explicit() -> None:
    assert is_responses_reasoning_delta_event("response.reasoning_summary_text.delta") is True
    assert is_responses_reasoning_delta_event("response.reasoning_summary.delta") is True
    assert is_responses_reasoning_delta_event("response.reasoning.delta") is False

    assert is_responses_text_delta_event("response.output_text.delta") is True
    assert is_responses_text_delta_event("response.text.delta") is True
    assert is_responses_text_delta_event("response.output_item.added") is False
