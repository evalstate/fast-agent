"""Shared OpenAI Responses stream event classifications."""

from __future__ import annotations

RESPONSES_TERMINAL_EVENT_TYPES = frozenset(
    {
        "response.completed",
        "response.done",
        "response.incomplete",
    }
)
RESPONSES_FAILURE_EVENT_TYPES = frozenset({"error", "response.failed"})
RESPONSES_REASONING_DELTA_EVENT_TYPES = frozenset(
    {
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary.delta",
    }
)
RESPONSES_TEXT_DELTA_EVENT_TYPES = frozenset(
    {
        "response.output_text.delta",
        "response.text.delta",
    }
)


def is_responses_terminal_event(event_type: object) -> bool:
    return event_type in RESPONSES_TERMINAL_EVENT_TYPES


def is_responses_failure_event(event_type: object) -> bool:
    return event_type in RESPONSES_FAILURE_EVENT_TYPES


def is_responses_reasoning_delta_event(event_type: object) -> bool:
    return event_type in RESPONSES_REASONING_DELTA_EVENT_TYPES


def is_responses_text_delta_event(event_type: object) -> bool:
    return event_type in RESPONSES_TEXT_DELTA_EVENT_TYPES
