"""Tests for progress payload normalization helpers."""

from fast_agent.core.logging.progress_payloads import build_progress_payload
from fast_agent.event_progress import ProgressAction


def test_build_progress_payload_includes_expected_fields() -> None:
    payload = build_progress_payload(
        action=ProgressAction.TOOL_PROGRESS,
        agent_name="assistant",
        tool_name="execute",
        server_name="local",
        tool_use_id="call-1",
        tool_call_id="call-1",
        tool_event="stop",
        progress=1.0,
        total=1.0,
        details="completed",
    )

    assert payload == {
        "progress_action": ProgressAction.TOOL_PROGRESS,
        "agent_name": "assistant",
        "tool_name": "execute",
        "server_name": "local",
        "tool_use_id": "call-1",
        "tool_call_id": "call-1",
        "tool_event": "stop",
        "progress": 1.0,
        "total": 1.0,
        "details": "completed",
    }


def test_build_progress_payload_omits_none_and_merges_extra_fields() -> None:
    payload = build_progress_payload(
        action=ProgressAction.CALLING_TOOL,
        server_name="docs",
        extra={"resource_uri": "file://README.md"},
    )

    assert payload == {
        "progress_action": ProgressAction.CALLING_TOOL,
        "server_name": "docs",
        "resource_uri": "file://README.md",
    }

