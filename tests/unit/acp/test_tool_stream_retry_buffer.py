from fast_agent.acp.server.session_runtime import _ACPToolStreamAttemptBuffer
from fast_agent.llm.stream_types import StreamChunk


def test_failed_attempt_tool_events_are_discarded_before_commit() -> None:
    forwarded: list[tuple[str, dict[str, object] | None]] = []

    def record(event_type: str, payload: dict[str, object] | None) -> None:
        forwarded.append((event_type, payload))

    buffer = _ACPToolStreamAttemptBuffer(record)

    buffer.handle_tool_event(
        "start",
        {"tool_name": "first_tool", "tool_use_id": "failed-attempt"},
    )
    buffer.handle_tool_event("delta", {"tool_use_id": "failed-attempt", "chunk": "{}"})
    buffer.handle_stream_chunk(StreamChunk(event="rollback"))

    buffer.handle_tool_event(
        "start",
        {"tool_name": "second_tool", "tool_use_id": "successful-attempt"},
    )
    buffer.handle_tool_event(
        "delta",
        {"tool_use_id": "successful-attempt", "chunk": '{"path":'},
    )
    buffer.handle_tool_event("stop", {"tool_use_id": "successful-attempt"})
    buffer.handle_stream_chunk(StreamChunk(event="commit"))

    assert forwarded == [
        (
            "start",
            {"tool_name": "second_tool", "tool_use_id": "successful-attempt"},
        ),
        (
            "delta",
            {"tool_use_id": "successful-attempt", "chunk": '{"path":'},
        ),
        ("stop", {"tool_use_id": "successful-attempt"}),
    ]
