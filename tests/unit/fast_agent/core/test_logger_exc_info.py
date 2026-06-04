from typing import Any

from fast_agent.core.logging.events import EventContext, EventType
from fast_agent.core.logging.logger import Logger


class _RecordingLogger(Logger):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def event(
        self,
        etype: EventType,
        ename: str | None,
        message: str,
        context: EventContext | None,
        data: dict,
    ) -> None:
        self.events.append(
            {
                "type": etype,
                "name": ename,
                "message": message,
                "context": context,
                "data": data,
            }
        )


def test_error_removes_falsey_exc_info_from_event_data() -> None:
    logger = _RecordingLogger()

    logger.error("hello", exc_info=None, detail="kept")

    assert logger.events[0]["data"] == {"detail": "kept"}


def test_error_formats_exception_object_in_event_data() -> None:
    logger = _RecordingLogger()
    error = RuntimeError("boom")

    logger.error("hello", exc_info=error)

    payload = logger.events[0]["data"]
    assert payload["error"] == "boom"
    assert payload["error_type"] == "RuntimeError"
    assert "RuntimeError: boom" in payload["exception"]


def test_error_preserves_existing_error_fields() -> None:
    logger = _RecordingLogger()
    error = ValueError("actual")

    logger.error(
        "hello",
        exc_info=(ValueError, error, error.__traceback__),
        error="custom",
        error_type="CustomError",
    )

    payload = logger.events[0]["data"]
    assert payload["error"] == "custom"
    assert payload["error_type"] == "CustomError"
    assert "ValueError: actual" in payload["exception"]


def test_error_falls_back_for_invalid_exc_info_tuple() -> None:
    logger = _RecordingLogger()

    logger.error("hello", exc_info=("not-an-exception", "boom", None))

    payload = logger.events[0]["data"]
    assert payload == {"exception": "('not-an-exception', 'boom', None)"}
