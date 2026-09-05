from __future__ import annotations

from datetime import datetime

from fast_agent.llm.provider import stream_capture


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 1, 12, 34, 56, 789012, tzinfo=tz)


def test_stream_capture_filename_includes_microseconds(monkeypatch) -> None:
    monkeypatch.setattr(stream_capture, "STREAM_CAPTURE_ENABLED", True)
    monkeypatch.setattr(stream_capture, "datetime", _FixedDatetime)

    filename = stream_capture.stream_capture_filename(3, label="google_")

    assert filename is not None
    assert filename.name == "20260901_123456_789012_google_turn3"
