from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.llm.provider import stream_capture

if TYPE_CHECKING:
    from datetime import datetime

    import pytest


def test_stream_capture_filename_includes_microseconds(
    monkeypatch: pytest.MonkeyPatch, fixed_datetime: type[datetime]
) -> None:
    monkeypatch.setattr(stream_capture, "STREAM_CAPTURE_ENABLED", True)
    monkeypatch.setattr(stream_capture, "datetime", fixed_datetime)

    filename = stream_capture.stream_capture_filename(3, label="google_")

    assert filename is not None
    assert filename.name == "20260901_123456_789012_google_turn3"
