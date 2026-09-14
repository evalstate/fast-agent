from __future__ import annotations

from datetime import datetime

from fast_agent.llm.provider.anthropic import llm_anthropic
from fast_agent.llm.provider.anthropic.llm_anthropic import (
    _serialize_for_trace,
    _stream_capture_filename,
)


class _Dumpable:
    def model_dump(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {"nested": _NestedDumpable(), "items": [_NestedDumpable()]}


class _NestedDumpable:
    def model_dump(self, **kwargs: object) -> dict[str, str]:
        del kwargs
        return {"value": "ok"}


class _BrokenDumpable:
    def model_dump(self, **kwargs: object) -> dict[str, str]:
        del kwargs
        raise RuntimeError("boom")


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 1, 12, 34, 56, 789012, tzinfo=tz)


def test_stream_capture_filename_includes_microseconds(monkeypatch) -> None:
    monkeypatch.setattr(llm_anthropic, "STREAM_CAPTURE_ENABLED", True)
    monkeypatch.setattr(llm_anthropic, "datetime", _FixedDatetime)

    filename = _stream_capture_filename(3)

    assert filename is not None
    assert filename.name == "anthropic_20260901_123456_789012_turn3"


def test_serialize_for_trace_recurses_model_dump_payloads() -> None:
    assert _serialize_for_trace({"chunk": _Dumpable()}) == {
        "chunk": {
            "nested": {"value": "ok"},
            "items": [{"value": "ok"}],
        }
    }


def test_serialize_for_trace_falls_back_for_broken_model_dump() -> None:
    serialized = _serialize_for_trace(_BrokenDumpable())

    assert serialized.startswith("<test_stream_capture._BrokenDumpable object")
