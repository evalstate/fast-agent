from __future__ import annotations

from fast_agent.llm.provider.anthropic.llm_anthropic import _serialize_for_trace


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
