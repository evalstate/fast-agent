from __future__ import annotations

import json
import warnings

import pytest

from fast_agent.llm.provider.openai._stream_capture import save_stream_chunk, save_stream_request


class _WarningChunk:
    def model_dump(self, **kwargs: object) -> dict[str, str]:
        _ = kwargs
        warnings.warn(
            "PydanticSerializationUnexpectedValue(Expected `ResponseFunctionWebSearch`)",
            UserWarning,
            stacklevel=1,
        )
        return {"type": "response.output_text.delta", "delta": "hello"}


class _LegacyChunk:
    def model_dump(self) -> dict[str, bool]:
        return {"legacy": True}


class _NonJsonPayload:
    pass


class _ChunkWithNonJsonPayload:
    def model_dump(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        return {"payload": _NonJsonPayload()}


@pytest.mark.unit
def test_save_stream_request_places_large_content_after_controls(tmp_path) -> None:
    filename_base = tmp_path / "capture"

    save_stream_request(
        filename_base,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "max",
            "model": "model",
            "max_tokens": 1024,
        },
    )

    payload = json.loads(filename_base.with_name("capture_request.json").read_text())
    assert list(payload) == ["max_tokens", "model", "reasoning_effort", "messages"]


@pytest.mark.unit
def test_save_stream_chunk_suppresses_pydantic_serialization_warning(tmp_path) -> None:
    filename_base = tmp_path / "capture"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_stream_chunk(filename_base, _WarningChunk())

    assert caught == []
    lines = filename_base.with_name("capture_chunks.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"type": "response.output_text.delta", "delta": "hello"}


@pytest.mark.unit
def test_save_stream_chunk_supports_model_dump_without_warnings_kw(tmp_path) -> None:
    filename_base = tmp_path / "capture"

    save_stream_chunk(filename_base, _LegacyChunk())

    lines = filename_base.with_name("capture_chunks.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"legacy": True}


@pytest.mark.unit
def test_save_stream_chunk_serializes_non_json_payloads(tmp_path) -> None:
    filename_base = tmp_path / "capture"

    save_stream_chunk(filename_base, _ChunkWithNonJsonPayload())

    lines = filename_base.with_name("capture_chunks.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["payload"].startswith(
        "<test_openai_stream_capture._NonJsonPayload object"
    )
