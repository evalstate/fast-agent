from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from fast_agent.llm.provider.google._stream_capture import save_stream_chunk


class _WarningChunk:
    def model_dump(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        warnings.warn(
            "PydanticSerializationUnexpectedValue(Expected `Part`)",
            UserWarning,
            stacklevel=1,
        )
        return {"path": Path("example.txt"), "items": ({1, 2},)}


class _LegacyChunk:
    def model_dump(self, *, mode: str) -> dict[str, bool]:
        assert mode == "json"
        return {"legacy": True}


@pytest.mark.unit
def test_save_stream_chunk_suppresses_pydantic_serialization_warning(tmp_path: Path) -> None:
    filename_base = tmp_path / "capture"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        save_stream_chunk(filename_base, _WarningChunk())

    assert caught == []
    lines = filename_base.with_name("capture_chunks.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"items": [[1, 2]], "path": "example.txt"}


@pytest.mark.unit
def test_save_stream_chunk_supports_model_dump_without_warnings_kw(tmp_path: Path) -> None:
    filename_base = tmp_path / "capture"

    save_stream_chunk(filename_base, _LegacyChunk())

    lines = filename_base.with_name("capture_chunks.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"legacy": True}
