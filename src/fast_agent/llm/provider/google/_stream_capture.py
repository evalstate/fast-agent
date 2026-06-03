"""Google stream capture utilities for provider debugging."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.llm.provider import stream_capture


def stream_capture_filename(turn: int) -> Path | None:
    return stream_capture.stream_capture_filename(turn, label="google_")


def save_stream_request(filename_base: Path | None, arguments: dict[str, Any]) -> None:
    stream_capture.save_stream_request(filename_base, arguments, logger_name=__name__)


def save_stream_chunk(filename_base: Path | None, chunk: Any) -> None:
    stream_capture.save_stream_chunk(filename_base, chunk, logger_name=__name__)
