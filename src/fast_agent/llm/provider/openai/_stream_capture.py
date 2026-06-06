"""Stream capture utilities for OpenAI provider debugging.

When FAST_AGENT_LLM_TRACE environment variable is set, streaming chunks
are captured to files for debugging purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.llm.provider import stream_capture


def stream_capture_filename(turn: int) -> Path | None:
    """Generate filename for stream capture. Returns None if capture is disabled."""
    return stream_capture.stream_capture_filename(turn, label="")


def save_stream_request(filename_base: Path | None, arguments: dict[str, Any]) -> None:
    """Save the request arguments to a _request.json file."""
    stream_capture.save_stream_request(filename_base, arguments, logger_name=__name__)


def save_stream_chunk(filename_base: Path | None, chunk: Any) -> None:
    """Save a streaming chunk to file when capture mode is enabled."""
    stream_capture.save_stream_chunk(filename_base, chunk, logger_name=__name__)
