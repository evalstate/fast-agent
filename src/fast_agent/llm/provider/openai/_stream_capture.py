"""Stream capture utilities for OpenAI provider debugging.

When FAST_AGENT_LLM_TRACE environment variable is set, streaming chunks
are captured to files for debugging purposes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fast_agent.core.logging.logger import get_logger

_logger = get_logger(__name__)

STREAM_CAPTURE_ENABLED = bool(os.environ.get("FAST_AGENT_LLM_TRACE"))
STREAM_CAPTURE_DIR = Path("stream-debug")


def stream_capture_filename(turn: int) -> Path | None:
    """Generate filename for stream capture. Returns None if capture is disabled."""
    if not STREAM_CAPTURE_ENABLED:
        return None
    STREAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STREAM_CAPTURE_DIR / f"{timestamp}_turn{turn}"


def save_stream_request(filename_base: Path | None, arguments: dict[str, Any]) -> None:
    """Save the request arguments to a _request.json file."""
    if not filename_base:
        return
    try:
        request_file = filename_base.with_name(f"{filename_base.name}_request.json")
        with request_file.open("w") as handle:
            json.dump(arguments, handle, indent=2, default=str)
    except Exception as exc:
        _logger.debug(f"Failed to save stream request: {exc}")


def save_stream_chunk(filename_base: Path | None, chunk: Any) -> None:
    """Save a streaming chunk to file when capture mode is enabled."""
    if not filename_base:
        return
    try:
        chunk_file = filename_base.with_name(f"{filename_base.name}_chunks.jsonl")
        with chunk_file.open("a") as handle:
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else str(chunk)
            handle.write(json.dumps(chunk_dict, default=str) + "\n")
    except Exception as exc:
        _logger.debug(f"Failed to save stream chunk: {exc}")
