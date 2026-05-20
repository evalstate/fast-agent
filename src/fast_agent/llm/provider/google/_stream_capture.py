"""Google stream capture utilities for provider debugging."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.trace import llm_trace_enabled

_logger = get_logger(__name__)

STREAM_CAPTURE_DIR = Path("stream-debug")


def stream_capture_filename(turn: int) -> Path | None:
    if not llm_trace_enabled():
        return None
    STREAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STREAM_CAPTURE_DIR / f"{timestamp}_google_turn{turn}"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*PydanticSerializationUnexpectedValue.*",
                category=UserWarning,
            )
            try:
                return _jsonable(model_dump(mode="json", warnings="none"))
            except TypeError:
                return _jsonable(model_dump(mode="json"))
            except Exception:
                return str(value)
    return str(value)


def save_stream_request(filename_base: Path | None, arguments: dict[str, Any]) -> None:
    if filename_base is None:
        return
    try:
        request_file = filename_base.with_name(f"{filename_base.name}_request.json")
        request_file.write_text(
            json.dumps(_jsonable(arguments), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        _logger.debug(f"Failed to save Google stream request: {exc}")


def save_stream_chunk(filename_base: Path | None, chunk: Any) -> None:
    if filename_base is None:
        return
    try:
        chunk_file = filename_base.with_name(f"{filename_base.name}_chunks.jsonl")
        with chunk_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(chunk), sort_keys=True) + "\n")
    except Exception as exc:
        _logger.debug(f"Failed to save Google stream chunk: {exc}")
