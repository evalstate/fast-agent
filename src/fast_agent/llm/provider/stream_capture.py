"""Shared stream capture utilities for provider debugging."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fast_agent.core.logging.logger import get_logger

STREAM_CAPTURE_ENABLED = bool(os.environ.get("FAST_AGENT_LLM_TRACE"))
STREAM_CAPTURE_DIR = Path("stream-debug")


@runtime_checkable
class ModelDumpable(Protocol):
    def model_dump(self, *args: Any, **kwargs: Any) -> Any: ...


def stream_capture_filename(turn: int, *, label: str) -> Path | None:
    if not STREAM_CAPTURE_ENABLED:
        return None
    STREAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STREAM_CAPTURE_DIR / f"{timestamp}_{label}turn{turn}"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, ModelDumpable):
        return _model_dump_jsonable(value)
    return str(value)


def save_stream_request(
    filename_base: Path | None,
    arguments: dict[str, Any],
    *,
    logger_name: str,
) -> None:
    if filename_base is None:
        return
    logger = get_logger(logger_name)
    try:
        request_file = filename_base.with_name(f"{filename_base.name}_request.json")
        request_file.write_text(
            json.dumps(jsonable(arguments), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug(f"Failed to save stream request: {exc}")


def save_stream_chunk(
    filename_base: Path | None,
    chunk: Any,
    *,
    logger_name: str,
) -> None:
    if filename_base is None:
        return
    logger = get_logger(logger_name)
    try:
        chunk_file = filename_base.with_name(f"{filename_base.name}_chunks.jsonl")
        with chunk_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(chunk), sort_keys=True) + "\n")
    except Exception as exc:
        logger.debug(f"Failed to save stream chunk: {exc}")


def _model_dump_jsonable(value: ModelDumpable) -> Any:
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
            dumped = value.model_dump(mode="json", warnings="none")
        except TypeError:
            try:
                dumped = value.model_dump(mode="json")
            except TypeError:
                try:
                    dumped = value.model_dump(warnings="none")
                except TypeError:
                    dumped = value.model_dump()
        except Exception:
            return str(value)
    return jsonable(dumped)
