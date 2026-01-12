from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)

from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.stream_types import StreamChunk

_logger = get_logger(__name__)

STREAM_CAPTURE_ENABLED = bool(os.environ.get("FAST_AGENT_LLM_TRACE"))
STREAM_CAPTURE_DIR = Path("stream-debug")


def _stream_capture_filename(turn: int) -> Path | None:
    """Generate filename for stream capture. Returns None if capture is disabled."""
    if not STREAM_CAPTURE_ENABLED:
        return None
    STREAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return STREAM_CAPTURE_DIR / f"{timestamp}_turn{turn}"


def _save_stream_request(filename_base: Path | None, arguments: dict[str, Any]) -> None:
    """Save the request arguments to a _request.json file."""
    if not filename_base:
        return
    try:
        request_file = filename_base.with_name(f"{filename_base.name}_request.json")
        with request_file.open("w") as handle:
            json.dump(arguments, handle, indent=2, default=str)
    except Exception as exc:
        _logger.debug(f"Failed to save stream request: {exc}")


def _save_stream_chunk(filename_base: Path | None, chunk: Any) -> None:
    """Save a streaming chunk to file when capture mode is enabled."""
    if not filename_base:
        return
    try:
        chunk_file = filename_base.with_name(f"{filename_base.name}.jsonl")
        try:
            payload: Any = chunk.model_dump()
        except Exception:
            payload = str(chunk)

        with chunk_file.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception as exc:
        _logger.debug(f"Failed to save stream chunk: {exc}")


class ResponsesStreamingMixin:
    if TYPE_CHECKING:
        from fast_agent.core.logging.logger import Logger

        logger: Logger
        name: str | None

        def _notify_stream_listeners(self, chunk: StreamChunk) -> None: ...

        def _notify_tool_stream_listeners(
            self, event_type: str, payload: dict[str, Any] | None = None
        ) -> None: ...

        def _update_streaming_progress(
            self, chunk: str, model: str, current_total: int
        ) -> int: ...

        def chat_turn(self) -> int: ...

    async def _process_stream(
        self, stream: Any, model: str, capture_filename: Path | None
    ) -> tuple[Any, list[str]]:
        estimated_tokens = 0
        reasoning_chars = 0
        reasoning_segments: list[str] = []
        tool_streams: dict[int, dict[str, Any]] = {}
        notified_tool_indices: set[int] = set()
        final_response: Any | None = None

        async for event in stream:
            _save_stream_chunk(capture_filename, event)
            if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
                if event.delta:
                    reasoning_segments.append(event.delta)
                    self._notify_stream_listeners(
                        StreamChunk(text=event.delta, is_reasoning=True)
                    )
                    reasoning_chars += len(event.delta)
                    await self._emit_streaming_progress(
                        model=f"{model} (summary)",
                        new_total=reasoning_chars,
                        type=ProgressAction.THINKING,
                    )
                continue

            if isinstance(event, ResponseTextDeltaEvent):
                if event.delta:
                    self._notify_stream_listeners(
                        StreamChunk(text=event.delta, is_reasoning=False)
                    )
                    estimated_tokens = self._update_streaming_progress(
                        event.delta, model, estimated_tokens
                    )
                    self._notify_tool_stream_listeners(
                        "text",
                        {
                            "chunk": event.delta,
                        },
                    )
                continue

            event_type = getattr(event, "type", None)
            if event_type in {"response.completed", "response.incomplete"}:
                final_response = getattr(event, "response", None) or final_response
                continue
            if event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    index = getattr(event, "output_index", None)
                    if index is None:
                        continue
                    tool_info = {
                        "tool_name": getattr(item, "name", None),
                        "tool_use_id": getattr(item, "call_id", None)
                        or getattr(item, "id", None),
                        "notified": False,
                    }
                    tool_streams[index] = tool_info
                    if tool_info["tool_name"] and tool_info["tool_use_id"]:
                        self._notify_tool_stream_listeners(
                            "start",
                            {
                                "tool_name": tool_info["tool_name"],
                                "tool_use_id": tool_info["tool_use_id"],
                                "index": index,
                            },
                        )
                        self.logger.info(
                            "Model started streaming tool call",
                            data={
                                "progress_action": ProgressAction.CALLING_TOOL,
                                "agent_name": self.name,
                                "model": model,
                                "tool_name": tool_info["tool_name"],
                                "tool_use_id": tool_info["tool_use_id"],
                                "tool_event": "start",
                            },
                        )
                        tool_info["notified"] = True
                        notified_tool_indices.add(index)
                continue

            if event_type == "response.function_call_arguments.delta":
                index = getattr(event, "output_index", None)
                if index is None:
                    continue
                tool_info = tool_streams.get(index, {})
                self._notify_tool_stream_listeners(
                    "delta",
                    {
                        "tool_name": tool_info.get("tool_name"),
                        "tool_use_id": tool_info.get("tool_use_id"),
                        "index": index,
                        "chunk": getattr(event, "delta", None),
                    },
                )
                continue

            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) != "function_call":
                    continue
                index = getattr(event, "output_index", None)
                tool_info = tool_streams.pop(index, {}) if index is not None else {}
                tool_name = getattr(item, "name", None) or tool_info.get("tool_name")
                tool_use_id = (
                    getattr(item, "call_id", None)
                    or getattr(item, "id", None)
                    or tool_info.get("tool_use_id")
                )
                if index is None:
                    index = -1
                self._notify_tool_stream_listeners(
                    "stop",
                    {
                        "tool_name": tool_name,
                        "tool_use_id": tool_use_id,
                        "index": index,
                    },
                )
                self.logger.info(
                    "Model finished streaming tool call",
                    data={
                        "progress_action": ProgressAction.CALLING_TOOL,
                        "agent_name": self.name,
                        "model": model,
                        "tool_name": tool_name,
                        "tool_use_id": tool_use_id,
                        "tool_event": "stop",
                    },
                )
                if index >= 0:
                    notified_tool_indices.add(index)
                continue

        if final_response is None:
            try:
                final_response = await stream.get_final_response()
            except Exception as exc:
                self.logger.warning("Failed to fetch final Responses payload", exc_info=exc)
                raise

        usage = getattr(final_response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            token_str = str(output_tokens).rjust(5)
            data = {
                "progress_action": ProgressAction.STREAMING,
                "model": model,
                "agent_name": self.name,
                "chat_turn": self.chat_turn(),
                "details": token_str.strip(),
            }
            self.logger.info("Streaming progress", data=data)
            self.logger.info(
                f"Streaming complete - Model: {model}, Input tokens: {input_tokens}, Output tokens: {output_tokens}"
            )

        output_items = list(getattr(final_response, "output", []) or [])
        self._emit_tool_notification_fallback(
            output_items,
            notified_tool_indices,
            model=model,
        )

        return final_response, reasoning_segments

    def _emit_tool_notification_fallback(
        self,
        output_items: list[Any],
        notified_indices: set[int],
        *,
        model: str,
    ) -> None:
        """Emit start/stop notifications when streaming metadata was missing."""
        if not output_items:
            return

        for index, item in enumerate(output_items):
            if index in notified_indices:
                continue
            if getattr(item, "type", None) != "function_call":
                continue

            tool_name = getattr(item, "name", None) or "tool"
            tool_use_id = (
                getattr(item, "call_id", None)
                or getattr(item, "id", None)
                or f"tool-{index}"
            )

            payload = {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "index": index,
            }

            self._notify_tool_stream_listeners("start", payload)
            self.logger.info(
                "Model emitted fallback tool notification",
                data={
                    "progress_action": ProgressAction.CALLING_TOOL,
                    "agent_name": self.name,
                    "model": model,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "tool_event": "start",
                    "fallback": True,
                },
            )
            self._notify_tool_stream_listeners("stop", payload)
            self.logger.info(
                "Model emitted fallback tool notification",
                data={
                    "progress_action": ProgressAction.CALLING_TOOL,
                    "agent_name": self.name,
                    "model": model,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "tool_event": "stop",
                    "fallback": True,
                },
            )

    async def _emit_streaming_progress(
        self,
        model: str,
        new_total: int,
        type: ProgressAction = ProgressAction.STREAMING,
    ) -> None:
        """Emit a streaming progress event.

        Args:
            model: The model being used.
            new_total: The new total token count.
        """
        token_str = str(new_total).rjust(5)

        data = {
            "progress_action": type,
            "model": model,
            "agent_name": self.name,
            "chat_turn": self.chat_turn(),
            "details": token_str.strip(),
        }
        self.logger.info("Streaming progress", data=data)
