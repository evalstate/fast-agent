"""Codex rollout-style session trace export writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from mcp.types import (
    AudioContent,
    CallToolRequest,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from fast_agent.constants import FAST_AGENT_USAGE, REASONING
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    get_image_data,
    get_text,
    is_resource_content,
    is_resource_link,
    is_text_content,
)
from fast_agent.mcp.mime_utils import is_image_mime_type, is_text_mime_type
from fast_agent.session.trace_export_models import ExportResult, ResolvedSessionExport

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_timestamp(value: datetime) -> str:
    value = _normalize_utc(value)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp_or_none(value: datetime | None) -> int | None:
    if value is None:
        return None
    value = _normalize_utc(value)
    return int(value.timestamp())


def _package_version() -> str:
    try:
        return version("fast-agent-mcp")
    except PackageNotFoundError:
        return "unknown"


def _json_arguments(arguments: object) -> str:
    if arguments is None:
        arguments = {}
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _data_url(image: ImageContent) -> str:
    return f"data:{image.mimeType};base64,{image.data}"


def _message_texts(blocks: Iterable[ContentBlock]) -> list[str]:
    return [block.text for block in blocks if isinstance(block, TextContent)]


def _user_images(blocks: Iterable[ContentBlock]) -> list[str]:
    return [_data_url(block) for block in blocks if isinstance(block, ImageContent)]


def _content_mime_type(block: ContentBlock) -> str | None:
    if isinstance(block, (AudioContent, ImageContent, ResourceLink)):
        return block.mimeType
    if isinstance(block, EmbeddedResource):
        return block.resource.mimeType
    return None


def _content_filename(block: ContentBlock) -> str | None:
    if isinstance(block, ResourceLink):
        uri = block.uri
    elif isinstance(block, EmbeddedResource):
        uri = block.resource.uri
    else:
        return None

    uri_str = str(uri)
    filename = uri_str.rsplit("/", 1)[-1] if "/" in uri_str else uri_str
    return filename or None


def _embedded_text_item(block: EmbeddedResource, *, output_text: bool) -> dict[str, object] | None:
    resource = block.resource
    if not isinstance(resource, TextResourceContents):
        return None

    mime_type = resource.mimeType or "text/plain"
    if not is_text_mime_type(mime_type):
        return None

    filename = _content_filename(block) or "resource"
    item_type = "output_text" if output_text else "input_text"
    return {
        "type": item_type,
        "text": (
            f'<fastagent:file title="{filename}" mimetype="{mime_type}">\n'
            f"{resource.text}\n"
            "</fastagent:file>"
        ),
    }


def _text_item(text: str, *, output_text: bool) -> dict[str, object]:
    return {
        "type": "output_text" if output_text else "input_text",
        "text": text,
    }


def _attachment_summary_text(block: ContentBlock) -> str | None:
    mime_type = _content_mime_type(block)

    if isinstance(block, ResourceLink):
        resource_uri = str(block.uri)
        filename = block.name or _content_filename(block) or resource_uri
        if mime_type:
            return f"Attached resource: {filename} ({mime_type}) — {resource_uri}"
        return f"Attached resource: {filename} — {resource_uri}"

    if isinstance(block, AudioContent):
        return f"Attached audio ({mime_type})" if mime_type else "Attached audio"

    filename = _content_filename(block)
    if filename and mime_type:
        return f"Attached file: {filename} ({mime_type})"
    if filename:
        return f"Attached file: {filename}"
    if mime_type:
        return f"Attached file ({mime_type})"
    return None


def _tool_attachment_item(block: ContentBlock) -> dict[str, object] | None:
    if isinstance(block, AudioContent):
        return {"type": "input_file", "file_data": block.data}

    mime_type = _content_mime_type(block)
    data = get_image_data(block)
    if data is not None:
        if mime_type and is_image_mime_type(mime_type):
            return {"type": "input_image", "image_url": f"data:{mime_type};base64,{data}"}

        item: dict[str, object] = {"type": "input_file", "file_data": data}
        filename = _content_filename(block)
        if filename is not None:
            item["filename"] = filename
        return item

    if is_resource_content(block):
        resource_uri = str(block.resource.uri)
        if mime_type and is_image_mime_type(mime_type):
            return {"type": "input_image", "image_url": resource_uri}
        return {"type": "input_file", "file_url": resource_uri}

    if is_resource_link(block):
        resource_uri = str(block.uri)
        if mime_type and is_image_mime_type(mime_type):
            return {"type": "input_image", "image_url": resource_uri}
        return {"type": "input_file", "file_url": resource_uri}

    return None


def _message_attachment_item(block: ContentBlock, *, output_text: bool) -> dict[str, object] | None:
    attachment = _tool_attachment_item(block)
    if attachment is None:
        return None
    if attachment.get("type") == "input_image":
        return attachment

    summary = _attachment_summary_text(block)
    if summary is None:
        return None
    return _text_item(summary, output_text=output_text)


def _message_content_items(message: PromptMessageExtended) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    output_text = message.role != "user"
    for block in message.content:
        if is_text_content(block):
            text = get_text(block) or ""
            items.append(_text_item(text, output_text=output_text))
            continue

        if isinstance(block, EmbeddedResource):
            text_item = _embedded_text_item(block, output_text=output_text)
            if text_item is not None:
                items.append(text_item)
                continue

        input_item = _message_attachment_item(block, output_text=output_text)
        if input_item is not None:
            items.append(input_item)
    return items


def _reasoning_texts(message: PromptMessageExtended) -> list[str]:
    channels = message.channels
    if channels is None:
        return []
    blocks = channels.get(REASONING)
    if blocks is None:
        return []
    return _message_texts(blocks)


def _reasoning_item(message: PromptMessageExtended) -> dict[str, object] | None:
    texts = _reasoning_texts(message)
    if not texts:
        return None
    return {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text} for text in texts],
    }


def _developer_message_item(system_prompt: str) -> dict[str, object]:
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": system_prompt}],
    }


def _assistant_message_item(message: PromptMessageExtended) -> dict[str, object] | None:
    content = _message_content_items(message)
    if not content:
        return None

    payload: dict[str, object] = {
        "type": "message",
        "role": "assistant",
        "content": content,
    }
    if message.stop_reason == "endTurn":
        payload["end_turn"] = True
    if message.phase is not None:
        payload["phase"] = message.phase
    return payload


def _user_message_item(message: PromptMessageExtended) -> dict[str, object] | None:
    content = _message_content_items(message)
    if not content:
        return None
    return {
        "type": "message",
        "role": "user",
        "content": content,
    }


def _function_call_items(message: PromptMessageExtended) -> list[dict[str, object]]:
    if message.tool_calls is None:
        return []

    items: list[dict[str, object]] = []
    for call_id, call in message.tool_calls.items():
        items.append(_function_call_item(call_id, call))
    return items


def _function_call_item(call_id: str, call: CallToolRequest) -> dict[str, object]:
    return {
        "type": "function_call",
        "name": call.params.name,
        "arguments": _json_arguments(call.params.arguments),
        "call_id": call_id,
    }


def _tool_result_output(result: CallToolResult) -> object:
    items: list[dict[str, object]] = []
    text_parts: list[str] = []

    def flush_text_parts() -> None:
        if not text_parts:
            return
        items.append({"type": "input_text", "text": "\n".join(text_parts)})
        text_parts.clear()

    for block in canonicalize_tool_result_content_for_llm(result):
        if isinstance(block, TextContent):
            text_parts.append(block.text)
            continue

        flush_text_parts()

        if isinstance(block, EmbeddedResource):
            text_item = _embedded_text_item(block, output_text=False)
            if text_item is not None:
                items.append(text_item)
                continue

        attachment = _tool_attachment_item(block)
        if attachment is not None:
            items.append(attachment)

    flush_text_parts()

    if items:
        if len(items) == 1 and items[0].get("type") == "input_text":
            text = items[0].get("text")
            if isinstance(text, str):
                return text
        return items
    return ""


def _object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(key, str):
            result[key] = item
    return result


def _string_field(mapping: dict[str, object] | None, key: str) -> str | None:
    if mapping is None:
        return None
    value = mapping.get(key)
    return value if isinstance(value, str) and value else None


def _int_field(mapping: dict[str, object] | None, key: str) -> int | None:
    if mapping is None:
        return None
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _message_usage_payload(message: PromptMessageExtended) -> dict[str, object] | None:
    channels = message.channels
    if channels is None:
        return None

    blocks = channels.get(FAST_AGENT_USAGE)
    if blocks is None:
        return None

    for text in reversed(_message_texts(blocks)):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        parsed = _object_mapping(payload)
        if parsed is not None:
            return parsed

    return None


def _usage_turn_payload(message: PromptMessageExtended) -> dict[str, object] | None:
    return _object_mapping((_message_usage_payload(message) or {}).get("turn"))


def _usage_summary_payload(message: PromptMessageExtended) -> dict[str, object] | None:
    return _object_mapping((_message_usage_payload(message) or {}).get("summary"))


@dataclass(frozen=True, slots=True)
class _TraceMeta:
    model: str | None
    provider: str | None
    model_context_window: int | None


def _trace_meta(resolved: ResolvedSessionExport) -> _TraceMeta:
    agent_snapshot = resolved.snapshot.continuation.agents[resolved.agent_name]
    model = agent_snapshot.model
    provider = agent_snapshot.provider
    model_context_window = ModelDatabase.get_context_window(model) if model else None

    for message in resolved.history:
        turn_payload = _usage_turn_payload(message)
        summary_payload = _usage_summary_payload(message)
        if model is None:
            model = _string_field(turn_payload, "model")
        if provider is None:
            provider = _string_field(turn_payload, "provider")
        if model_context_window is None:
            model_context_window = _int_field(summary_payload, "context_window_size")
        if model_context_window is None and model is not None:
            model_context_window = ModelDatabase.get_context_window(model)
        if model is not None and provider is not None and model_context_window is not None:
            break

    return _TraceMeta(
        model=model,
        provider=provider,
        model_context_window=model_context_window,
    )


def _cached_input_tokens(turn_payload: dict[str, object] | None) -> int | None:
    cache_payload = _object_mapping((turn_payload or {}).get("cache_usage"))
    cache_read_tokens = _int_field(cache_payload, "cache_read_tokens")
    if cache_read_tokens not in {None, 0}:
        return cache_read_tokens
    return _int_field(cache_payload, "cache_hit_tokens")


def _token_count_payload(
    message: PromptMessageExtended,
    *,
    model_context_window: int | None,
) -> dict[str, object] | None:
    turn_payload = _usage_turn_payload(message)
    if turn_payload is None:
        return None

    token_usage: dict[str, object] = {}

    input_tokens = _int_field(turn_payload, "display_input_tokens")
    if input_tokens is None:
        input_tokens = _int_field(turn_payload, "input_tokens")
    if input_tokens is not None:
        token_usage["input_tokens"] = input_tokens

    cached_input_tokens = _cached_input_tokens(turn_payload)
    if cached_input_tokens is not None:
        token_usage["cached_input_tokens"] = cached_input_tokens

    output_tokens = _int_field(turn_payload, "output_tokens")
    if output_tokens is not None:
        token_usage["output_tokens"] = output_tokens

    reasoning_output_tokens = _int_field(turn_payload, "reasoning_tokens")
    if reasoning_output_tokens is not None:
        token_usage["reasoning_output_tokens"] = reasoning_output_tokens

    total_tokens = _int_field(turn_payload, "total_tokens")
    if total_tokens is not None:
        token_usage["total_tokens"] = total_tokens

    if not token_usage:
        return None

    info: dict[str, object] = {
        "last_token_usage": token_usage,
    }
    if model_context_window is not None:
        info["model_context_window"] = model_context_window

    return {
        "type": "token_count",
        "info": info,
    }


def _function_call_output_items(message: PromptMessageExtended) -> list[dict[str, object]]:
    if message.tool_results is None:
        return []

    items: list[dict[str, object]] = []
    for call_id, result in message.tool_results.items():
        items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _tool_result_output(result),
            }
        )
    return items


def _session_cwd(resolved: ResolvedSessionExport) -> str:
    cwd = resolved.snapshot.continuation.cwd
    return cwd if isinstance(cwd, str) and cwd else "."


def _session_meta_payload(resolved: ResolvedSessionExport, meta: _TraceMeta) -> dict[str, object]:
    agent_snapshot = resolved.snapshot.continuation.agents[resolved.agent_name]
    payload: dict[str, object] = {
        "id": resolved.session_id,
        "timestamp": _utc_timestamp(resolved.snapshot.created_at),
        "cwd": _session_cwd(resolved),
        "originator": "fast-agent",
        "cli_version": _package_version(),
        "source": "cli",
    }
    if meta.provider is not None:
        payload["model_provider"] = meta.provider
    if agent_snapshot.resolved_prompt:
        payload["base_instructions"] = {"text": agent_snapshot.resolved_prompt}
    return payload


def _turn_context_payload(
    resolved: ResolvedSessionExport,
    *,
    turn_id: str,
    meta: _TraceMeta,
    turn_timestamp: datetime | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "turn_id": turn_id,
        "cwd": _session_cwd(resolved),
        "timezone": "UTC",
        "summary": "auto",
    }
    if turn_timestamp is not None:
        payload["current_date"] = _normalize_utc(turn_timestamp).date().isoformat()
    if meta.model is not None:
        payload["model"] = meta.model
    return payload


def _turn_started_payload(
    turn_id: str,
    *,
    model_context_window: int | None,
    started_at: datetime | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "turn_started",
        "turn_id": turn_id,
        "started_at": _timestamp_or_none(started_at),
        "collaboration_mode_kind": "default",
    }
    if model_context_window is not None:
        payload["model_context_window"] = model_context_window
    return payload


def _user_event_payload(message: PromptMessageExtended) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "user_message",
        "message": "\n".join(_message_texts(message.content)),
        "local_images": [],
        "text_elements": [],
    }
    images = _user_images(message.content)
    if images:
        payload["images"] = images
    return payload


def _turn_complete_payload(turn_id: str, last_agent_message: str | None) -> dict[str, object]:
    return {
        "type": "turn_complete",
        "turn_id": turn_id,
        "last_agent_message": last_agent_message,
    }


def _response_items(message: PromptMessageExtended) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []

    if message.role == "user":
        user_item = _user_message_item(message)
        if user_item is not None:
            items.append(user_item)
        items.extend(_function_call_output_items(message))
        return items

    reasoning_item = _reasoning_item(message)
    if reasoning_item is not None:
        items.append(reasoning_item)

    assistant_item = _assistant_message_item(message)
    if assistant_item is not None:
        items.append(assistant_item)

    items.extend(_function_call_items(message))
    return items


def _is_turn_start(message: PromptMessageExtended) -> bool:
    return message.role == "user" and not message.tool_results


@dataclass(slots=True)
class _TurnState:
    turn_id: str
    last_agent_message: str | None = None


class _TimestampCursor:
    def __init__(self, start: datetime) -> None:
        self._current = _normalize_utc(start)

    def next(self, *, at: datetime | None = None) -> str:
        if at is not None:
            normalized = _normalize_utc(at)
            if normalized > self._current:
                self._current = normalized
        timestamp = _utc_timestamp(self._current)
        self._current += timedelta(milliseconds=1)
        return timestamp


def _turn_timestamps(resolved: ResolvedSessionExport) -> list[datetime | None]:
    turn_timestamps: list[datetime | None] = []
    for message, message_timestamp in zip(
        resolved.history, resolved.message_timestamps, strict=True
    ):
        if _is_turn_start(message) or not turn_timestamps:
            turn_timestamps.append(message_timestamp)
            continue
        if turn_timestamps[-1] is None and message_timestamp is not None:
            turn_timestamps[-1] = message_timestamp
    return turn_timestamps


class CodexTraceWriter:
    """Write a resolved session export as native Codex rollout JSONL."""

    def write(self, resolved: ResolvedSessionExport, output_path: Path) -> ExportResult:
        records = list(self._records(resolved))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        return ExportResult(
            session_id=resolved.session_id,
            agent_name=resolved.agent_name,
            format="codex",
            output_path=output_path,
            record_count=len(records),
        )

    def _records(self, resolved: ResolvedSessionExport) -> list[dict[str, object]]:
        cursor = _TimestampCursor(resolved.snapshot.created_at)
        meta = _trace_meta(resolved)
        turn_timestamps = _turn_timestamps(resolved)
        records: list[dict[str, object]] = []
        records.append(
            {
                "timestamp": cursor.next(),
                "type": "session_meta",
                "payload": _session_meta_payload(resolved, meta),
            }
        )

        agent_snapshot = resolved.snapshot.continuation.agents[resolved.agent_name]
        if agent_snapshot.resolved_prompt:
            records.append(
                {
                    "timestamp": cursor.next(),
                    "type": "response_item",
                    "payload": _developer_message_item(agent_snapshot.resolved_prompt),
                }
            )

        turn_counter = 0
        current_turn: _TurnState | None = None

        def start_turn(user_message: PromptMessageExtended | None) -> None:
            nonlocal turn_counter, current_turn
            turn_timestamp = (
                turn_timestamps[turn_counter] if turn_counter < len(turn_timestamps) else None
            )
            turn_counter += 1
            current_turn = _TurnState(turn_id=f"turn-{turn_counter}")
            records.append(
                {
                    "timestamp": cursor.next(at=turn_timestamp),
                    "type": "event_msg",
                    "payload": _turn_started_payload(
                        current_turn.turn_id,
                        model_context_window=meta.model_context_window,
                        started_at=turn_timestamp,
                    ),
                }
            )
            if user_message is not None:
                records.append(
                    {
                        "timestamp": cursor.next(),
                        "type": "event_msg",
                        "payload": _user_event_payload(user_message),
                    }
                )
            records.append(
                {
                    "timestamp": cursor.next(),
                    "type": "turn_context",
                    "payload": _turn_context_payload(
                        resolved,
                        turn_id=current_turn.turn_id,
                        meta=meta,
                        turn_timestamp=turn_timestamp,
                    ),
                }
            )

        def finish_turn() -> None:
            nonlocal current_turn
            if current_turn is None:
                return
            records.append(
                {
                    "timestamp": cursor.next(),
                    "type": "event_msg",
                    "payload": _turn_complete_payload(
                        current_turn.turn_id,
                        current_turn.last_agent_message,
                    ),
                }
            )
            current_turn = None

        for message, message_timestamp in zip(
            resolved.history, resolved.message_timestamps, strict=True
        ):
            if _is_turn_start(message):
                finish_turn()
                start_turn(message)
            elif current_turn is None:
                start_turn(None)

            if current_turn is not None and message.role == "assistant":
                texts = _message_texts(message.content)
                if texts:
                    current_turn.last_agent_message = texts[-1]

            for item in _response_items(message):
                records.append(
                    {
                        "timestamp": cursor.next(at=message_timestamp),
                        "type": "response_item",
                        "payload": item,
                    }
                )
                message_timestamp = None
            if message.role == "assistant":
                token_count = _token_count_payload(
                    message,
                    model_context_window=meta.model_context_window,
                )
                if token_count is not None:
                    records.append(
                        {
                            "timestamp": cursor.next(at=message_timestamp),
                            "type": "event_msg",
                            "payload": token_count,
                        }
                    )

        finish_turn()
        return records
