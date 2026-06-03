"""Codex rollout-style session trace export writer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import parse_qsl, urlencode

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

from fast_agent.constants import (
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    FAST_AGENT_TIMING,
    FAST_AGENT_USAGE,
    OPENAI_ASSISTANT_MESSAGE_ITEMS,
    REASONING,
)
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
from fast_agent.privacy.sanitizer import RedactionAccumulator, RedactionSummary, TraceSanitizer
from fast_agent.session.trace_export_models import ExportResult, ResolvedSessionExport
from fast_agent.utils.count_display import format_count

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.privacy.sanitizer import SanitizedText
    from fast_agent.session.snapshot import SessionAgentSnapshot


@dataclass(frozen=True, slots=True)
class _WebSearchFields:
    action: str | None
    query: str | None
    queries: list[str]
    url: str | None
    pattern: str | None


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


def _record(
    record_type: str,
    payload: dict[str, object],
    *,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "type": record_type,
        "payload": payload,
    }
    if timestamp is not None:
        record["timestamp"] = _utc_timestamp(timestamp)
    return record


def _package_version() -> str:
    try:
        return version("fast-agent-mcp")
    except PackageNotFoundError:
        return "unknown"


def _json_arguments(arguments: object) -> str:
    if arguments is None:
        arguments = {}
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


class _TraceSanitization:
    def __init__(
        self,
        sanitizer: TraceSanitizer,
        *,
        total_texts: int = 0,
        total_characters: int = 0,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._sanitizer = sanitizer
        self._redactions = RedactionAccumulator(model=sanitizer.model_info)
        self._cache: dict[str, SanitizedText] = {}
        self._started = time.perf_counter()
        self._total_texts = total_texts
        self._total_characters = total_characters
        self._processed_texts = 0
        self._processed_characters = 0
        self._last_progress_percent = -1
        self._progress_callback = progress_callback
        if total_texts > 0:
            self._emit_progress(
                "Privacy filter: sanitizing "
                f"{format_count(total_texts, 'text value')}, "
                f"{format_count(total_characters, 'character')} total..."
            )

    def text(self, value: str) -> str:
        sanitized = self._cache.get(value)
        if sanitized is None:
            sanitized = self._sanitizer.sanitize_text(value)
            self._cache[value] = sanitized
            self._processed_texts += 1
            self._processed_characters += len(value)
            self._emit_overall_progress()
        self._redactions.add(sanitized.spans)
        return sanitized.text

    def summary(self) -> RedactionSummary:
        self._redactions.elapsed = timedelta(seconds=time.perf_counter() - self._started)
        return self._redactions.summary()

    def _emit_overall_progress(self) -> None:
        if self._progress_callback is None or self._total_texts <= 0:
            return
        percent = min(100, round((self._processed_texts / self._total_texts) * 100))
        if (
            self._processed_texts not in (1, self._total_texts)
            and percent < self._last_progress_percent + 5
        ):
            return
        self._last_progress_percent = percent
        self._emit_progress(
            "Privacy filter: overall "
            f"{self._processed_texts:,} of {format_count(self._total_texts, 'text value')} "
            f"({percent}%, "
            f"{self._processed_characters:,} of "
            f"{format_count(self._total_characters, 'character')})..."
        )

    def _emit_progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)


class _TraceSanitizationPlan:
    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.unique_text_count = 0
        self.total_characters = 0

    def text(self, value: str) -> str:
        if value not in self._seen:
            self._seen.add(value)
            self.unique_text_count += 1
            self.total_characters += len(value)
        return value


class _TextSanitization(Protocol):
    def text(self, value: str) -> str: ...


def _sanitize_text(sanitization: _TextSanitization | None, text: str) -> str:
    if sanitization is None or not text:
        return text
    return sanitization.text(text)


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


def _embedded_text_item(
    block: EmbeddedResource,
    *,
    output_text: bool,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object] | None:
    resource = block.resource
    if not isinstance(resource, TextResourceContents):
        return None

    mime_type = resource.mimeType or "text/plain"
    if not is_text_mime_type(mime_type):
        return None

    filename = _content_filename(block) or "resource"
    item_type = "output_text" if output_text else "input_text"
    text = (
        f'<fastagent:file title="{filename}" mimetype="{mime_type}">\n'
        f"{resource.text}\n"
        "</fastagent:file>"
    )
    return {"type": item_type, "text": _sanitize_text(sanitization, text)}


def _text_item(
    text: str,
    *,
    output_text: bool,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object]:
    return {
        "type": "output_text" if output_text else "input_text",
        "text": _sanitize_text(sanitization, text),
    }


def _resource_link_summary(block: ResourceLink, mime_type: str | None) -> str:
    resource_uri = str(block.uri)
    filename = block.name or _content_filename(block) or resource_uri
    if mime_type:
        return f"Attached resource: {filename} ({mime_type}) — {resource_uri}"
    return f"Attached resource: {filename} — {resource_uri}"


def _file_summary(filename: str | None, mime_type: str | None) -> str | None:
    if filename and mime_type:
        return f"Attached file: {filename} ({mime_type})"
    if filename:
        return f"Attached file: {filename}"
    if mime_type:
        return f"Attached file ({mime_type})"
    return None


def _attachment_summary_text(block: ContentBlock) -> str | None:
    mime_type = _content_mime_type(block)

    if isinstance(block, ResourceLink):
        return _resource_link_summary(block, mime_type)

    if isinstance(block, AudioContent):
        return f"Attached audio ({mime_type})" if mime_type else "Attached audio"

    return _file_summary(_content_filename(block), mime_type)


def _file_data_item(block: ContentBlock, data: str) -> dict[str, object]:
    item: dict[str, object] = {"type": "input_file", "file_data": data}
    filename = _content_filename(block)
    if filename is not None:
        item["filename"] = filename
    return item


def _uri_attachment_item(
    resource_uri: str,
    mime_type: str | None,
) -> dict[str, object]:
    if mime_type and is_image_mime_type(mime_type):
        return {"type": "input_image", "image_url": resource_uri}
    return {"type": "input_file", "file_url": resource_uri}


def _tool_attachment_item(block: ContentBlock) -> dict[str, object] | None:
    if isinstance(block, AudioContent):
        return {"type": "input_file", "file_data": block.data}

    mime_type = _content_mime_type(block)
    data = get_image_data(block)
    if data is not None:
        if mime_type and is_image_mime_type(mime_type):
            return {"type": "input_image", "image_url": f"data:{mime_type};base64,{data}"}
        return _file_data_item(block, data)

    if is_resource_content(block):
        return _uri_attachment_item(str(block.resource.uri), mime_type)

    if is_resource_link(block):
        return _uri_attachment_item(str(block.uri), mime_type)

    return None


def _message_attachment_item(
    block: ContentBlock,
    *,
    output_text: bool,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object] | None:
    attachment = _tool_attachment_item(block)
    if attachment is None:
        return None
    if attachment.get("type") == "input_image":
        return attachment

    summary = _attachment_summary_text(block)
    if summary is None:
        return None
    return _text_item(summary, output_text=output_text, sanitization=sanitization)


def _message_content_items(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    output_text = message.role != "user"
    for block in message.content:
        if is_text_content(block):
            text = get_text(block) or ""
            items.append(_text_item(text, output_text=output_text, sanitization=sanitization))
            continue

        if isinstance(block, EmbeddedResource):
            text_item = _embedded_text_item(
                block,
                output_text=output_text,
                sanitization=sanitization,
            )
            if text_item is not None:
                items.append(text_item)
                continue

        input_item = _message_attachment_item(
            block,
            output_text=output_text,
            sanitization=sanitization,
        )
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


def _reasoning_item(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object] | None:
    texts = _reasoning_texts(message)
    if not texts:
        return None
    return {
        "type": "reasoning",
        "summary": [
            {"type": "summary_text", "text": _sanitize_text(sanitization, text)}
            for text in texts
        ],
    }


def _developer_message_item(
    system_prompt: str,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object]:
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": _sanitize_text(sanitization, system_prompt)}],
    }


def _content_item_from_mapping(
    item: dict[str, object],
    *,
    output_text: bool,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object] | None:
    item_type = _string_field(item, "type")
    if item_type in {"input_text", "output_text"}:
        text = _string_field(item, "text")
        if text is None:
            return None
        normalized_type = "output_text" if output_text else "input_text"
        content_item: dict[str, object] = {
            "type": normalized_type,
            "text": _sanitize_text(sanitization, text),
        }
        annotations = item.get("annotations")
        if isinstance(annotations, list):
            content_item["annotations"] = annotations
        return content_item

    if item_type == "input_image":
        image_url = _string_field(item, "image_url")
        if image_url is None:
            return None
        content_item: dict[str, object] = {"type": "input_image", "image_url": image_url}
        detail = _string_field(item, "detail")
        if detail is not None:
            content_item["detail"] = detail
        return content_item

    return None


def _raw_assistant_message_items(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> list[dict[str, object]]:
    """Return provider-captured assistant message items in Codex-compatible shape.

    OpenAI Responses can emit multiple assistant message items in one fast-agent history
    message (for example commentary followed by final_answer). The channel preserves that
    item boundary and phase metadata; prefer it over reconstructing a single collapsed item.
    """

    items: list[dict[str, object]] = []
    for payload in _json_channel_payloads(message, OPENAI_ASSISTANT_MESSAGE_ITEMS):
        if payload.get("type") != "message":
            continue

        role = _string_field(payload, "role") or "assistant"
        if role != "assistant":
            continue

        content_items: list[dict[str, object]] = []
        content = payload.get("content")
        if isinstance(content, list):
            for raw_content_item in content:
                content_item = _content_item_from_mapping(
                    _object_mapping(raw_content_item) or {},
                    output_text=True,
                    sanitization=sanitization,
                )
                if content_item is not None:
                    content_items.append(content_item)

        if not content_items:
            continue

        item: dict[str, object] = {
            "type": "message",
            "role": "assistant",
            "content": content_items,
        }
        phase = _string_field(payload, "phase")
        if phase is not None:
            item["phase"] = phase
        items.append(item)

    return items


def _server_tool_input(payload: dict[str, object]) -> dict[str, object] | None:
    input_payload = _object_mapping(payload.get("input"))
    if input_payload is not None:
        return input_payload
    return payload


def _string_list_field(mapping: dict[str, object] | None, key: str) -> list[str]:
    if mapping is None:
        return []
    value = mapping.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _web_search_fields(
    payload: dict[str, object],
    sanitization: _TextSanitization | None,
) -> _WebSearchFields:
    tool_input = _server_tool_input(payload)
    query = _string_field(tool_input, "query") or _string_field(payload, "query")
    pattern = _string_field(tool_input, "pattern") or _string_field(payload, "pattern")
    return _WebSearchFields(
        action=_string_field(payload, "action"),
        query=_sanitize_text(sanitization, query) if query is not None else None,
        queries=[
            _sanitize_text(sanitization, item)
            for item in (
                _string_list_field(tool_input, "queries")
                or _string_list_field(payload, "queries")
            )
        ],
        url=_string_field(tool_input, "url") or _string_field(payload, "url"),
        pattern=_sanitize_text(sanitization, pattern) if pattern is not None else None,
    )


def _open_page_action(fields: _WebSearchFields) -> dict[str, object] | None:
    if fields.action not in {"open_page", "open_url", "fetch"} and (
        fields.url is None or fields.query or fields.queries
    ):
        return None
    result: dict[str, object] = {"type": "open_page"}
    if fields.url is not None:
        result["url"] = fields.url
    return result


def _find_in_page_action(fields: _WebSearchFields) -> dict[str, object] | None:
    if fields.action not in {"find_in_page", "find"} and (
        fields.url is None or fields.pattern is None
    ):
        return None
    result: dict[str, object] = {"type": "find_in_page"}
    if fields.url is not None:
        result["url"] = fields.url
    if fields.pattern is not None:
        result["pattern"] = fields.pattern
    return result


def _search_action(fields: _WebSearchFields) -> dict[str, object] | None:
    if fields.query is None and not fields.queries:
        return None
    result: dict[str, object] = {"type": "search"}
    if fields.query is not None:
        result["query"] = fields.query
    if fields.queries:
        result["queries"] = fields.queries
    return result


def _web_search_action(
    payload: dict[str, object],
    sanitization: _TextSanitization | None = None,
) -> dict[str, object] | None:
    fields = _web_search_fields(payload, sanitization)
    return (
        _open_page_action(fields)
        or _find_in_page_action(fields)
        or _search_action(fields)
    )


def _server_tool_response_items(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []

    for payload in _json_channel_payloads(message, ANTHROPIC_SERVER_TOOLS_CHANNEL):
        if payload.get("type") != "server_tool_use":
            continue

        tool_name = _string_field(payload, "name")
        if tool_name not in {"web_search", "web_fetch", "web_search_call"}:
            continue

        item: dict[str, object] = {"type": "web_search_call"}
        item_id = _string_field(payload, "id")
        if item_id is not None:
            item["id"] = item_id
        status = _string_field(payload, "status")
        if status is not None:
            item["status"] = status
        action = _web_search_action(payload, sanitization=sanitization)
        if action is not None:
            item["action"] = action
        items.append(item)

    return items


def _assistant_message_item(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object] | None:
    content = _message_content_items(message, sanitization=sanitization)
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


def _user_message_item(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object] | None:
    content = _message_content_items(message, sanitization=sanitization)
    if not content:
        return None
    return {
        "type": "message",
        "role": "user",
        "content": content,
    }


def _function_call_items(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> list[dict[str, object]]:
    if message.tool_calls is None:
        return []

    items: list[dict[str, object]] = []
    for call_id, call in message.tool_calls.items():
        items.append(_function_call_item(call_id, call, sanitization=sanitization))
    return items


def _function_call_item(
    call_id: str,
    call: CallToolRequest,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object]:
    return {
        "type": "function_call",
        "name": call.params.name,
        "arguments": _sanitize_text(sanitization, _json_arguments(call.params.arguments)),
        "call_id": call_id,
    }


def _flush_tool_text_parts(
    items: list[dict[str, object]],
    text_parts: list[str],
    sanitization: _TextSanitization | None,
) -> None:
    if not text_parts:
        return
    text = _sanitize_text(sanitization, "\n".join(text_parts))
    items.append({"type": "input_text", "text": text})
    text_parts.clear()


def _tool_result_block_item(
    block: ContentBlock,
    sanitization: _TextSanitization | None,
) -> dict[str, object] | None:
    if isinstance(block, EmbeddedResource):
        text_item = _embedded_text_item(
            block,
            output_text=False,
            sanitization=sanitization,
        )
        if text_item is not None:
            return text_item
    return _tool_attachment_item(block)


def _collapse_tool_result_items(items: list[dict[str, object]]) -> object:
    if not items:
        return ""
    if len(items) == 1 and items[0].get("type") == "input_text":
        text = items[0].get("text")
        if isinstance(text, str):
            return text
    return items


def _tool_result_output(
    result: CallToolResult,
    sanitization: _TextSanitization | None = None,
) -> object:
    items: list[dict[str, object]] = []
    text_parts: list[str] = []

    for block in canonicalize_tool_result_content_for_llm(result):
        if isinstance(block, TextContent):
            text_parts.append(block.text)
            continue

        _flush_tool_text_parts(items, text_parts, sanitization)

        item = _tool_result_block_item(block, sanitization)
        if item is not None:
            items.append(item)

    _flush_tool_text_parts(items, text_parts, sanitization)
    return _collapse_tool_result_items(items)


def _tool_result_status(result: CallToolResult) -> str:
    return "error" if result.isError else "success"


def _object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items() if isinstance(key, str)}


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


def _float_field(mapping: dict[str, object] | None, key: str) -> float | None:
    if mapping is None:
        return None
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _milliseconds_field(mapping: dict[str, object] | None, key: str) -> int | None:
    value = _float_field(mapping, key)
    if value is None:
        return None
    return max(0, round(value))


def _json_channel_payloads(
    message: PromptMessageExtended,
    channel_name: str,
) -> list[dict[str, object]]:
    channels = message.channels
    if channels is None:
        return []

    blocks = channels.get(channel_name)
    if blocks is None:
        return []

    payloads: list[dict[str, object]] = []
    for text in _message_texts(blocks):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        parsed = _object_mapping(payload)
        if parsed is not None:
            payloads.append(parsed)
    return payloads


def _message_usage_payload(message: PromptMessageExtended) -> dict[str, object] | None:
    payloads = _json_channel_payloads(message, FAST_AGENT_USAGE)
    return payloads[-1] if payloads else None


def _message_timing_payload(message: PromptMessageExtended) -> dict[str, object] | None:
    payloads = _json_channel_payloads(message, FAST_AGENT_TIMING)
    return payloads[0] if payloads else None


def _usage_turn_payload(message: PromptMessageExtended) -> dict[str, object] | None:
    return _object_mapping((_message_usage_payload(message) or {}).get("turn"))


def _usage_summary_payload(message: PromptMessageExtended) -> dict[str, object] | None:
    return _object_mapping((_message_usage_payload(message) or {}).get("summary"))


@dataclass(frozen=True, slots=True)
class _TraceMeta:
    model: str | None
    model_spec: str | None
    provider: str | None
    model_context_window: int | None


def _service_tier_model_spec(
    model_spec: str | None,
    agent_snapshot: "SessionAgentSnapshot",
) -> str | None:
    if model_spec is None:
        return None
    request_settings = agent_snapshot.request_settings
    service_tier = None if request_settings is None else request_settings.service_tier
    if service_tier is None:
        return model_spec

    base_model_spec, _, query = model_spec.partition("?")
    query_params = dict(parse_qsl(query, keep_blank_values=True))
    query_params["service_tier"] = service_tier
    encoded_query = urlencode(query_params)
    if not encoded_query:
        return base_model_spec
    return f"{base_model_spec}?{encoded_query}"


def _full_model_spec(agent_snapshot: "SessionAgentSnapshot") -> str | None:
    model_spec = agent_snapshot.model_spec
    if model_spec is not None:
        stripped = model_spec.strip()
        model_spec = stripped or None
    model = agent_snapshot.model
    if model_spec is None and model is not None:
        stripped = model.strip()
        if stripped:
            model_spec = stripped
    return _service_tier_model_spec(model_spec, agent_snapshot)


def _model_context_window(model_spec: str | None, model: str | None) -> int | None:
    if model_spec is not None:
        context_window = ModelDatabase.get_context_window(model_spec)
        if context_window is not None:
            return context_window
    if model is not None:
        return ModelDatabase.get_context_window(model)
    return None


def _trace_meta(resolved: ResolvedSessionExport) -> _TraceMeta:
    agent_snapshot = resolved.snapshot.continuation.agents[resolved.agent_name]
    model = agent_snapshot.model
    model_spec = _full_model_spec(agent_snapshot)
    provider = agent_snapshot.provider
    model_context_window = _model_context_window(model_spec, model)

    for message in resolved.history:
        turn_payload = _usage_turn_payload(message)
        summary_payload = _usage_summary_payload(message)
        if model is None:
            model = _string_field(turn_payload, "model")
        if model_spec is None:
            model_spec = _string_field(turn_payload, "model")
        if provider is None:
            provider = _string_field(turn_payload, "provider")
        if model_context_window is None:
            model_context_window = _int_field(summary_payload, "context_window_size")
        if model_context_window is None:
            model_context_window = _model_context_window(model_spec, model)
        if (
            model is not None
            and model_spec is not None
            and provider is not None
            and model_context_window is not None
        ):
            break

    model_spec = _service_tier_model_spec(model_spec or model, agent_snapshot)

    return _TraceMeta(
        model=model,
        model_spec=model_spec,
        provider=provider,
        model_context_window=model_context_window,
    )


def _cached_input_tokens(turn_payload: dict[str, object] | None) -> int | None:
    cache_payload = _object_mapping((turn_payload or {}).get("cache_usage"))
    cache_read_tokens = _int_field(cache_payload, "cache_read_tokens")
    if cache_read_tokens not in {None, 0}:
        return cache_read_tokens
    return _int_field(cache_payload, "cache_hit_tokens")


def _token_usage_from_turn_payload(turn_payload: dict[str, object]) -> dict[str, object] | None:
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

    return token_usage


def _cached_input_tokens_from_summary(summary_payload: dict[str, object] | None) -> int | None:
    cache_read_tokens = _int_field(summary_payload, "cumulative_cache_read_tokens")
    cache_hit_tokens = _int_field(summary_payload, "cumulative_cache_hit_tokens")
    total = (cache_read_tokens or 0) + (cache_hit_tokens or 0)
    return total if total > 0 else None


def _token_usage_from_summary_payload(
    summary_payload: dict[str, object] | None,
) -> dict[str, object] | None:
    token_usage: dict[str, object] = {}

    input_tokens = _int_field(summary_payload, "cumulative_input_tokens")
    if input_tokens is not None:
        token_usage["input_tokens"] = input_tokens

    cached_input_tokens = _cached_input_tokens_from_summary(summary_payload)
    if cached_input_tokens is not None:
        token_usage["cached_input_tokens"] = cached_input_tokens

    output_tokens = _int_field(summary_payload, "cumulative_output_tokens")
    if output_tokens is not None:
        token_usage["output_tokens"] = output_tokens

    reasoning_output_tokens = _int_field(summary_payload, "cumulative_reasoning_tokens")
    if reasoning_output_tokens is not None:
        token_usage["reasoning_output_tokens"] = reasoning_output_tokens

    total_tokens = _int_field(summary_payload, "cumulative_billing_tokens")
    if total_tokens is None:
        total_tokens = _int_field(summary_payload, "current_context_tokens")
    if total_tokens is not None:
        token_usage["total_tokens"] = total_tokens

    if not token_usage:
        return None

    return token_usage


def _token_count_payload(
    message: PromptMessageExtended,
    *,
    model_context_window: int | None,
) -> dict[str, object] | None:
    turn_payload = _usage_turn_payload(message)
    if turn_payload is None:
        return None

    last_token_usage = _token_usage_from_turn_payload(turn_payload)
    if last_token_usage is None:
        return None

    total_token_usage = _token_usage_from_summary_payload(_usage_summary_payload(message))
    if total_token_usage is None:
        total_token_usage = dict(last_token_usage)

    info: dict[str, object] = {
        "total_token_usage": total_token_usage,
        "last_token_usage": last_token_usage,
    }
    if model_context_window is not None:
        info["model_context_window"] = model_context_window

    return {
        "type": "token_count",
        "info": info,
    }


def _function_call_output_items(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> list[dict[str, object]]:
    if message.tool_results is None:
        return []

    items: list[dict[str, object]] = []
    for call_id, result in message.tool_results.items():
        items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _tool_result_output(result, sanitization=sanitization),
                "status": _tool_result_status(result),
            }
        )
    return items


def _session_cwd(resolved: ResolvedSessionExport) -> str:
    cwd = resolved.snapshot.continuation.cwd
    return cwd if isinstance(cwd, str) and cwd else "."


def _session_meta_payload(
    resolved: ResolvedSessionExport,
    meta: _TraceMeta,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object]:
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
    if meta.model_spec is not None:
        payload["model_spec"] = meta.model_spec
    if agent_snapshot.resolved_prompt:
        payload["base_instructions"] = {
            "text": _sanitize_text(sanitization, agent_snapshot.resolved_prompt)
        }
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
    if meta.model_spec is not None:
        payload["model_spec"] = meta.model_spec
    return payload


def _turn_started_payload(
    turn_id: str,
    *,
    model_context_window: int | None,
    started_at: datetime | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "task_started",
        "turn_id": turn_id,
        "collaboration_mode_kind": "default",
    }
    started_at_timestamp = _timestamp_or_none(started_at)
    if started_at_timestamp is not None:
        payload["started_at"] = started_at_timestamp
    if model_context_window is not None:
        payload["model_context_window"] = model_context_window
    return payload


def _user_event_payload(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "user_message",
        "message": _sanitize_text(sanitization, "\n".join(_message_texts(message.content))),
        "local_images": [],
        "text_elements": [],
    }
    images = _user_images(message.content)
    if images:
        payload["images"] = images
    return payload


def _turn_complete_payload(
    turn_id: str,
    last_agent_message: str | None,
    *,
    completed_at: datetime | None = None,
    duration_ms: int | None = None,
    time_to_first_token_ms: int | None = None,
    sanitization: _TextSanitization | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "task_complete",
        "turn_id": turn_id,
        "last_agent_message": (
            None
            if last_agent_message is None
            else _sanitize_text(sanitization, last_agent_message)
        ),
    }
    completed_at_timestamp = _timestamp_or_none(completed_at)
    if completed_at_timestamp is not None:
        payload["completed_at"] = completed_at_timestamp
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    if time_to_first_token_ms is not None:
        payload["time_to_first_token_ms"] = time_to_first_token_ms
    return payload


def _response_items(
    message: PromptMessageExtended,
    sanitization: _TextSanitization | None = None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []

    if message.role == "user":
        user_item = _user_message_item(message, sanitization=sanitization)
        if user_item is not None:
            items.append(user_item)
        items.extend(_function_call_output_items(message, sanitization=sanitization))
        return items

    reasoning_item = _reasoning_item(message, sanitization=sanitization)
    if reasoning_item is not None:
        items.append(reasoning_item)

    items.extend(_server_tool_response_items(message, sanitization=sanitization))

    raw_assistant_items = _raw_assistant_message_items(message, sanitization=sanitization)
    if raw_assistant_items:
        items.extend(raw_assistant_items)
    else:
        assistant_item = _assistant_message_item(message, sanitization=sanitization)
        if assistant_item is not None:
            items.append(assistant_item)

    items.extend(_function_call_items(message, sanitization=sanitization))
    return items


def _is_turn_start(message: PromptMessageExtended) -> bool:
    return message.role == "user" and not message.tool_results


def _elapsed_ms(started_at: datetime | None, completed_at: datetime | None) -> int | None:
    if started_at is None or completed_at is None:
        return None
    elapsed_ms = (_normalize_utc(completed_at) - _normalize_utc(started_at)).total_seconds() * 1000
    if elapsed_ms <= 0:
        return None
    return round(elapsed_ms)


def _message_time_to_first_token_ms(message: PromptMessageExtended) -> int | None:
    timing_payload = _message_timing_payload(message)
    for key in (
        "ttft_ms",
        "time_to_first_token_ms",
        "first_token_ms",
        "first_token_latency_ms",
        "time_to_response_ms",
    ):
        value = _milliseconds_field(timing_payload, key)
        if value is not None:
            return value
    return None


@dataclass(slots=True)
class _TurnState:
    turn_id: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    llm_duration_ms: int = 0
    time_to_first_token_ms: int | None = None
    last_agent_message: str | None = None

    def observe_assistant_message(
        self,
        message: PromptMessageExtended,
        message_timestamp: datetime | None,
    ) -> None:
        if message_timestamp is not None:
            self.completed_at = message_timestamp

        timing_payload = _message_timing_payload(message)
        duration_ms = _milliseconds_field(timing_payload, "duration_ms")
        if duration_ms is not None:
            self.llm_duration_ms += duration_ms

        if self.time_to_first_token_ms is None:
            self.time_to_first_token_ms = _message_time_to_first_token_ms(message)

    def duration_ms(self) -> int | None:
        duration_ms = _elapsed_ms(self.started_at, self.completed_at)
        if duration_ms is not None:
            return duration_ms
        return self.llm_duration_ms if self.llm_duration_ms > 0 else None


def _initial_session_records(
    resolved: ResolvedSessionExport,
    meta: _TraceMeta,
    sanitization: _TextSanitization | None,
) -> list[dict[str, object]]:
    records = [
        _record(
            "session_meta",
            _session_meta_payload(resolved, meta, sanitization=sanitization),
            timestamp=resolved.snapshot.created_at,
        )
    ]
    agent_snapshot = resolved.snapshot.continuation.agents[resolved.agent_name]
    if agent_snapshot.resolved_prompt:
        records.append(
            _record(
                "response_item",
                _developer_message_item(
                    agent_snapshot.resolved_prompt,
                    sanitization=sanitization,
                ),
            )
        )
    return records


def _start_turn_records(
    resolved: ResolvedSessionExport,
    meta: _TraceMeta,
    turn: _TurnState,
    user_message: PromptMessageExtended | None,
    sanitization: _TextSanitization | None,
) -> list[dict[str, object]]:
    records = [
        _record(
            "event_msg",
            _turn_started_payload(
                turn.turn_id,
                model_context_window=meta.model_context_window,
                started_at=turn.started_at,
            ),
            timestamp=turn.started_at,
        )
    ]
    if user_message is not None:
        records.append(
            _record(
                "event_msg",
                _user_event_payload(user_message, sanitization=sanitization),
                timestamp=turn.started_at,
            )
        )
    records.append(
        _record(
            "turn_context",
            _turn_context_payload(
                resolved,
                turn_id=turn.turn_id,
                meta=meta,
                turn_timestamp=turn.started_at,
            ),
        )
    )
    return records


def _finish_turn_record(
    turn: _TurnState,
    sanitization: _TextSanitization | None,
) -> dict[str, object]:
    completed_at = turn.completed_at
    return _record(
        "event_msg",
        _turn_complete_payload(
            turn.turn_id,
            turn.last_agent_message,
            completed_at=completed_at,
            duration_ms=turn.duration_ms(),
            time_to_first_token_ms=turn.time_to_first_token_ms,
            sanitization=sanitization,
        ),
        timestamp=completed_at,
    )


def _observe_turn_message(
    turn: _TurnState,
    message: PromptMessageExtended,
    message_timestamp: datetime | None,
) -> None:
    if message.role != "assistant":
        return
    turn.observe_assistant_message(message, message_timestamp)
    texts = _message_texts(message.content)
    if texts:
        turn.last_agent_message = texts[-1]


def _message_records(
    message: PromptMessageExtended,
    message_timestamp: datetime | None,
    meta: _TraceMeta,
    sanitization: _TextSanitization | None,
) -> list[dict[str, object]]:
    records = [
        _record(
            "response_item",
            item,
            timestamp=message_timestamp,
        )
        for item in _response_items(message, sanitization=sanitization)
    ]
    if message.role != "assistant":
        return records

    token_count = _token_count_payload(
        message,
        model_context_window=meta.model_context_window,
    )
    if token_count is not None:
        records.append(
            _record(
                "event_msg",
                token_count,
                timestamp=message_timestamp,
            )
        )
    return records


_PRIVACY_FILTER_LIMITATIONS = [
    "file_paths_not_redacted",
    "directory_names_not_redacted",
    "filenames_not_redacted",
    "resource_urls_not_redacted",
    "binary_payloads_not_redacted",
    "images_audio_not_redacted",
]


def _privacy_filter_metadata(summary: RedactionSummary) -> dict[str, object]:
    redactions: dict[str, object] = {
        "total": summary.total,
        "by_label": summary.by_label,
    }
    if summary.elapsed is not None:
        redactions["elapsed_seconds"] = round(summary.elapsed.total_seconds(), 3)
    metadata: dict[str, object] = {
        "applied": True,
        "mode": "content-only",
        "redactions": redactions,
        "limitations": list(_PRIVACY_FILTER_LIMITATIONS),
    }
    if summary.model is not None:
        metadata["backend"] = summary.model.backend
        model: dict[str, object] = {}
        if summary.model.repo_id is not None:
            model["repo_id"] = summary.model.repo_id
        if summary.model.revision is not None:
            model["revision"] = summary.model.revision
        if summary.model.variant is not None:
            model["variant"] = summary.model.variant
        if model:
            metadata["model"] = model
    return metadata


def _add_privacy_filter_metadata(
    records: list[dict[str, object]],
    summary: RedactionSummary,
) -> None:
    if not records:
        return
    payload = records[0].get("payload")
    if isinstance(payload, dict):
        payload_map = cast("dict[str, object]", payload)
        payload_map["privacy_filter"] = _privacy_filter_metadata(summary)


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

    def __init__(
        self,
        sanitizer: TraceSanitizer | None = None,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._sanitizer = sanitizer
        self._progress_callback = progress_callback

    def write(self, resolved: ResolvedSessionExport, output_path: Path) -> ExportResult:
        sanitization = None
        if self._sanitizer is not None:
            plan = _TraceSanitizationPlan()
            self._records(resolved, sanitization=plan)
            sanitization = _TraceSanitization(
                self._sanitizer,
                total_texts=plan.unique_text_count,
                total_characters=plan.total_characters,
                progress_callback=self._progress_callback,
            )
        records = list(self._records(resolved, sanitization=sanitization))
        redaction = sanitization.summary() if sanitization is not None else None
        if redaction is not None:
            _add_privacy_filter_metadata(records, redaction)
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
            redaction=redaction,
        )

    def _records(
        self,
        resolved: ResolvedSessionExport,
        *,
        sanitization: _TextSanitization | None = None,
    ) -> list[dict[str, object]]:
        meta = _trace_meta(resolved)
        turn_timestamps = _turn_timestamps(resolved)
        records = _initial_session_records(resolved, meta, sanitization)

        turn_counter = 0
        current_turn: _TurnState | None = None

        def start_turn(user_message: PromptMessageExtended | None) -> None:
            nonlocal turn_counter, current_turn
            turn_timestamp = (
                turn_timestamps[turn_counter] if turn_counter < len(turn_timestamps) else None
            )
            turn_counter += 1
            current_turn = _TurnState(
                turn_id=f"turn-{turn_counter}",
                started_at=turn_timestamp,
            )
            records.extend(
                _start_turn_records(
                    resolved,
                    meta,
                    current_turn,
                    user_message,
                    sanitization,
                )
            )

        def finish_turn() -> None:
            nonlocal current_turn
            if current_turn is None:
                return
            records.append(_finish_turn_record(current_turn, sanitization))
            current_turn = None

        for message, message_timestamp in zip(
            resolved.history, resolved.message_timestamps, strict=True
        ):
            if _is_turn_start(message):
                finish_turn()
                start_turn(message)
            elif current_turn is None:
                start_turn(None)

            if current_turn is not None:
                _observe_turn_message(current_turn, message, message_timestamp)
            records.extend(_message_records(message, message_timestamp, meta, sanitization))

        finish_turn()
        return records
