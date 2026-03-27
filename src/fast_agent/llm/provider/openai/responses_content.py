from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from mcp.types import CallToolRequest, ContentBlock, EmbeddedResource

from fast_agent.constants import (
    OPENAI_ASSISTANT_MESSAGE_ITEMS,
    OPENAI_REASONING_ENCRYPTED,
    REASONING,
)
from fast_agent.mcp.helpers.content_helpers import (
    get_image_data,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
)
from fast_agent.mcp.mime_utils import is_image_mime_type, is_text_mime_type
from fast_agent.tools.apply_patch_tool import (
    extract_apply_patch_input,
)
from fast_agent.types.assistant_message_phase import coerce_assistant_message_phase

if TYPE_CHECKING:
    from fast_agent.core.logging.logger import Logger
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.types.assistant_message_phase import AssistantMessagePhase


class ResponsesContentMixin:
    if TYPE_CHECKING:
        logger: Logger
        _tool_call_id_map: dict[str, str]
        _tool_name_map: dict[str, str]
        _tool_kind_map: dict[str, str]

    def _tool_kind_state(self) -> dict[str, str]:
        tool_kind_map = getattr(self, "_tool_kind_map", None)
        if isinstance(tool_kind_map, dict):
            return tool_kind_map

        tool_kind_map = {}
        self._tool_kind_map = tool_kind_map
        return tool_kind_map

    def _resolve_tool_call_kind(
        self,
        *,
        tool_use_id: str | None,
        fc_id: str | None,
        call_id: str | None,
    ) -> str:
        tool_kind_map = self._tool_kind_state()
        for key in (tool_use_id, fc_id, call_id):
            if not key:
                continue
            kind = tool_kind_map.get(key)
            if kind in {"custom", "function"}:
                return kind
        return "function"

    def _record_tool_call_kind(
        self,
        *,
        tool_use_id: str,
        fc_id: str,
        call_id: str,
        kind: str,
    ) -> None:
        tool_kind_map = self._tool_kind_state()
        tool_kind_map[tool_use_id] = kind
        tool_kind_map[fc_id] = kind
        tool_kind_map[call_id] = kind

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for msg in messages:
            items.extend(self._convert_message_to_items(msg))
        return self._dedupe_input_items(items)

    def _dedupe_input_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_ids: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in items:
            item_id = item.get("id")
            if item_id:
                item_id_str = str(item_id)
                if item_id_str in seen_ids:
                    self.logger.debug(
                        "Dropping duplicate Responses item id",
                        duplicate_id=item_id_str,
                    )
                    continue
                seen_ids.add(item_id_str)
            deduped.append(item)
        return deduped

    def _convert_message_to_items(self, msg: PromptMessageExtended) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        items.extend(self._extract_encrypted_reasoning_items(msg.channels))

        raw_assistant_items = self._extract_assistant_message_items(msg)

        if msg.tool_results:
            items.extend(self._convert_tool_results(msg.tool_results))
            if raw_assistant_items:
                items.extend(raw_assistant_items)
            else:
                message_item = self._build_message_item(msg.role, msg.content, phase=msg.phase)
                if message_item:
                    items.append(message_item)
            return items

        if raw_assistant_items:
            items.extend(raw_assistant_items)
        else:
            message_item = self._build_message_item(msg.role, msg.content, phase=msg.phase)
            if message_item:
                items.append(message_item)

        if msg.tool_calls:
            items.extend(self._convert_tool_calls(msg.tool_calls))

        return items

    def _extract_assistant_message_items(
        self, msg: PromptMessageExtended
    ) -> list[dict[str, Any]]:
        if msg.role != "assistant" or not msg.channels:
            return []

        raw_blocks = msg.channels.get(OPENAI_ASSISTANT_MESSAGE_ITEMS)
        if not raw_blocks:
            return []

        items: list[dict[str, Any]] = []
        for block in raw_blocks:
            text = get_text(block)
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                self.logger.debug("Skipping malformed OpenAI assistant message item")
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "message":
                continue
            payload = dict(payload)
            payload["role"] = "assistant"
            phase = payload.get("phase")
            if phase is None and msg.phase is not None:
                payload["phase"] = msg.phase
            normalized_phase = coerce_assistant_message_phase(payload.get("phase"))
            if normalized_phase is not None:
                payload["phase"] = normalized_phase
            else:
                payload.pop("phase", None)
            items.append(payload)
        return items

    def _extract_encrypted_reasoning_items(
        self, channels: Mapping[str, Iterable[ContentBlock]] | None
    ) -> list[dict[str, Any]]:
        if not channels:
            return []
        encrypted_blocks = channels.get(OPENAI_REASONING_ENCRYPTED)
        if not encrypted_blocks:
            return []

        summary = self._build_reasoning_summary_payload(channels)
        items: list[dict[str, Any]] = []
        for block in encrypted_blocks:
            text = get_text(block)
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                self.logger.debug("Skipping malformed encrypted reasoning block")
                continue
            if isinstance(data, dict) and data.get("encrypted_content"):
                item = dict(data)
                item.setdefault("type", "reasoning")
                if item.get("summary") is None:
                    item["summary"] = summary
                items.append(item)
        return items

    def _build_reasoning_summary_payload(
        self, channels: Mapping[str, Iterable[ContentBlock]] | None
    ) -> list[dict[str, str]]:
        if not channels:
            return []
        reasoning_blocks = channels.get(REASONING) or []
        summary_texts: list[str] = []
        for block in reasoning_blocks:
            text = get_text(block)
            if text:
                summary_texts.append(text)
        summary_text = "\n".join(summary_texts).strip()
        if not summary_text:
            return []
        return [{"type": "summary_text", "text": summary_text}]

    @staticmethod
    def _content_mime_type(content: ContentBlock) -> str | None:
        mime_type = getattr(content, "mimeType", None)
        if isinstance(content, EmbeddedResource):
            mime_type = getattr(content.resource, "mimeType", None)
        return mime_type

    @staticmethod
    def _content_filename(content: ContentBlock) -> str | None:
        uri = getattr(content, "uri", None)
        if isinstance(content, EmbeddedResource):
            uri = getattr(content.resource, "uri", None)
        if not uri:
            return None
        uri_str = str(uri)
        filename = uri_str.rsplit("/", 1)[-1] if "/" in uri_str else uri_str
        return filename or None

    def _content_to_input_text_part(self, content: ContentBlock) -> dict[str, Any] | None:
        if not is_resource_content(content):
            return None

        mime_type = self._content_mime_type(content) or "text/plain"
        if not is_text_mime_type(mime_type):
            return None

        text = get_text(content)
        if text is None:
            return None

        filename = self._content_filename(content) or "resource"
        return {
            "type": "input_text",
            "text": (
                f'<fastagent:file title="{filename}" mimetype="{mime_type}">\n'
                f"{text}\n"
                f"</fastagent:file>"
            ),
        }

    def _content_to_input_part(self, content: ContentBlock) -> dict[str, Any] | None:
        mime_type = self._content_mime_type(content)
        data = get_image_data(content)
        if data:
            if mime_type and is_image_mime_type(mime_type):
                return {"type": "input_image", "image_url": f"data:{mime_type};base64,{data}"}
            if mime_type:
                input_part: dict[str, Any] = {"type": "input_file", "file_data": data}
                filename = self._content_filename(content)
                if filename:
                    input_part["filename"] = filename
                return input_part
            return None

        if is_resource_content(content):
            resource_uri = get_resource_uri(content)
            if resource_uri:
                if mime_type and is_image_mime_type(mime_type):
                    return {"type": "input_image", "image_url": resource_uri}
                return {"type": "input_file", "file_url": resource_uri}
        if is_resource_link(content):
            resource_uri = getattr(content, "uri", None)
            if resource_uri:
                if mime_type and is_image_mime_type(mime_type):
                    return {"type": "input_image", "image_url": str(resource_uri)}
                return {"type": "input_file", "file_url": str(resource_uri)}

        return None

    def _normalize_tool_ids(self, tool_use_id: str | None) -> tuple[str, str]:
        tool_use_id = tool_use_id or ""
        if tool_use_id.startswith("fc"):
            call_id = self._tool_call_id_map.get(tool_use_id)
            if call_id:
                return tool_use_id, call_id
            suffix = tool_use_id[3:] if tool_use_id.startswith("fc_") else tool_use_id[2:]
            call_id = f"call_{suffix}" if suffix else f"call_{tool_use_id}"
            return tool_use_id, call_id
        if tool_use_id.startswith("call_"):
            suffix = tool_use_id[len("call_") :]
            fc_id = f"fc_{suffix}" if suffix else f"fc_{tool_use_id}"
            return fc_id, tool_use_id
        return f"fc_{tool_use_id}", f"call_{tool_use_id}"

    @staticmethod
    def _normalize_text_format(response_format: Any) -> Any:
        if not isinstance(response_format, dict):
            return response_format
        if response_format.get("type") != "json_schema":
            return response_format

        normalized: dict[str, Any] = {"type": "json_schema"}
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            if "name" in json_schema:
                normalized["name"] = json_schema["name"]
            if "schema" in json_schema:
                normalized["schema"] = json_schema["schema"]
            if "strict" in json_schema:
                normalized["strict"] = json_schema["strict"]

        for key in ("name", "schema", "strict"):
            if key in response_format:
                normalized[key] = response_format[key]

        return normalized

    def _build_message_item(
        self,
        role: str,
        content: list[ContentBlock],
        *,
        phase: AssistantMessagePhase | None = None,
    ) -> dict[str, Any] | None:
        if not content:
            return None
        parts = self._convert_content_parts(content, role)
        if not parts:
            return None
        item: dict[str, Any] = {
            "type": "message",
            "role": role,
            "content": parts,
        }
        if role == "assistant" and phase is not None:
            item["phase"] = phase
        return item

    def _convert_content_parts(
        self, content: list[ContentBlock], role: str
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        text_type = "output_text" if role == "assistant" else "input_text"

        for item in content:
            if is_text_content(item):
                text = get_text(item) or ""
                parts.append({"type": text_type, "text": text})
                continue

            if is_resource_content(item):
                text_part = self._content_to_input_text_part(item)
                if text_part:
                    parts.append(text_part)
                    continue

            if is_image_content(item) or is_resource_content(item) or is_resource_link(item):
                input_part = self._content_to_input_part(item)
                if input_part:
                    parts.append(input_part)
                    continue

            if is_resource_link(item):
                name = getattr(item, "name", None) or "resource"
                uri = getattr(item, "uri", None)
                if uri:
                    parts.append({"type": text_type, "text": f"[{name}]({uri})"})
                    continue

            resource_uri = get_resource_uri(item)
            if resource_uri:
                parts.append({"type": text_type, "text": f"[Resource]({resource_uri})"})
                continue

            parts.append({"type": text_type, "text": f"[Unsupported content: {type(item).__name__}]"})

        return parts

    def _content_to_image_url(self, item: ContentBlock) -> str | None:
        data = get_image_data(item)
        if not data:
            if is_resource_link(item):
                mime_type = self._content_mime_type(item)
                uri = getattr(item, "uri", None)
                if uri and mime_type and is_image_mime_type(mime_type):
                    return str(uri)
            return None
        mime_type = getattr(item, "mimeType", None)
        if not mime_type and is_resource_content(item):
            resource = getattr(item, "resource", None)
            mime_type = getattr(resource, "mimeType", None) if resource else None
        if not mime_type or not is_image_mime_type(mime_type):
            return None
        return f"data:{mime_type};base64,{data}"

    def _convert_tool_calls(self, tool_calls: dict[str, CallToolRequest]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, (tool_use_id, request) in enumerate(tool_calls.items()):
            tool_use_id = tool_use_id or f"tool_{index}"
            params = getattr(request, "params", None)
            name = getattr(params, "name", None) or "tool"
            arguments = getattr(params, "arguments", None) or {}
            fc_id, call_id = self._normalize_tool_ids(tool_use_id)
            self._tool_call_id_map[tool_use_id] = call_id
            self._tool_name_map[tool_use_id] = name
            tool_kind = self._resolve_tool_call_kind(
                tool_use_id=tool_use_id,
                fc_id=fc_id,
                call_id=call_id,
            )
            self._record_tool_call_kind(
                tool_use_id=tool_use_id,
                fc_id=fc_id,
                call_id=call_id,
                kind=tool_kind,
            )
            if tool_kind == "custom":
                patch_input = extract_apply_patch_input(arguments)
                items.append(
                    {
                        "type": "custom_tool_call",
                        "call_id": call_id,
                        "name": name,
                        "input": patch_input or "",
                    }
                )
                continue

            items.append(
                {
                    "type": "function_call",
                    "id": fc_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            )
        return items

    def _convert_tool_results(
        self, tool_results: dict[str, Any]
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, (tool_use_id, result) in enumerate(tool_results.items()):
            tool_use_id = tool_use_id or f"tool_{index}"
            fc_id, normalized_call_id = self._normalize_tool_ids(tool_use_id)
            call_id = self._tool_call_id_map.get(tool_use_id)
            if not call_id:
                call_id = normalized_call_id
            output = self._tool_result_to_text(result)
            tool_kind = self._resolve_tool_call_kind(
                tool_use_id=tool_use_id,
                fc_id=fc_id,
                call_id=call_id,
            )
            self._record_tool_call_kind(
                tool_use_id=tool_use_id,
                fc_id=fc_id,
                call_id=call_id,
                kind=tool_kind,
            )
            output_type = (
                "custom_tool_call_output" if tool_kind == "custom" else "function_call_output"
            )
            items.append(
                {
                    "type": output_type,
                    "call_id": call_id,
                    "output": output,
                }
            )
            attachment_parts = self._tool_result_to_input_parts(result)
            if attachment_parts:
                items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": attachment_parts,
                    }
                )
        return items

    def _tool_result_to_text(self, result: Any) -> str:
        contents = getattr(result, "content", None) or []
        chunks: list[str] = []
        for item in contents:
            text = get_text(item)
            if text is not None:
                chunks.append(text)
                continue
            if is_image_content(item) or is_resource_content(item) or is_resource_link(item):
                image_url = self._content_to_image_url(item)
                if image_url:
                    chunks.append(f"![Image]({image_url})")
                    continue
                input_part = self._content_to_input_part(item)
                if input_part and input_part.get("type") == "input_file":
                    file_url = input_part.get("file_url")
                    if isinstance(file_url, str):
                        chunks.append(f"[Resource]({file_url})")
                        continue
            resource_uri = get_resource_uri(item)
            if resource_uri:
                chunks.append(f"[Resource]({resource_uri})")
                continue
            if is_resource_link(item):
                uri = getattr(item, "uri", None)
                if uri:
                    chunks.append(f"[Resource]({uri})")
                    continue
            chunks.append(f"[Unsupported content: {type(item).__name__}]")
        return "\n".join(chunk for chunk in chunks if chunk)

    def _tool_result_to_input_parts(self, result: Any) -> list[dict[str, Any]]:
        contents = getattr(result, "content", None) or []
        parts: list[dict[str, Any]] = []
        for item in contents:
            if is_image_content(item) or is_resource_content(item) or is_resource_link(item):
                input_part = self._content_to_input_part(item)
                if input_part:
                    parts.append(input_part)
        return parts
