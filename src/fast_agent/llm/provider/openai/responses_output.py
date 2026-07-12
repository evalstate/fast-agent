from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolRequest, CallToolRequestParams, ContentBlock, TextContent
from openai.types.responses import ResponseReasoningItem
from pydantic_core import from_json

from fast_agent.core.logging.json_serializer import snapshot_json_value
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.provider.openai.tool_event_helpers import (
    first_nonempty_string,
    item_type_is_responses_function_tool_call,
    responses_item_tool_use_id,
)
from fast_agent.llm.provider.openai.web_tools import (
    extract_url_citation_payload,
    normalize_web_search_call_payload,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import CacheUsage, TurnUsage
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.tools.apply_patch_tool import APPLY_PATCH_INPUT_FIELD
from fast_agent.types.assistant_message_phase import (
    AssistantMessagePhase,
    coerce_assistant_message_phase,
)
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.reasoning_chunk_join import (
    join_reasoning_segments,
    normalize_reasoning_summary_parts,
)
from fast_agent.utils.text import strip_casefold


class ResponsesOutputMixin:
    if TYPE_CHECKING:
        from fast_agent.core.logging.logger import Logger

        logger: Logger
        _tool_call_id_map: dict[str, str]
        _tool_name_map: dict[str, str]
        _tool_kind_map: dict[str, str]
        _seen_tool_call_ids: set[str]
        _tool_call_diagnostics: dict[str, Any] | None

        def _finalize_turn_usage(self, usage: TurnUsage) -> None: ...

        def _normalize_tool_ids(self, tool_use_id: str | None) -> tuple[str, str]: ...

        @property
        def provider(self) -> Provider: ...

    def _consume_tool_call_diagnostics(self) -> dict[str, Any] | None:
        diagnostics = getattr(self, "_tool_call_diagnostics", None)
        self._tool_call_diagnostics = None
        return diagnostics

    def _is_provider_managed_function_call(self, name: str) -> bool:
        del name
        return False

    def _seen_tool_call_ids_state(self) -> set[str]:
        seen = getattr(self, "_seen_tool_call_ids", None)
        if seen is None:
            seen = set()
            self._seen_tool_call_ids = seen

        if self._tool_call_id_map:
            seen.update(call_id for call_id in self._tool_call_id_map.values() if call_id)

        return seen

    def _tool_kind_state(self) -> dict[str, str]:
        tool_kind_map = getattr(self, "_tool_kind_map", None)
        if isinstance(tool_kind_map, dict):
            return tool_kind_map

        tool_kind_map = {}
        self._tool_kind_map = tool_kind_map
        return tool_kind_map

    @staticmethod
    def _coerce_assistant_message_phase(
        raw_phase: object,
    ) -> AssistantMessagePhase | None:
        return coerce_assistant_message_phase(raw_phase)

    @staticmethod
    def _extract_phase_message_text(content: object) -> str | None:
        if not isinstance(content, Sequence) or isinstance(content, str):
            return None

        text_segments: list[str] = []
        for part in content:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                text_segments.append(part_text)

        combined_text = "".join(text_segments).strip()
        return combined_text or None

    @classmethod
    def _print_phase_message(
        cls,
        output_item: Any,
        serialized_item: Mapping[str, Any] | None = None,
    ) -> None:
        phase = cls._coerce_assistant_message_phase(getattr(output_item, "phase", None))
        if phase is None:
            return

        content_text = cls._extract_phase_message_text(getattr(output_item, "content", None))
        if content_text is None and serialized_item is not None:
            content_text = json.dumps(dict(serialized_item), ensure_ascii=False)
        if not content_text:
            return

    @classmethod
    def _model_dump_mapping(cls, item: Any) -> dict[str, Any] | None:
        model_dump = getattr(item, "model_dump", None)
        if not callable(model_dump):
            return None

        try:
            payload = model_dump(mode="json", by_alias=True, exclude_none=True)
        except TypeError:
            payload = model_dump()
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        return payload

    @classmethod
    def _normalize_serialized_message_phase(cls, payload: dict[str, Any]) -> dict[str, Any]:
        raw_phase = payload.get("phase")
        normalized_phase = cls._coerce_assistant_message_phase(raw_phase)
        if normalized_phase is not None:
            payload["phase"] = normalized_phase
        else:
            payload.pop("phase", None)
        return payload

    @classmethod
    def _serialize_assistant_content_part(cls, part: Any) -> dict[str, Any] | None:
        part_payload = cls._model_dump_mapping(part)
        if part_payload is not None:
            return part_payload

        part_type = getattr(part, "type", None)
        if not isinstance(part_type, str) or not part_type:
            return None

        payload: dict[str, Any] = {"type": part_type}
        part_text = getattr(part, "text", None)
        if isinstance(part_text, str):
            payload["text"] = part_text
        return payload

    @classmethod
    def _serialize_model_dumped_assistant_message(cls, item: Any) -> dict[str, Any] | None:
        payload = cls._model_dump_mapping(item)
        if payload is None:
            return None
        return cls._normalize_serialized_message_phase(payload)

    @classmethod
    def _serialize_assistant_message_item(cls, item: Any) -> dict[str, Any] | None:
        payload = cls._serialize_model_dumped_assistant_message(item)
        if payload is not None:
            return payload

        if getattr(item, "type", None) != "message":
            return None

        payload = cls._assistant_message_base_payload(item)
        serialized_content = [
            part_payload
            for part in getattr(item, "content", []) or []
            if (part_payload := cls._serialize_assistant_content_part(part)) is not None
        ]
        if serialized_content:
            payload["content"] = serialized_content

        return payload

    @classmethod
    def _assistant_message_base_payload(cls, item: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "message",
            "role": getattr(item, "role", None) or "assistant",
        }
        item_id = getattr(item, "id", None)
        if item_id:
            payload["id"] = item_id
        status = getattr(item, "status", None)
        if status:
            payload["status"] = status
        phase = cls._coerce_assistant_message_phase(getattr(item, "phase", None))
        if phase is not None:
            payload["phase"] = phase
        return payload

    def _extract_raw_assistant_message_items(
        self,
        response: Any,
    ) -> tuple[list[ContentBlock], AssistantMessagePhase | None]:
        serialized_items: list[dict[str, Any]] = []
        phases: list[AssistantMessagePhase] = []

        for output_item in getattr(response, "output", []) or []:
            if getattr(output_item, "type", None) != "message":
                continue

            phase = self._coerce_assistant_message_phase(getattr(output_item, "phase", None))
            if phase is not None:
                phases.append(phase)

            serialized_item = self._serialize_assistant_message_item(output_item)
            if serialized_item is not None:
                self._print_phase_message(output_item, serialized_item)
                serialized_items.append(serialized_item)

        if not serialized_items:
            return [], None

        blocks: list[ContentBlock] = [
            TextContent(type="text", text=json.dumps(payload)) for payload in serialized_items
        ]
        if not phases or len(phases) != len(serialized_items):
            return blocks, None

        unique_phases = set(phases)
        message_phase = phases[0] if len(unique_phases) == 1 else None
        return blocks, message_phase

    def _record_usage(self, usage: Any, model_name: str) -> None:
        try:
            provider_value = getattr(self, "provider", Provider.RESPONSES)
            provider = (
                provider_value if isinstance(provider_value, Provider) else Provider.RESPONSES
            )
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)
            cached_tokens = 0
            details = getattr(usage, "input_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", 0) or 0
            reasoning_tokens = 0
            output_details = getattr(usage, "output_tokens_details", None)
            if output_details is not None:
                reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0

            cache_usage = CacheUsage(cache_hit_tokens=cached_tokens)
            turn_usage = TurnUsage(
                provider=provider,
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_usage=cache_usage,
                reasoning_tokens=reasoning_tokens,
                raw_usage=snapshot_json_value(usage),
            )
            self._finalize_turn_usage(turn_usage)
        except Exception as e:
            self.logger.warning(f"Failed to track Responses usage: {e}")

    def _extract_tool_calls(self, response: Any) -> dict[str, CallToolRequest] | None:
        tool_calls: dict[str, CallToolRequest] = {}
        duplicate_call_ids: list[str] = []
        duplicate_call_names: dict[str, str] = {}
        seen_tool_call_ids = self._seen_tool_call_ids_state()
        tool_kind_map = self._tool_kind_state()
        raw_function_call_count = 0
        model_name = getattr(response, "model", None)
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if not isinstance(item_type, str) or not item_type_is_responses_function_tool_call(
                item_type
            ):
                continue
            raw_function_call_count += 1
            tool_kind = "custom" if item_type == "custom_tool_call" else "function"
            item_id = first_nonempty_string(getattr(item, "id", None))
            call_id = first_nonempty_string(getattr(item, "call_id", None))
            name = first_nonempty_string(getattr(item, "name", None)) or "tool"
            if self._is_provider_managed_function_call(name):
                continue
            arguments = self._tool_call_arguments(item, item_type=item_type)
            # Use call_id as the primary tool identifier.
            #
            # Streaming tool notifications (and tool results) use call_id, while id can
            # refer to the output item ("fc_*"). If we prefer id here, ACP clients can
            # end up with duplicated/stuck tool cards: a stream-start for call_id and
            # execution/completion for id. Aligning on call_id keeps tool_use_id stable
            # across streaming → execution → completion.
            tool_use_id = call_id or item_id or f"fc_{len(tool_calls)}"
            if not call_id:
                call_id = self._normalize_tool_ids(tool_use_id)[1]

            if call_id in seen_tool_call_ids:
                duplicate_call_ids.append(call_id)
                if call_id not in duplicate_call_names:
                    duplicate_call_names[call_id] = name
                continue

            self._record_extracted_tool_call(
                tool_use_id=tool_use_id,
                call_id=call_id,
                item_id=item_id,
                name=name,
                tool_kind=tool_kind,
                tool_kind_map=tool_kind_map,
                seen_tool_call_ids=seen_tool_call_ids,
            )
            tool_calls[tool_use_id] = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=name, arguments=arguments),
            )

        self._record_duplicate_tool_call_diagnostics(
            duplicate_call_ids=duplicate_call_ids,
            duplicate_call_names=duplicate_call_names,
            raw_function_call_count=raw_function_call_count,
            new_function_call_count=len(tool_calls),
            model_name=model_name,
        )

        return tool_calls or None

    @staticmethod
    def _tool_call_arguments(item: Any, *, item_type: str) -> dict[str, Any]:
        if item_type == "custom_tool_call":
            custom_input = getattr(item, "input", None)
            if isinstance(custom_input, str):
                return {APPLY_PATCH_INPUT_FIELD: custom_input}
            return {}

        arguments_raw = getattr(item, "arguments", None)
        if not arguments_raw:
            return {}
        try:
            arguments = from_json(arguments_raw, allow_partial=True)
        except Exception:
            return {}
        return arguments if isinstance(arguments, dict) else {}

    def _record_extracted_tool_call(
        self,
        *,
        tool_use_id: str,
        call_id: str,
        item_id: str | None,
        name: str,
        tool_kind: str,
        tool_kind_map: dict[str, str],
        seen_tool_call_ids: set[str],
    ) -> None:
        self._tool_call_id_map[tool_use_id] = call_id
        self._tool_name_map[tool_use_id] = name
        tool_kind_map[tool_use_id] = tool_kind
        tool_kind_map[call_id] = tool_kind
        if item_id:
            tool_kind_map[item_id] = tool_kind
        seen_tool_call_ids.add(call_id)

    def _record_duplicate_tool_call_diagnostics(
        self,
        *,
        duplicate_call_ids: list[str],
        duplicate_call_names: dict[str, str],
        raw_function_call_count: int,
        new_function_call_count: int,
        model_name: str | None,
    ) -> None:
        if not duplicate_call_ids:
            self._tool_call_diagnostics = None
            return

        duplicate_ids = sorted(set(duplicate_call_ids))
        diagnostics = {
            "kind": "duplicate_tool_calls_filtered",
            "duplicate_count": len(duplicate_call_ids),
            "duplicate_tool_call_ids": duplicate_ids,
            "raw_function_call_count": raw_function_call_count,
            "new_function_call_count": new_function_call_count,
        }
        self._tool_call_diagnostics = diagnostics
        logger = getattr(self, "logger", None)
        if logger is None:
            return

        logger.warning("Filtered duplicate Responses tool calls", data=diagnostics)
        agent_name = getattr(self, "name", None)
        for call_id in duplicate_ids:
            logger.info(
                "Filtered duplicate Responses tool call",
                data={
                    "progress_action": ProgressAction.CALLING_TOOL,
                    "agent_name": agent_name,
                    "model": model_name,
                    "tool_name": duplicate_call_names.get(call_id, "tool"),
                    "tool_use_id": call_id,
                    "tool_event": "stop",
                    "tool_terminal": True,
                },
            )

    def _map_response_stop_reason(self, response: Any) -> LlmStopReason:
        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) if details else None
            if reason == "max_output_tokens":
                return LlmStopReason.MAX_TOKENS
        return LlmStopReason.END_TURN

    def _extract_reasoning_summary(
        self, response: Any, streamed_summary: list[str]
    ) -> list[ContentBlock]:
        reasoning_blocks: list[ContentBlock] = []
        for output_item in getattr(response, "output", []) or []:
            if (
                not isinstance(output_item, ResponseReasoningItem)
                and getattr(output_item, "type", None) != "reasoning"
            ):
                continue
            summary = getattr(output_item, "summary", None) or []
            summary_parts: list[str] = []
            for part in summary:
                part_text = getattr(part, "text", None)
                if not isinstance(part_text, str) or not part_text:
                    continue
                summary_parts.append(part_text)
            summary_text = join_reasoning_segments(
                normalize_reasoning_summary_parts(summary_parts)
            )
            if summary_text.strip():
                reasoning_blocks.append(text_content(summary_text.strip()))
        if reasoning_blocks:
            return reasoning_blocks
        if streamed_summary:
            summary_text = join_reasoning_segments(
                normalize_reasoning_summary_parts(streamed_summary)
            ).strip()
            if summary_text:
                return [text_content(summary_text)]
        return []

    def _extract_encrypted_reasoning(self, response: Any) -> list[ContentBlock]:
        encrypted_blocks: list[ContentBlock] = []
        for output_item in getattr(response, "output", []) or []:
            if getattr(output_item, "type", None) != "reasoning":
                continue
            encrypted_content = getattr(output_item, "encrypted_content", None)
            if not encrypted_content:
                continue
            payload: dict[str, Any] = {
                "type": "reasoning",
                "encrypted_content": encrypted_content,
            }
            item_id = getattr(output_item, "id", None)
            if item_id:
                payload["id"] = item_id
            encrypted_blocks.append(TextContent(type="text", text=json.dumps(payload)))
        return encrypted_blocks

    @staticmethod
    def _web_citation_dedupe_key(
        payload: Mapping[str, Any],
    ) -> tuple[str, str] | tuple[str, str, str]:
        raw_url = payload.get("url")
        if isinstance(raw_url, str) and raw_url:
            return ("url", strip_casefold(raw_url))

        title_key = strip_casefold(str(payload.get("title") or ""))
        source_key = strip_casefold(str(payload.get("source") or ""))
        return ("meta", title_key, source_key)

    def _extract_web_search_metadata(
        self,
        response: Any,
    ) -> tuple[list[ContentBlock], list[ContentBlock]]:
        web_tool_payloads: list[ContentBlock] = []
        citation_payloads: list[ContentBlock] = []
        seen_citations: set[tuple[str, str] | tuple[str, str, str]] = set()

        def append_citation(payload: Mapping[str, Any]) -> None:
            key = self._web_citation_dedupe_key(payload)
            if key in seen_citations:
                return
            seen_citations.add(key)
            citation_payloads.append(TextContent(type="text", text=json.dumps(dict(payload))))

        for output_item in getattr(response, "output", []) or []:
            item_type = getattr(output_item, "type", None)
            if item_type == "web_search_call":
                tool_payload, call_citations = normalize_web_search_call_payload(output_item)
                if tool_payload is not None:
                    web_tool_payloads.append(
                        TextContent(type="text", text=json.dumps(tool_payload))
                    )
                for citation in call_citations:
                    append_citation(citation)
                continue

            if item_type != "message":
                continue

            for payload in self._extract_message_url_citation_payloads(output_item):
                append_citation(payload)

        return web_tool_payloads, citation_payloads

    @staticmethod
    def _extract_message_url_citation_payloads(
        output_item: Any,
    ) -> list[Mapping[str, Any]]:
        payloads: list[Mapping[str, Any]] = []
        for part in getattr(output_item, "content", []) or []:
            if getattr(part, "type", None) != "output_text":
                continue

            annotations = getattr(part, "annotations", None)
            if not isinstance(annotations, Sequence) or isinstance(annotations, str):
                continue

            for annotation in annotations:
                payload = extract_url_citation_payload(annotation)
                if payload is not None:
                    payloads.append(payload)
        return payloads

    @staticmethod
    def _normalize_tool_search_output_item(output_item: Any) -> dict[str, Any] | None:
        item_type = getattr(output_item, "type", None)
        if item_type not in {"tool_search_call", "tool_search_output"}:
            return None

        payload: dict[str, Any] = {
            "type": "server_tool_use",
            "provider_tool_type": item_type,
            "name": "tool_search",
        }
        item_id = first_nonempty_string(getattr(output_item, "id", None))
        if item_id is not None:
            payload["id"] = item_id
        status = first_nonempty_string(getattr(output_item, "status", None))
        if status is not None:
            payload["status"] = status
        execution = first_nonempty_string(getattr(output_item, "execution", None))
        if execution is not None:
            payload["execution"] = execution
        call_id = first_nonempty_string(getattr(output_item, "call_id", None))
        if call_id is not None:
            payload["call_id"] = call_id

        if item_type == "tool_search_call":
            arguments = getattr(output_item, "arguments", None)
            serialized_arguments = snapshot_json_value(arguments)
            if isinstance(serialized_arguments, Mapping):
                payload["input"] = dict(serialized_arguments)
            elif serialized_arguments is not None:
                payload["arguments"] = serialized_arguments
            return payload

        tools = getattr(output_item, "tools", None)
        serialized_tools = snapshot_json_value(tools)
        if isinstance(serialized_tools, Sequence) and not isinstance(serialized_tools, str):
            payload["tools"] = list(serialized_tools)
            payload["tool_count"] = len(serialized_tools)
        return payload

    def _extract_tool_search_metadata(
        self,
        response: Any,
    ) -> list[ContentBlock]:
        payloads: list[ContentBlock] = []
        for output_item in getattr(response, "output", []) or []:
            payload = self._normalize_tool_search_output_item(output_item)
            if payload is not None:
                payloads.append(TextContent(type="text", text=json.dumps(payload)))
        return payloads

    @classmethod
    def _serialize_mcp_list_tools_item(cls, output_item: Any) -> dict[str, Any] | None:
        if getattr(output_item, "type", None) != "mcp_list_tools":
            return None

        payload = cls._model_dump_mapping(output_item)
        if payload is not None:
            return payload

        payload: dict[str, Any] = {"type": "mcp_list_tools"}
        item_id = first_nonempty_string(getattr(output_item, "id", None))
        if item_id is not None:
            payload["id"] = item_id
        server_label = first_nonempty_string(getattr(output_item, "server_label", None))
        if server_label is not None:
            payload["server_label"] = server_label
        tools = snapshot_json_value(getattr(output_item, "tools", None))
        if isinstance(tools, Sequence) and not isinstance(tools, str):
            payload["tools"] = list(tools)
        return payload

    def _extract_raw_mcp_list_tools_items(
        self,
        response: Any,
    ) -> list[ContentBlock]:
        items: list[ContentBlock] = []
        for output_item in getattr(response, "output", []) or []:
            payload = self._serialize_mcp_list_tools_item(output_item)
            if payload is not None:
                items.append(TextContent(type="text", text=json.dumps(payload)))
        return items

    @staticmethod
    def _normalize_provider_mcp_output_item(output_item: Any) -> dict[str, Any] | None:
        item_type = getattr(output_item, "type", None)
        if item_type not in {"mcp_list_tools", "mcp_call"}:
            return None

        payload: dict[str, Any] = {
            "type": "mcp_tool_use",
            "provider_tool_type": item_type,
            "name": first_nonempty_string(
                getattr(output_item, "name", None),
                getattr(output_item, "tool_name", None),
            )
            or item_type,
        }
        item_id = responses_item_tool_use_id(output_item)
        if item_id is not None:
            payload["id"] = item_id
        status = first_nonempty_string(getattr(output_item, "status", None))
        if status is not None:
            payload["status"] = status
        server_label = first_nonempty_string(getattr(output_item, "server_label", None))
        if server_label is not None:
            payload["server_name"] = server_label
        arguments = first_nonempty_string(getattr(output_item, "arguments", None))
        if arguments is not None:
            payload["arguments"] = arguments
            try:
                parsed_arguments = from_json(arguments, allow_partial=True)
            except Exception:
                parsed_arguments = None
            if isinstance(parsed_arguments, Mapping):
                payload["input"] = dict(parsed_arguments)
        return payload

    @staticmethod
    def _normalize_provider_mcp_result_item(output_item: Any) -> dict[str, Any] | None:
        if getattr(output_item, "type", None) != "mcp_call":
            return None

        tool_use_id = responses_item_tool_use_id(output_item)
        if tool_use_id is None:
            return None

        status = getattr(output_item, "status", None)
        output = snapshot_json_value(getattr(output_item, "output", None))
        if output is None and not isinstance(status, str):
            return None

        content_text: str | None = None
        if isinstance(output, str):
            content_text = output
        elif output is not None:
            content_text = json.dumps(output, ensure_ascii=False, sort_keys=True)

        content: list[dict[str, str]] = []
        if content_text:
            content.append({"type": "text", "text": content_text})

        return {
            "type": "mcp_tool_result",
            "tool_use_id": tool_use_id,
            "is_error": status == "failed",
            "content": content,
        }

    def _extract_provider_mcp_metadata(
        self,
        response: Any,
    ) -> list[ContentBlock]:
        payloads: list[ContentBlock] = []
        for output_item in getattr(response, "output", []) or []:
            item_type = getattr(output_item, "type", None)
            if item_type == "mcp_approval_request":
                raise RuntimeError(
                    "OpenAI MCP approval requests are not supported by fast-agent yet."
                )
            payload = self._normalize_provider_mcp_output_item(output_item)
            if payload is not None:
                payloads.append(TextContent(type="text", text=json.dumps(payload)))
            result_payload = self._normalize_provider_mcp_result_item(output_item)
            if result_payload is not None:
                payloads.append(TextContent(type="text", text=json.dumps(result_payload)))
        return payloads
