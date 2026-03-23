from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolRequest, CallToolRequestParams, ContentBlock, TextContent
from openai.types.responses import ResponseReasoningItem
from pydantic_core import from_json

from fast_agent.core.logging.json_serializer import snapshot_json_value
from fast_agent.event_progress import ProgressAction
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

    def _seen_tool_call_ids_state(self) -> set[str]:
        seen = getattr(self, "_seen_tool_call_ids", None)
        if seen is None:
            seen = set()
            self._seen_tool_call_ids = seen

        if self._tool_call_id_map:
            seen.update(call_id for call_id in self._tool_call_id_map.values() if call_id)

        return seen

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
    def _serialize_assistant_message_item(cls, item: Any) -> dict[str, Any] | None:
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            try:
                payload = model_dump(mode="json", by_alias=True, exclude_none=True)
            except TypeError:
                payload = model_dump()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                raw_phase = payload.get("phase")
                normalized_phase = cls._coerce_assistant_message_phase(raw_phase)
                if normalized_phase is not None:
                    payload["phase"] = normalized_phase
                else:
                    payload.pop("phase", None)
                return payload

        if getattr(item, "type", None) != "message":
            return None

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

        serialized_content: list[dict[str, Any]] = []
        for part in getattr(item, "content", []) or []:
            part_model_dump = getattr(part, "model_dump", None)
            if callable(part_model_dump):
                try:
                    part_payload = part_model_dump(mode="json", by_alias=True, exclude_none=True)
                except TypeError:
                    part_payload = part_model_dump()
                except Exception:
                    part_payload = None
                if isinstance(part_payload, dict):
                    serialized_content.append(part_payload)
                    continue

            part_type = getattr(part, "type", None)
            if not isinstance(part_type, str) or not part_type:
                continue
            part_payload = {"type": part_type}
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                part_payload["text"] = part_text
            serialized_content.append(part_payload)

        if serialized_content:
            payload["content"] = serialized_content

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

        if not phases:
            return [], None

        unique_phases = set(phases)
        message_phase = phases[0] if len(unique_phases) == 1 else None
        blocks: list[ContentBlock] = [
            TextContent(type="text", text=json.dumps(payload)) for payload in serialized_items
        ]
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
        tool_kind_map = getattr(self, "_tool_kind_map", None)
        if not isinstance(tool_kind_map, dict):
            tool_kind_map = {}
            self._tool_kind_map = tool_kind_map
        raw_function_call_count = 0
        model_name = getattr(response, "model", None)
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", None)
            if item_type not in {"function_call", "custom_tool_call"}:
                continue
            raw_function_call_count += 1
            tool_kind = "custom" if item_type == "custom_tool_call" else "function"
            item_id = getattr(item, "id", None)
            call_id = getattr(item, "call_id", None)
            name = getattr(item, "name", None) or "tool"
            if item_type == "custom_tool_call":
                custom_input = getattr(item, "input", None)
                if isinstance(custom_input, str):
                    arguments = {APPLY_PATCH_INPUT_FIELD: custom_input}
                else:
                    arguments = {}
            else:
                arguments_raw = getattr(item, "arguments", None)
                if arguments_raw:
                    try:
                        arguments = from_json(arguments_raw, allow_partial=True)
                    except Exception:
                        arguments = {}
                else:
                    arguments = {}
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

            self._tool_call_id_map[tool_use_id] = call_id
            self._tool_name_map[tool_use_id] = name
            tool_kind_map[tool_use_id] = tool_kind
            tool_kind_map[call_id] = tool_kind
            if item_id:
                tool_kind_map[item_id] = tool_kind
            seen_tool_call_ids.add(call_id)
            tool_calls[tool_use_id] = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=name, arguments=arguments),
            )

        if duplicate_call_ids:
            duplicate_ids = sorted(set(duplicate_call_ids))
            self._tool_call_diagnostics = {
                "kind": "duplicate_tool_calls_filtered",
                "duplicate_count": len(duplicate_call_ids),
                "duplicate_tool_call_ids": duplicate_ids,
                "raw_function_call_count": raw_function_call_count,
                "new_function_call_count": len(tool_calls),
            }
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.warning(
                    "Filtered duplicate Responses tool calls",
                    data={
                        "duplicate_count": len(duplicate_call_ids),
                        "duplicate_tool_call_ids": duplicate_ids,
                        "raw_function_call_count": raw_function_call_count,
                        "new_function_call_count": len(tool_calls),
                    },
                )
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
                        },
                    )
        else:
            self._tool_call_diagnostics = None

        return tool_calls or None

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
            summary_text = "\n".join(part.text for part in summary if getattr(part, "text", None))
            if summary_text.strip():
                reasoning_blocks.append(text_content(summary_text.strip()))
        if reasoning_blocks:
            return reasoning_blocks
        if streamed_summary:
            summary_text = "".join(streamed_summary).strip()
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
            return ("url", raw_url.strip().lower())

        title_key = str(payload.get("title") or "").strip().lower()
        source_key = str(payload.get("source") or "").strip().lower()
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

            for part in getattr(output_item, "content", []) or []:
                if getattr(part, "type", None) != "output_text":
                    continue

                annotations = getattr(part, "annotations", None)
                if not isinstance(annotations, Sequence) or isinstance(annotations, str):
                    continue

                for annotation in annotations:
                    payload = extract_url_citation_payload(annotation)
                    if payload is not None:
                        append_citation(payload)

        return web_tool_payloads, citation_payloads
