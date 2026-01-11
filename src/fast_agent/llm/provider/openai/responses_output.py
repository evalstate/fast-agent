from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolRequest, CallToolRequestParams, ContentBlock, TextContent
from openai.types.responses import ResponseReasoningItem
from pydantic_core import from_json

from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import CacheUsage, TurnUsage
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types.llm_stop_reason import LlmStopReason


class ResponsesOutputMixin:
    if TYPE_CHECKING:
        from fast_agent.core.logging.logger import Logger

        logger: Logger
        _tool_call_id_map: dict[str, str]

        def _finalize_turn_usage(self, usage: TurnUsage) -> None: ...

        def _normalize_tool_ids(self, tool_use_id: str | None) -> tuple[str, str]: ...

    def _record_usage(self, usage: Any, model_name: str) -> None:
        try:
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
                provider=Provider.RESPONSES,
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_usage=cache_usage,
                reasoning_tokens=reasoning_tokens,
                raw_usage=usage,
            )
            self._finalize_turn_usage(turn_usage)
        except Exception as e:
            self.logger.warning(f"Failed to track Responses usage: {e}")

    def _extract_tool_calls(self, response: Any) -> dict[str, CallToolRequest] | None:
        tool_calls: dict[str, CallToolRequest] = {}
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            item_id = getattr(item, "id", None)
            call_id = getattr(item, "call_id", None)
            name = getattr(item, "name", None) or "tool"
            arguments_raw = getattr(item, "arguments", None)
            if arguments_raw:
                try:
                    arguments = from_json(arguments_raw, allow_partial=True)
                except Exception:
                    arguments = {}
            else:
                arguments = {}
            tool_use_id = item_id or call_id or f"fc_{len(tool_calls)}"
            if not call_id:
                call_id = self._normalize_tool_ids(tool_use_id)[1]
            self._tool_call_id_map[tool_use_id] = call_id
            tool_calls[tool_use_id] = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=name, arguments=arguments),
            )
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
            if not isinstance(output_item, ResponseReasoningItem) and getattr(
                output_item, "type", None
            ) != "reasoning":
                continue
            summary = getattr(output_item, "summary", None) or []
            summary_text = "\n".join(
                part.text for part in summary if getattr(part, "text", None)
            )
            if summary_text.strip():
                reasoning_blocks.append(text_content(summary_text.strip()))
        if reasoning_blocks:
            return reasoning_blocks
        if streamed_summary:
            return [text_content("".join(streamed_summary))]
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
