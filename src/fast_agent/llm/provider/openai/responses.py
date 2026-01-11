import asyncio
import json
from typing import Any, Iterable, Mapping

from mcp import Tool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)
from openai import APIError, AsyncOpenAI, AuthenticationError, DefaultAioHttpClient
from openai.types.responses import (
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from pydantic_core import from_json

from fast_agent.constants import OPENAI_REASONING_ENCRYPTED, REASONING
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.llm.usage_tracking import CacheUsage, TurnUsage
from fast_agent.mcp.helpers.content_helpers import (
    get_image_data,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
    text_content,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason

_logger = get_logger(__name__)

DEFAULT_RESPONSES_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "medium"


class ResponsesLLM(FastAgentLLM[dict[str, Any], Any]):
    """LLM implementation for OpenAI's Responses models."""

    RESPONSES_EXCLUDE_FIELDS = {
        FastAgentLLM.PARAM_MESSAGES,
        FastAgentLLM.PARAM_MODEL,
        FastAgentLLM.PARAM_MAX_TOKENS,
        FastAgentLLM.PARAM_SYSTEM_PROMPT,
        FastAgentLLM.PARAM_STOP_SEQUENCES,
        FastAgentLLM.PARAM_USE_HISTORY,
        FastAgentLLM.PARAM_MAX_ITERATIONS,
        FastAgentLLM.PARAM_TEMPLATE_VARS,
        FastAgentLLM.PARAM_MCP_METADATA,
        FastAgentLLM.PARAM_PARALLEL_TOOL_CALLS,
    }

    def __init__(self, provider: Provider = Provider.RESPONSES, **kwargs) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=provider, **kwargs)
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        self._reasoning_effort = kwargs.get("reasoning_effort", None)
        if self.context and self.context.config and self.context.config.openai:
            if self._reasoning_effort is None and hasattr(
                self.context.config.openai, "reasoning_effort"
            ):
                self._reasoning_effort = self.context.config.openai.reasoning_effort

        chosen_model = self.default_request_params.model if self.default_request_params else None
        self._reasoning_mode = ModelDatabase.get_reasoning(chosen_model) if chosen_model else None
        self._reasoning = self._reasoning_mode == "openai"
        if self._reasoning_mode:
            self.logger.info(
                f"Using Responses model '{chosen_model}' (mode='{self._reasoning_mode}') with "
                f"'{self._reasoning_effort}' reasoning effort"
            )

    def _initialize_default_params(self, kwargs: dict[str, Any]) -> RequestParams:
        base_params = super()._initialize_default_params(kwargs)
        chosen_model = kwargs.get("model", DEFAULT_RESPONSES_MODEL)
        base_params.model = chosen_model
        return base_params

    def _openai_settings(self):
        if self.context and self.context.config:
            return getattr(self.context.config, "openai", None)
        return None

    def _base_url(self) -> str | None:
        settings = self._openai_settings()
        return settings.base_url if settings else None

    def _default_headers(self) -> dict[str, str] | None:
        settings = self._openai_settings()
        return settings.default_headers if settings else None

    def _responses_client(self) -> AsyncOpenAI:
        try:
            kwargs: dict[str, Any] = {
                "api_key": self._api_key(),
                "base_url": self._base_url(),
                "http_client": DefaultAioHttpClient(),
            }
            default_headers = self._default_headers()
            if default_headers:
                kwargs["default_headers"] = default_headers
            return AsyncOpenAI(**kwargs)
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e

    def _adjust_schema(self, input_schema: dict[str, Any]) -> dict[str, Any]:
        if "properties" in input_schema:
            return input_schema
        result = input_schema.copy()
        result["properties"] = {}
        return result

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for msg in messages:
            items.extend(self._convert_message_to_items(msg))
        return items

    def _convert_message_to_items(self, msg: PromptMessageExtended) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        items.extend(self._extract_encrypted_reasoning_items(msg.channels))

        if msg.tool_results:
            items.extend(self._convert_tool_results(msg.tool_results))
            message_item = self._build_message_item(msg.role, msg.content)
            if message_item:
                items.append(message_item)
            return items

        message_item = self._build_message_item(msg.role, msg.content)
        if message_item:
            items.append(message_item)

        if msg.tool_calls:
            items.extend(self._convert_tool_calls(msg.tool_calls))

        return items

    def _extract_encrypted_reasoning_items(
        self, channels: Mapping[str, Iterable[ContentBlock]] | None
    ) -> list[dict[str, Any]]:
        if not channels:
            return []
        encrypted_blocks = channels.get(OPENAI_REASONING_ENCRYPTED)
        if not encrypted_blocks:
            return []

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
                items.append(data)
        return items

    def _build_message_item(
        self, role: str, content: list[ContentBlock]
    ) -> dict[str, Any] | None:
        if not content:
            return None
        parts = self._convert_content_parts(content, role)
        if not parts:
            return None
        return {
            "type": "message",
            "role": role,
            "content": parts,
        }

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

            if is_image_content(item) or is_resource_content(item):
                image_url = self._content_to_image_url(item)
                if image_url:
                    parts.append({"type": "input_image", "image_url": image_url})
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
            return None
        mime_type = None
        if hasattr(item, "mimeType"):
            mime_type = getattr(item, "mimeType", None)
        if not mime_type and is_resource_content(item):
            resource = getattr(item, "resource", None)
            mime_type = getattr(resource, "mimeType", None) if resource else None
        if not mime_type:
            mime_type = "image/png"
        return f"data:{mime_type};base64,{data}"

    def _convert_tool_calls(self, tool_calls: dict[str, CallToolRequest]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, (tool_use_id, request) in enumerate(tool_calls.items()):
            params = getattr(request, "params", None)
            name = getattr(params, "name", None) or "tool"
            arguments = getattr(params, "arguments", None) or {}
            call_id = tool_use_id or f"call-{index}"
            items.append(
                {
                    "type": "function_call",
                    "id": call_id,
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
            call_id = tool_use_id or f"call-{index}"
            output = self._tool_result_to_text(result)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
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
            if is_image_content(item) or is_resource_content(item):
                image_url = self._content_to_image_url(item)
                if image_url:
                    chunks.append(f"![Image]({image_url})")
                    continue
            resource_uri = get_resource_uri(item)
            if resource_uri:
                chunks.append(f"[Resource]({resource_uri})")
                continue
            chunks.append(f"[Unsupported content: {type(item).__name__}]")
        return "\n".join(chunk for chunk in chunks if chunk)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        req_params = self.get_request_params(request_params)

        last_message = multipart_messages[-1]
        if last_message.role == "assistant":
            return last_message

        input_items = self._convert_to_provider_format(multipart_messages)
        if not input_items:
            input_items = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": ""}],
                }
            ]

        return await self._responses_completion(input_items, req_params, tools)

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        model = self.default_request_params.model or DEFAULT_RESPONSES_MODEL
        base_args: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": request_params.parallel_tool_calls,
        }

        system_prompt = self.instruction or request_params.systemPrompt
        if system_prompt:
            base_args["instructions"] = system_prompt

        if tools:
            base_args["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": self._adjust_schema(tool.inputSchema),
                }
                for tool in tools
            ]

        if self._reasoning:
            base_args["reasoning"] = {
                "summary": "auto",
                "effort": self._reasoning_effort or DEFAULT_REASONING_EFFORT,
            }

        if request_params.maxTokens is not None:
            base_args["max_output_tokens"] = request_params.maxTokens

        if request_params.response_format:
            base_args["text"] = {"format": request_params.response_format}

        return self.prepare_provider_arguments(
            base_args, request_params, self.RESPONSES_EXCLUDE_FIELDS
        )

    async def _responses_completion(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        response_content_blocks: list[ContentBlock] = []
        model_name = self.default_request_params.model or DEFAULT_RESPONSES_MODEL

        arguments = self._build_response_args(input_items, request_params, tools)
        self.logger.debug("Responses request", data=arguments)

        self._log_chat_progress(self.chat_turn(), model=model_name)

        try:
            async with self._responses_client() as client:
                async with client.responses.stream(**arguments) as stream:
                    response, streamed_summary = await self._process_stream(
                        stream, model_name
                    )
        except AuthenticationError as e:
            raise ProviderKeyError(
                "Invalid OpenAI API key",
                "The configured OpenAI API key was rejected.\n"
                "Please check that your API key is valid and not expired.",
            ) from e
        except APIError as error:
            self.logger.error("Streaming APIError during Responses completion", exc_info=error)
            raise
        except asyncio.CancelledError:
            return Prompt.assistant(
                TextContent(type="text", text=""),
                stop_reason=LlmStopReason.CANCELLED,
            )

        if response is None:
            raise RuntimeError("Responses stream did not return a final response")

        self._log_chat_finished(model=model_name)

        channels: dict[str, list[ContentBlock]] | None = None
        reasoning_blocks = self._extract_reasoning_summary(response, streamed_summary)
        encrypted_blocks = self._extract_encrypted_reasoning(response)
        if reasoning_blocks or encrypted_blocks:
            channels = {}
            if reasoning_blocks:
                channels[REASONING] = reasoning_blocks
            if encrypted_blocks:
                channels[OPENAI_REASONING_ENCRYPTED] = encrypted_blocks

        tool_calls = self._extract_tool_calls(response)
        if tool_calls:
            stop_reason = LlmStopReason.TOOL_USE
        else:
            stop_reason = self._map_response_stop_reason(response)

        for output_item in getattr(response, "output", []) or []:
            if getattr(output_item, "type", None) != "message":
                continue
            for part in getattr(output_item, "content", []) or []:
                if getattr(part, "type", None) == "output_text":
                    response_content_blocks.append(
                        TextContent(type="text", text=getattr(part, "text", ""))
                    )

        if not response_content_blocks:
            output_text = getattr(response, "output_text", None)
            if output_text:
                response_content_blocks.append(TextContent(type="text", text=output_text))

        if getattr(response, "usage", None):
            self._record_usage(response.usage, model_name)

        self.history.set(input_items)

        return PromptMessageExtended(
            role="assistant",
            content=response_content_blocks,
            tool_calls=tool_calls,
            channels=channels,
            stop_reason=stop_reason,
        )

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
            call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            name = getattr(item, "name", None) or "tool"
            arguments_raw = getattr(item, "arguments", None)
            if arguments_raw:
                try:
                    arguments = from_json(arguments_raw, allow_partial=True)
                except Exception:
                    arguments = {}
            else:
                arguments = {}
            if not call_id:
                call_id = f"call-{len(tool_calls)}"
            tool_calls[call_id] = CallToolRequest(
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

    async def _process_stream(
        self, stream: Any, model: str
    ) -> tuple[Any, list[str]]:
        estimated_tokens = 0
        reasoning_chars = 0
        reasoning_segments: list[str] = []
        tool_streams: dict[int, dict[str, Any]] = {}

        async for event in stream:
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
                continue

        final_response = await stream.get_final_response()
        return final_response, reasoning_segments

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

        # Emit progress event
        data = {
            "progress_action": type,
            "model": model,
            "agent_name": self.name,
            "chat_turn": self.chat_turn(),
            "details": token_str.strip(),  # Token count goes in details for STREAMING action
        }
        self.logger.info("Streaming progress", data=data)
