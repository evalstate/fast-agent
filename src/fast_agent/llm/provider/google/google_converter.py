import base64
from typing import Any, TypeAlias, cast

# Import necessary types from google.genai
from google.genai import types
from mcp import Tool
from mcp.types import (
    BlobResourceContents,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from fast_agent.llm.structured_schema import resolve_local_ref
from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    get_image_data,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
)
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.utils.text import strip_casefold

GoogleToolResult: TypeAlias = tuple[str, str | None, CallToolResult]


class GoogleConverter:
    """
    Converts between fast-agent and google.genai data structures.
    """

    @staticmethod
    def _is_thought_part(part: types.Part) -> bool:
        """Return True when Gemini marks a part as internal reasoning."""
        return bool(getattr(part, "thought", False))

    def _clean_schema_for_google(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively removes unsupported JSON schema keywords for google.genai.types.Schema.
        Specifically removes 'additionalProperties', '$schema', 'exclusiveMaximum', and 'exclusiveMinimum'.
        Also resolves $ref references and inlines $defs.

        Pydantic structured outputs are passed to google.genai as model classes
        so the SDK can normalize them. This helper is only for arbitrary raw
        schema dicts, where fast-agent strips known-unsupported keywords before
        handing the schema to the SDK.
        """
        # First, resolve any $ref references in the schema
        schema = self._resolve_refs(schema, schema)

        cleaned_schema = {}
        unsupported_keys = {
            "additionalProperties",
            "$schema",
            "exclusiveMaximum",
            "exclusiveMinimum",
            "$defs",  # Remove $defs after resolving references
        }
        supported_string_formats = {"enum", "date-time"}

        for key, value in schema.items():
            if key in unsupported_keys:
                continue  # Skip this key

            # Rewrite unsupported 'const' to a safe form for Gemini tools
            # - For string const, convert to enum [value]
            # - For non-string const (booleans/numbers), drop the constraint
            if key == "const":
                if isinstance(value, str):
                    cleaned_schema["enum"] = [value]
                continue

            if (
                key == "format"
                and schema.get("type") == "string"
                and value not in supported_string_formats
            ):
                continue  # Remove unsupported string formats

            if isinstance(value, dict):
                cleaned_schema[key] = self._clean_schema_for_google(value)
            elif isinstance(value, list):
                cleaned_schema[key] = [
                    self._clean_schema_for_google(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned_schema[key] = value
        return cleaned_schema

    def _resolve_refs(self, schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve $ref references in a JSON schema by inlining the referenced definitions.

        Args:
            schema: The current schema fragment being processed
            root_schema: The root schema containing $defs

        Returns:
            Schema with $ref references resolved
        """
        if not isinstance(schema, dict):
            return schema

        if "$ref" in schema:
            ref_path = schema["$ref"]
            if isinstance(ref_path, str):
                ref_target = resolve_local_ref(root_schema, ref_path)
                if isinstance(ref_target, dict):
                    schema = {**ref_target, **schema}
                    schema.pop("$ref", None)

        # Otherwise, recursively process all values in the schema
        resolved = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_refs(value, root_schema)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_refs(item, root_schema) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved[key] = value

        return resolved

    def convert_to_google_content(
        self, messages: list[PromptMessageExtended]
    ) -> list[types.Content]:
        """
        Converts a list of fast-agent PromptMessageExtended to google.genai types.Content.
        Handles different roles and content types (text, images, etc.).
        """
        google_contents: list[types.Content] = []
        for message in messages:
            parts = [
                google_part
                for part_content in message.content
                if (google_part := self._content_block_to_google_part(part_content)) is not None
            ]

            if parts:
                google_role = (
                    "user"
                    if message.role == "user"
                    else ("model" if message.role == "assistant" else "tool")
                )
                google_contents.append(types.Content(role=google_role, parts=parts))
        return google_contents

    def _content_block_to_google_part(self, content: ContentBlock) -> types.Part | None:
        if is_text_content(content):
            return types.Part.from_text(text=get_text(content) or "")
        if is_image_content(content):
            assert isinstance(content, ImageContent)
            return self._image_content_to_google_part(content)
        if is_resource_content(content):
            assert isinstance(content, EmbeddedResource)
            return self._embedded_resource_to_google_part(content)
        if is_resource_link(content):
            assert isinstance(content, ResourceLink)
            return self._resource_link_to_google_part(content)
        return None

    @staticmethod
    def _image_content_to_google_part(content: ImageContent) -> types.Part:
        image_bytes = base64.b64decode(get_image_data(content) or "")
        return types.Part.from_bytes(mime_type=content.mimeType, data=image_bytes)

    def _embedded_resource_to_google_part(self, content: EmbeddedResource) -> types.Part:
        resource = content.resource
        mime_type = getattr(resource, "mimeType", None)
        if mime_type == "application/pdf" and isinstance(resource, BlobResourceContents):
            pdf_bytes = base64.b64decode(resource.blob)
            return types.Part.from_bytes(
                mime_type=resource.mimeType or "application/pdf",
                data=pdf_bytes,
            )

        if mime_type and mime_type.startswith(("video/", "audio/")):
            return self._media_resource_to_google_part(resource, mime_type=mime_type)

        if isinstance(resource, TextResourceContents):
            return types.Part.from_text(text=resource.text)

        uri_str = getattr(resource, "uri", "unknown_uri")
        mime_str = getattr(resource, "mimeType", "unknown_mime")
        return types.Part.from_text(text=f"[Resource: {uri_str}, MIME: {mime_str}]")

    @staticmethod
    def _media_resource_to_google_part(resource: Any, *, mime_type: str) -> types.Part:
        if isinstance(resource, BlobResourceContents):
            media_bytes = base64.b64decode(resource.blob)
            return types.Part.from_bytes(mime_type=mime_type, data=media_bytes)

        uri_str = getattr(resource, "uri", None)
        mime_str = getattr(resource, "mimeType", "application/octet-stream")
        if uri_str:
            return types.Part.from_uri(file_uri=str(uri_str), mime_type=mime_str)

        return types.Part.from_text(text=f"[Video Resource: No URI provided, MIME: {mime_str}]")

    @staticmethod
    def _resource_link_to_google_part(content: ResourceLink) -> types.Part | None:
        mime = content.mimeType
        uri_str = str(content.uri) if content.uri else None
        if uri_str and mime and mime.startswith(("video/", "audio/", "image/")):
            return types.Part.from_uri(file_uri=uri_str, mime_type=mime)

        text = get_text(content)
        if text:
            return types.Part.from_text(text=text)
        return None

    def convert_to_google_tools(self, tools: list[Tool]) -> list[types.Tool]:
        """
        Converts a list of fast-agent ToolDefinition to google.genai types.Tool.
        """
        google_tools: list[types.Tool] = []
        for tool in tools:
            cleaned_input_schema = self._clean_schema_for_google(tool.inputSchema)
            function_declaration = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description if tool.description else "",
                parameters=types.Schema(**cleaned_input_schema),
            )
            google_tools.append(types.Tool(function_declarations=[function_declaration]))
        return google_tools

    def convert_from_google_content(
        self, content: types.Content | None
    ) -> list[ContentBlock | CallToolRequestParams]:
        """
        Converts google.genai types.Content from a model response to a list of
        fast-agent content types or tool call requests.
        """
        fast_agent_parts: list[ContentBlock | CallToolRequestParams] = []

        if content is None:
            return []  # Google API response 'content' object is None. Cannot extract parts.

        parts = content.parts
        if parts is None:
            return []

        for part in parts:
            if self._is_thought_part(part):
                continue
            if part.text:
                fast_agent_parts.append(TextContent(type="text", text=part.text))
            elif part.function_call:
                fast_agent_parts.append(
                    CallToolRequestParams(
                        name=part.function_call.name or "unknown_function",
                        arguments=part.function_call.args,
                    )
                )
        return fast_agent_parts

    def convert_function_results_to_google(
        self, tool_results: list[GoogleToolResult]
    ) -> list[types.Content]:
        """
        Converts a list of fast-agent tool results to google.genai types.Content
        with role 'user'. Handles multimodal content in tool results.
        Returns a single types.Content with all function response parts.
        """
        if not tool_results:
            return []

        parts: list[types.Part] = []
        for tool_name, tool_call_id, tool_result in tool_results:
            textual_outputs, media_parts = self._google_tool_result_parts(tool_result)

            output_text = "\n".join(textual_outputs)
            function_response_payload: dict[str, Any] = (
                {"error": output_text or "Tool execution failed."}
                if tool_result.isError
                else {"result": output_text}
            )
            fn_response_part = types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    id=tool_call_id,
                    response=function_response_payload,
                    parts=media_parts or None,
                )
            )
            parts.append(fn_response_part)

        return [types.Content(role="user", parts=parts)]

    def _google_tool_result_parts(
        self, tool_result: CallToolResult
    ) -> tuple[list[str], list[types.FunctionResponsePart]]:
        textual_outputs: list[str] = []
        media_parts: list[types.FunctionResponsePart] = []
        canonical_content = canonicalize_tool_result_content_for_llm(
            tool_result,
            source="google",
        )

        for item in canonical_content:
            text, media_part = self._google_tool_result_item_part(item)
            if text is not None:
                textual_outputs.append(text)
            if media_part is not None:
                media_parts.append(media_part)
        return textual_outputs, media_parts

    def _google_tool_result_item_part(
        self, item: ContentBlock
    ) -> tuple[str | None, types.FunctionResponsePart | None]:
        if is_text_content(item):
            return get_text(item) or "", None
        if is_image_content(item):
            assert isinstance(item, ImageContent)
            return self._image_tool_result_part(item)
        if is_resource_content(item):
            assert isinstance(item, EmbeddedResource)
            return self._resource_tool_result_part(item)
        if is_resource_link(item):
            assert isinstance(item, ResourceLink)
            return self._resource_link_tool_result_part(item)
        return None, None

    def _image_tool_result_part(
        self, item: ImageContent
    ) -> tuple[str | None, types.FunctionResponsePart | None]:
        try:
            image_bytes = base64.b64decode(get_image_data(item) or "")
            return None, self._function_response_inline_part(
                data=image_bytes,
                mime_type=item.mimeType,
            )
        except Exception as e:
            return f"[Error processing image from tool result: {e}]", None

    def _resource_tool_result_part(
        self, item: EmbeddedResource
    ) -> tuple[str | None, types.FunctionResponsePart | None]:
        resource = item.resource
        mime_type = getattr(resource, "mimeType", None)
        if mime_type == "application/pdf" and isinstance(resource, BlobResourceContents):
            return self._blob_tool_result_part(
                resource,
                mime_type=mime_type or "application/pdf",
                label="PDF",
            )
        if (
            mime_type
            and mime_type.startswith(("video/", "audio/"))
            and isinstance(resource, BlobResourceContents)
        ):
            return self._blob_tool_result_part(resource, mime_type=mime_type, label="media")
        if isinstance(resource, TextResourceContents):
            return resource.text, None

        uri_str = getattr(resource, "uri", "unknown_uri")
        mime_str = getattr(resource, "mimeType", "unknown_mime")
        return f"[Unhandled Resource in Tool: {uri_str}, MIME: {mime_str}]", None

    def _blob_tool_result_part(
        self,
        resource: BlobResourceContents,
        *,
        mime_type: str,
        label: str,
    ) -> tuple[str | None, types.FunctionResponsePart | None]:
        try:
            data = base64.b64decode(resource.blob)
            return None, self._function_response_inline_part(
                data=data,
                mime_type=mime_type,
            )
        except Exception as e:
            return f"[Error processing {label} from tool result: {e}]", None

    def _resource_link_tool_result_part(
        self, item: ResourceLink
    ) -> tuple[str | None, types.FunctionResponsePart | None]:
        mime = item.mimeType
        uri_str = str(item.uri) if item.uri else None
        if uri_str and mime and mime.startswith(("video/", "audio/", "image/")):
            return None, self._function_response_file_part(file_uri=uri_str, mime_type=mime)

        text = get_text(item)
        if text:
            return text, None
        return None, None

    @staticmethod
    def _function_response_inline_part(
        *,
        data: bytes,
        mime_type: str | None,
    ) -> types.FunctionResponsePart:
        return types.FunctionResponsePart(
            inline_data=types.FunctionResponseBlob(
                data=data,
                mime_type=mime_type,
            )
        )

    @staticmethod
    def _function_response_file_part(
        *,
        file_uri: str,
        mime_type: str,
    ) -> types.FunctionResponsePart:
        return types.FunctionResponsePart(
            file_data=types.FunctionResponseFileData(
                file_uri=file_uri,
                mime_type=mime_type,
            )
        )

    def convert_request_params_to_google_config(
        self,
        request_params: RequestParams,
        *,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
    ) -> types.GenerateContentConfig:
        """
        Converts fast-agent RequestParams to google.genai types.GenerateContentConfig.

        Args:
            request_params: The request params to convert.
            thinking_budget: Optional thinking budget in tokens.
                0 disables thinking, -1 enables automatic budget, positive
                values set an explicit token budget.
            thinking_level: Optional SDK ThinkingLevel name
                (MINIMAL/LOW/MEDIUM/HIGH). When set, takes precedence over
                thinking_budget for named effort levels.
        """
        config_args = self._google_generation_config_args(
            request_params,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        )
        return types.GenerateContentConfig(**config_args)

    def _google_generation_config_args(
        self,
        request_params: RequestParams,
        *,
        thinking_budget: int | None,
        thinking_level: str | None,
    ) -> dict[str, Any]:
        config_args: dict[str, Any] = {}
        model = strip_casefold(request_params.model or "")
        is_gemini_3 = "gemini-3" in model or "gemini-3.5" in model
        config_args.update(self._google_sampling_config_args(request_params, is_gemini_3))
        config_args.update(self._google_base_config_args(request_params))

        thinking_config = self._google_thinking_config(
            is_gemini_3=is_gemini_3,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        )
        if thinking_config is not None:
            config_args["thinking_config"] = thinking_config
        return config_args

    @staticmethod
    def _request_param_value(request_params: RequestParams, *names: str) -> Any:
        for name in names:
            value = getattr(request_params, name, None)
            if value is not None:
                return value
        return None

    def _google_sampling_config_args(
        self, request_params: RequestParams, is_gemini_3: bool
    ) -> dict[str, Any]:
        if is_gemini_3:
            return {}

        config_args: dict[str, Any] = {}
        if request_params.temperature is not None:
            config_args["temperature"] = request_params.temperature

        top_k = self._request_param_value(request_params, "top_k", "topK")
        if top_k is not None:
            config_args["top_k"] = top_k

        top_p = self._request_param_value(request_params, "top_p", "topP")
        if top_p is not None:
            config_args["top_p"] = top_p
        return config_args

    def _google_base_config_args(self, request_params: RequestParams) -> dict[str, Any]:
        config_args: dict[str, Any] = {}
        if request_params.maxTokens is not None:
            config_args["max_output_tokens"] = request_params.maxTokens

        stop_sequences = getattr(request_params, "stopSequences", None)
        if stop_sequences is not None:
            config_args["stop_sequences"] = stop_sequences

        presence_penalty = self._request_param_value(
            request_params, "presence_penalty", "presencePenalty"
        )
        if presence_penalty is not None:
            config_args["presence_penalty"] = presence_penalty

        frequency_penalty = self._request_param_value(
            request_params, "frequency_penalty", "frequencyPenalty"
        )
        if frequency_penalty is not None:
            config_args["frequency_penalty"] = frequency_penalty

        if request_params.systemPrompt is not None:
            config_args["system_instruction"] = request_params.systemPrompt
        return config_args

    @staticmethod
    def _google_thinking_config(
        *,
        is_gemini_3: bool,
        thinking_budget: int | None,
        thinking_level: str | None,
    ) -> types.ThinkingConfig | None:
        if thinking_level is None and thinking_budget is None:
            return None

        sdk_thinking_level = cast("Any", thinking_level)
        if is_gemini_3:
            return GoogleConverter._google_gemini_3_thinking_config(
                thinking_budget=thinking_budget,
                thinking_level=sdk_thinking_level,
            )
        if thinking_level is not None and thinking_budget is not None:
            return types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=sdk_thinking_level,
                thinking_budget=thinking_budget,
            )
        if thinking_level is not None:
            return types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=sdk_thinking_level,
            )
        return types.ThinkingConfig(include_thoughts=True, thinking_budget=thinking_budget)

    @staticmethod
    def _google_gemini_3_thinking_config(
        *,
        thinking_budget: int | None,
        thinking_level: Any,
    ) -> types.ThinkingConfig:
        if thinking_level is not None:
            return types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=thinking_level,
            )
        if thinking_budget == 0:
            return types.ThinkingConfig(include_thoughts=True, thinking_budget=0)

        return types.ThinkingConfig(
            include_thoughts=True,
            thinking_level=cast("Any", "MEDIUM"),
        )
