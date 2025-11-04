import base64
from typing import Any, Dict, List, Tuple

# Import necessary types from google.genai
from google.genai import types
from mcp import Tool
from mcp.types import (
    BlobResourceContents,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    TextContent,
)

from fast_agent.mcp.helpers.content_helpers import (
    get_image_data,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
)
from fast_agent.types import PromptMessageExtended, RequestParams


class GoogleConverter:
    """
    Converts between fast-agent and google.genai data structures.
    """

    def _clean_schema_for_google(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively removes unsupported JSON schema keywords for google.genai.types.Schema.
        Specifically removes 'additionalProperties', '$schema', 'exclusiveMaximum', and 'exclusiveMinimum'.
        Also resolves $ref references and inlines $defs.
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

    def _resolve_refs(self, schema: Dict[str, Any], root_schema: Dict[str, Any]) -> Dict[str, Any]:
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

        # If this is a $ref, resolve it
        if "$ref" in schema:
            ref_path = schema["$ref"]
            if ref_path.startswith("#/"):
                # Parse the reference path (e.g., "#/$defs/HumanInputRequest")
                path_parts = ref_path[2:].split("/")  # Remove "#/" and split

                # Navigate to the referenced definition
                ref_target = root_schema
                for part in path_parts:
                    if part in ref_target:
                        ref_target = ref_target[part]
                    else:
                        # If reference not found, return the original schema
                        return schema

                # Return the resolved definition (recursively resolve any nested refs)
                return self._resolve_refs(ref_target, root_schema)

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
        self, messages: List[PromptMessageExtended]
    ) -> List[types.Content]:
        """
        Converts a list of fast-agent PromptMessageExtended to google.genai types.Content.
        Handles different roles and content types (text, images, etc.).
        """
        google_contents: List[types.Content] = []
        for message in messages:
            parts: List[types.Part] = []
            for part_content in message.content:  # renamed part to part_content to avoid conflict
                if is_text_content(part_content):
                    parts.append(types.Part.from_text(text=get_text(part_content) or ""))
                elif is_image_content(part_content):
                    assert isinstance(part_content, ImageContent)
                    image_bytes = base64.b64decode(get_image_data(part_content) or "")
                    parts.append(
                        types.Part.from_bytes(mime_type=part_content.mimeType, data=image_bytes)
                    )
                elif is_resource_content(part_content):
                    assert isinstance(part_content, EmbeddedResource)
                    if "application/pdf" == part_content.resource.mimeType and isinstance(
                        part_content.resource, BlobResourceContents
                    ):
                        pdf_bytes = base64.b64decode(part_content.resource.blob)
                        parts.append(
                            types.Part.from_bytes(
                                mime_type=part_content.resource.mimeType or "application/pdf",
                                data=pdf_bytes,
                            )
                        )
                    else:
                        # Check if the resource itself has text content
                        # Use get_text helper to extract text from various content types
                        resource_text = get_text(part_content.resource)

                        if resource_text is not None:
                            parts.append(types.Part.from_text(text=resource_text))
                        else:
                            # Fallback for other binary types or types without direct text
                            uri_str = getattr(part_content.resource, "uri", "unknown_uri")
                            mime_str = getattr(part_content.resource, "mimeType", "unknown_mime")
                            parts.append(
                                types.Part.from_text(
                                    text=f"[Resource: {uri_str}, MIME: {mime_str}]"
                                )
                            )

            if parts:
                google_role = (
                    "user"
                    if message.role == "user"
                    else ("model" if message.role == "assistant" else "tool")
                )
                google_contents.append(types.Content(role=google_role, parts=parts))
        return google_contents

    def convert_to_google_tools(self, tools: List[Tool]) -> List[types.Tool]:
        """
        Converts a list of fast-agent ToolDefinition to google.genai types.Tool.
        """
        google_tools: List[types.Tool] = []
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
        self, content: types.Content
    ) -> List[ContentBlock | CallToolRequestParams]:
        """
        Converts google.genai types.Content from a model response to a list of
        fast-agent content types or tool call requests.
        """
        fast_agent_parts: List[ContentBlock | CallToolRequestParams] = []

        if content is None or not hasattr(content, "parts") or content.parts is None:
            return []  # Google API response 'content' object is None. Cannot extract parts.

        for part in content.parts:
            if part.text:
                fast_agent_parts.append(TextContent(type="text", text=part.text))
            elif part.function_call:
                fast_agent_parts.append(
                    CallToolRequestParams(
                        name=part.function_call.name,
                        arguments=part.function_call.args,
                    )
                )
        return fast_agent_parts

    def convert_from_google_function_call(
        self, function_call: types.FunctionCall
    ) -> CallToolRequest:
        """
        Converts a single google.genai types.FunctionCall to a fast-agent CallToolRequest.
        """
        return CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name=function_call.name,
                arguments=function_call.args,
            ),
        )

    def convert_function_results_to_google(
        self, tool_results: List[Tuple[str, CallToolResult]]
    ) -> List[types.Content]:
        """
        Converts a list of fast-agent tool results to google.genai types.Content
        with role 'tool'. Handles multimodal content in tool results.
        """
        google_tool_response_contents: List[types.Content] = []
        for tool_name, tool_result in tool_results:
            current_content_parts: List[types.Part] = []
            textual_outputs: List[str] = []
            media_parts: List[types.Part] = []

            for item in tool_result.content:
                if is_text_content(item):
                    textual_outputs.append(get_text(item) or "")  # Ensure no None is added
                elif is_image_content(item):
                    assert isinstance(item, ImageContent)
                    try:
                        image_bytes = base64.b64decode(get_image_data(item) or "")
                        media_parts.append(
                            types.Part.from_bytes(data=image_bytes, mime_type=item.mimeType)
                        )
                    except Exception as e:
                        textual_outputs.append(f"[Error processing image from tool result: {e}]")
                elif is_resource_content(item):
                    assert isinstance(item, EmbeddedResource)
                    if (
                        "application/pdf" == item.resource.mimeType
                        and hasattr(item.resource, "blob")
                        and isinstance(item.resource, BlobResourceContents)
                    ):
                        try:
                            pdf_bytes = base64.b64decode(item.resource.blob)
                            media_parts.append(
                                types.Part.from_bytes(
                                    data=pdf_bytes,
                                    mime_type=item.resource.mimeType or "application/pdf",
                                )
                            )
                        except Exception as e:
                            textual_outputs.append(f"[Error processing PDF from tool result: {e}]")
                    else:
                        # Check if the resource itself has text content
                        # Use get_text helper to extract text from various content types
                        resource_text = get_text(item.resource)

                        if resource_text is not None:
                            textual_outputs.append(resource_text)
                        else:
                            uri_str = getattr(item.resource, "uri", "unknown_uri")
                            mime_str = getattr(item.resource, "mimeType", "unknown_mime")
                            textual_outputs.append(
                                f"[Unhandled Resource in Tool: {uri_str}, MIME: {mime_str}]"
                            )
                # Add handling for other content types if needed, for now they are skipped or become unhandled resource text

            function_response_payload: Dict[str, Any] = {"tool_name": tool_name}
            if textual_outputs:
                function_response_payload["text_content"] = "\n".join(textual_outputs)

            # Only add media_parts if there are some, otherwise Gemini might error on empty parts for function response
            if media_parts:
                # Create the main FunctionResponse part
                fn_response_part = types.Part.from_function_response(
                    name=tool_name, response=function_response_payload
                )
                current_content_parts.append(fn_response_part)
                current_content_parts.extend(
                    media_parts
                )  # Add media parts after the main response part
            else:  # If no media parts, the textual output (if any) is the sole content of the function response
                fn_response_part = types.Part.from_function_response(
                    name=tool_name, response=function_response_payload
                )
                current_content_parts.append(fn_response_part)

            google_tool_response_contents.append(
                types.Content(role="tool", parts=current_content_parts)
            )
        return google_tool_response_contents

    def convert_request_params_to_google_config(
        self, request_params: RequestParams
    ) -> types.GenerateContentConfig:
        """
        Converts fast-agent RequestParams to google.genai types.GenerateContentConfig.
        """
        config_args: Dict[str, Any] = {}
        if request_params.temperature is not None:
            config_args["temperature"] = request_params.temperature
        if request_params.maxTokens is not None:
            config_args["max_output_tokens"] = request_params.maxTokens
        if hasattr(request_params, "topK") and request_params.topK is not None:
            config_args["top_k"] = request_params.topK
        if hasattr(request_params, "topP") and request_params.topP is not None:
            config_args["top_p"] = request_params.topP
        if hasattr(request_params, "stopSequences") and request_params.stopSequences is not None:
            config_args["stop_sequences"] = request_params.stopSequences
        if (
            hasattr(request_params, "presencePenalty")
            and request_params.presencePenalty is not None
        ):
            config_args["presence_penalty"] = request_params.presencePenalty
        if (
            hasattr(request_params, "frequencyPenalty")
            and request_params.frequencyPenalty is not None
        ):
            config_args["frequency_penalty"] = request_params.frequencyPenalty
        if request_params.systemPrompt is not None:
            config_args["system_instruction"] = request_params.systemPrompt
        return types.GenerateContentConfig(**config_args)

    def convert_from_google_content_list(
        self, contents: List[types.Content]
    ) -> List[PromptMessageExtended]:
        """
        Converts a list of google.genai types.Content to a list of fast-agent PromptMessageExtended.
        """
        return [self._convert_from_google_content(content) for content in contents]

    def _convert_from_google_content(self, content: types.Content) -> PromptMessageExtended:
        """
        Converts a single google.genai types.Content to a fast-agent PromptMessageExtended.
        """
        # Official fix for GitHub issue #207: Handle None content or content.parts
        if content is None or not hasattr(content, "parts") or content.parts is None:
            return PromptMessageExtended(role="assistant", content=[])

        if content.role == "model" and any(part.function_call for part in content.parts):
            return PromptMessageExtended(role="assistant", content=[])

        fast_agent_parts: List[ContentBlock | CallToolRequestParams] = []
        for part in content.parts:
            if part.text:
                fast_agent_parts.append(TextContent(type="text", text=part.text))
            elif part.function_response:
                response_text = str(part.function_response.response)
                fast_agent_parts.append(TextContent(type="text", text=response_text))
            elif part.file_data:
                fast_agent_parts.append(
                    EmbeddedResource(
                        type="resource",
                        resource=TextContent(
                            uri=part.file_data.file_uri,
                            mimeType=part.file_data.mime_type,
                            text=f"[Resource: {part.file_data.file_uri}, MIME: {part.file_data.mime_type}]",
                        ),
                    )
                )

        fast_agent_role = "user" if content.role == "user" else "assistant"
        return PromptMessageExtended(role=fast_agent_role, content=fast_agent_parts)
