import base64
from typing import Any, Dict, List, Tuple, Union

# Import necessary types from google.genai
from google.genai import types
from mcp.types import (
    BlobResourceContents,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)

from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.helpers.content_helpers import (
    get_image_data,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.tools.tool_definition import ToolDefinition


class GoogleConverter:
    """
    Converts between fast-agent and google.genai data structures.
    """

    def _clean_schema_for_google(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively removes unsupported JSON schema keywords for google.genai.types.Schema.
        Specifically removes 'additionalProperties', '$schema', 'exclusiveMaximum', and 'exclusiveMinimum'.
        """
        cleaned_schema = {}
        unsupported_keys = {
            "additionalProperties",
            "$schema",
            "exclusiveMaximum",
            "exclusiveMinimum",
            # "title" is generally supported or ignored by Google's Schema, keep it for description if needed
        }
        # Only specific string formats are directly supported or need special handling by Google.
        # Others might be removed if they cause issues. For now, keeping most common.
        # Pydantic might generate "format": "date-time" which is fine.
        # "enum" itself is not a format, it's a keyword at the same level as type, description.
        # The previous 'supported_string_formats = {"enum", "date-time"}' was a bit misleading.
        # 'format' is a keyword. 'enum' is a separate keyword.

        for key, value in schema.items():
            if key in unsupported_keys:
                continue

            # Example: if Google's schema validation is strict about unknown 'format' values for strings:
            # if key == "format" and schema.get("type") == "string" and value not in {"date-time", ... /* other supported formats */}:
            #     continue

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

    def convert_to_google_content(
        self, messages: List[PromptMessageMultipart]
    ) -> List[types.Content]:
        """
        Converts a list of fast-agent PromptMessageMultipart to google.genai types.Content.
        Handles different roles and content types (text, images, PDF resources, and other generic resources).
        """
        google_contents: List[types.Content] = []
        for message in messages:
            parts: List[types.Part] = []
            for part_content in message.content:
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
                    if (
                        "application/pdf" == part_content.resource.mimeType
                        and hasattr(part_content.resource, "blob")
                        and isinstance(part_content.resource, BlobResourceContents)
                    ):
                        pdf_bytes = base64.b64decode(part_content.resource.blob)
                        parts.append(
                            types.Part.from_bytes(
                                mime_type=part_content.resource.mimeType or "application/pdf",
                                data=pdf_bytes,
                            )
                        )
                    else:
                        resource_text = None
                        if hasattr(part_content.resource, "text"):
                            resource_text = part_content.resource.text
                        elif (
                            hasattr(part_content.resource, "type")
                            and part_content.resource.type == "text"
                            and hasattr(part_content.resource, "text")
                        ):
                            resource_text = get_text(part_content.resource)

                        if resource_text is not None:
                            parts.append(types.Part.from_text(text=resource_text))
                        else:
                            uri_str = (
                                part_content.resource.uri
                                if hasattr(part_content.resource, "uri")
                                else "unknown_uri"
                            )
                            mime_str = (
                                part_content.resource.mimeType
                                if hasattr(part_content.resource, "mimeType")
                                else "unknown_mime"
                            )
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

    def convert_to_google_tools(self, tools: List[ToolDefinition]) -> List[types.Tool]:
        """
        Converts a list of fast-agent ToolDefinition to google.genai types.Tool.
        The input schema for each tool is converted to Google's format.
        """
        google_tools: List[types.Tool] = []
        for tool in tools:
            # For tool parameters, the inputSchema itself is the root for $refs.
            google_params_schema = self.json_schema_to_google_schema(
                tool.inputSchema, root_schema=tool.inputSchema
            )

            function_declaration = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description if tool.description else "",
                parameters=google_params_schema,
            )
            google_tools.append(types.Tool(function_declarations=[function_declaration]))
        return google_tools

    def _json_type_to_google_type(self, effective_json_type: str) -> types.Type:
        """Maps an effective JSON schema type (string) to google.generativeai.types.Type."""
        if effective_json_type == "string":
            return types.Type.STRING
        elif effective_json_type == "number":
            return types.Type.NUMBER
        elif effective_json_type == "integer":
            return types.Type.INTEGER
        elif effective_json_type == "boolean":
            return types.Type.BOOLEAN
        elif effective_json_type == "array":
            return types.Type.ARRAY
        elif effective_json_type == "object":
            return types.Type.OBJECT
        else:
            # Fallback for any other unexpected string type not caught by inference.
            # This case should ideally not be reached if inference is robust.
            return (
                types.Type.STRING
            )  # Defaulting to STRING might be safer than OBJECT if type is truly unknown.
            # Or raise ValueError(f"Unsupported effective JSON type: {effective_json_type}")

    def _resolve_ref(self, ref: str, root_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves a local JSON schema reference (e.g., '#/definitions/MyModel').
        Only supports references within the same schema document (root_schema).
        """
        if not ref.startswith("#/"):
            raise ValueError(
                f"Unsupported reference format: {ref}. Only local references starting with '#/' are supported."
            )

        path_parts = ref[2:].split("/")
        current_node = root_schema
        for part in path_parts:
            if isinstance(current_node, dict) and part in current_node:
                current_node = current_node[part]
            else:
                raise ValueError(
                    f"Reference '{ref}' not found in schema. Path part '{part}' is invalid in current node {current_node}."
                )
        if not isinstance(current_node, dict):
            raise ValueError(
                f"Reference '{ref}' did not resolve to a schema object (dict). Found: {type(current_node)}"
            )
        return current_node

    def json_schema_to_google_schema(
        self, json_schema_node: Dict[str, Any], root_schema: Dict[str, Any]
    ) -> types.Schema:
        """
        Recursively converts a JSON schema node (potentially with $refs) to a google.genai.types.Schema object.
        Handles type mapping, descriptions, nullability, enums, object properties, and array items.
        $refs are resolved against the root_schema.
        Unsupported JSON schema keywords are cleaned via _clean_schema_for_google.
        """
        current_processing_node = json_schema_node
        if "$ref" in json_schema_node:
            current_processing_node = self._resolve_ref(json_schema_node["$ref"], root_schema)
            # After resolving, we continue processing this resolved node.

        # Clean the node that we are actually processing (either original or resolved by $ref)
        cleaned_node = self._clean_schema_for_google(current_processing_node)

        original_node_type = cleaned_node.get(
            "type"
        )  # Type from the (resolved and cleaned) schema node
        enum_values = cleaned_node.get("enum")
        effective_json_type_str: str

        if original_node_type is None:
            if enum_values and all(
                isinstance(e, str) for e in enum_values
            ):  # Infer type string for string enums
                effective_json_type_str = "string"
            else:
                # Default to object if type is None and not a clear string enum (e.g. for schemas like {}).
                effective_json_type_str = "object"
        elif isinstance(original_node_type, list):
            # Handles nullable types like ["string", "null"]. Pick first non-null type.
            effective_json_type_str = next((t for t in original_node_type if t != "null"), "object")
        else:  # type is a single string
            effective_json_type_str = original_node_type

        google_type_enum = self._json_type_to_google_type(effective_json_type_str)

        description = cleaned_node.get("description") or cleaned_node.get(
            "title"
        )  # Use title as fallback for description

        is_nullable = False
        if isinstance(original_node_type, list) and "null" in original_node_type:
            is_nullable = True

        final_enum_values = cleaned_node.get("enum")
        if google_type_enum == types.Type.STRING and final_enum_values:
            final_enum_values = [
                str(val) for val in final_enum_values
            ]  # Ensure string enums are strings
        elif google_type_enum != types.Type.STRING and final_enum_values:
            # Non-string enums are not directly passed as Google's Schema enum currently expects strings.
            final_enum_values = None

        properties_map: Union[Dict[str, types.Schema], None] = None
        items_schema: Union[types.Schema, None] = None

        if google_type_enum == types.Type.OBJECT and "properties" in cleaned_node:
            properties_map = {
                key: self.json_schema_to_google_schema(prop_schema, root_schema)
                for key, prop_schema in cleaned_node["properties"].items()
            }
            # If properties_map becomes an empty dict (e.g. "properties": {}), it's passed as such.
            # types.Schema allows this.

        if google_type_enum == types.Type.ARRAY and "items" in cleaned_node:
            items_def = cleaned_node.get("items")
            if isinstance(items_def, dict):  # "items" must be a schema object
                items_schema = self.json_schema_to_google_schema(items_def, root_schema)
            # If "items" is not a dict, items_schema remains None.

        required_fields = cleaned_node.get("required")

        return types.Schema(
            type=google_type_enum,
            description=description,
            nullable=is_nullable,
            enum=final_enum_values,
            properties=properties_map,  # Pass properties_map (can be dict or None)
            items=items_schema,
            required=required_fields,
        )

    def convert_from_google_content(
        self, content: types.Content
    ) -> List[TextContent | ImageContent | EmbeddedResource | CallToolRequestParams]:
        """
        Converts google.genai types.Content from a model response to a list of
        fast-agent content types (TextContent, ImageContent, EmbeddedResource)
        or CallToolRequestParams if a function call is present.
        """
        fast_agent_parts: List[
            TextContent | ImageContent | EmbeddedResource | CallToolRequestParams
        ] = []
        if content.parts:
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
        with role 'tool'. Handles multimodal content (text, images, PDFs) in tool results,
        packaging them appropriately for Google's API.
        """
        google_tool_response_contents: List[types.Content] = []
        for tool_name, tool_result in tool_results:
            current_content_parts: List[types.Part] = []
            textual_outputs: List[str] = []
            media_parts: List[types.Part] = []  # For images, PDFs etc.

            for item in tool_result.content:
                if is_text_content(item):
                    textual_outputs.append(get_text(item) or "")
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
                    if (  # Handle PDF resources specifically
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
                    else:  # Handle other generic resources or resources with text
                        resource_text = None
                        if hasattr(item.resource, "text"):
                            resource_text = item.resource.text
                        elif (
                            hasattr(item.resource, "type")
                            and item.resource.type == "text"
                            and hasattr(item.resource, "text")
                        ):
                            resource_text = get_text(item.resource)

                        if resource_text is not None:
                            textual_outputs.append(resource_text)
                        else:  # Fallback for unhandled resource types
                            uri_str = (
                                item.resource.uri
                                if hasattr(item.resource, "uri")
                                else "unknown_uri"
                            )
                            mime_str = (
                                item.resource.mimeType
                                if hasattr(item.resource, "mimeType")
                                else "unknown_mime"
                            )
                            textual_outputs.append(
                                f"[Unhandled Resource in Tool: {uri_str}, MIME: {mime_str}]"
                            )

            function_response_payload: Dict[str, Any] = {"tool_name": tool_name}
            if textual_outputs:
                function_response_payload["text_content"] = "\n".join(textual_outputs)

            # The main FunctionResponse part must contain the textual outputs.
            # Media parts are added separately to the content parts list for the tool response.
            fn_response_part = types.Part.from_function_response(
                name=tool_name, response=function_response_payload
            )
            current_content_parts.append(fn_response_part)
            if media_parts:
                current_content_parts.extend(media_parts)

            google_tool_response_contents.append(
                types.Content(role="tool", parts=current_content_parts)
            )
        return google_tool_response_contents

    def convert_request_params_to_google_config(
        self, request_params: RequestParams
    ) -> types.GenerateContentConfig:
        """
        Converts fast-agent RequestParams to google.genai types.GenerateContentConfig.
        Maps parameters like temperature, maxTokens, topK, topP, stopSequences,
        presence/frequency penalties, and systemPrompt.
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
            # Assuming systemPrompt maps to system_instruction for Google's API
            config_args["system_instruction"] = request_params.systemPrompt
        return types.GenerateContentConfig(**config_args)

    def convert_from_google_content_list(
        self, contents: List[types.Content]
    ) -> List[PromptMessageMultipart]:
        """
        Converts a list of google.genai types.Content to a list of fast-agent PromptMessageMultipart.
        """
        return [self._convert_from_google_content(content) for content in contents]

    def _convert_from_google_content(self, content: types.Content) -> PromptMessageMultipart:
        """
        Converts a single google.genai types.Content to a fast-agent PromptMessageMultipart.
        Handles different Google content parts (text, function_response, file_data) and roles.
        If the content is a model response with a function call, it's treated as an empty assistant message
        as the function call itself is handled separately.
        """
        if content.role == "model" and any(part.function_call for part in content.parts):
            # Function calls are typically extracted and handled by CallToolRequestParams,
            # so the main message content might be empty or represent precursor text.
            # Here, we return an empty assistant message if a function_call is present in any part.
            return PromptMessageMultipart(role="assistant", content=[])

        fast_agent_parts: List[
            TextContent
            | ImageContent
            | EmbeddedResource
            | CallToolRequestParams  # Though CallToolRequestParams won't be added here due to above check
        ] = []
        for part in content.parts:
            if part.text:
                fast_agent_parts.append(TextContent(type="text", text=part.text))
            elif part.function_response:
                response_data = part.function_response.response
                if isinstance(response_data, dict) and "text_content" in response_data:
                    response_text = str(response_data["text_content"])
                else:
                    response_text = str(
                        response_data
                    )  # Fallback if response is not a dict or no "text_content"
                fast_agent_parts.append(TextContent(type="text", text=response_text))
            elif part.file_data:  # Convert file_data to a generic EmbeddedResource with TextContent
                fast_agent_parts.append(
                    EmbeddedResource(
                        type="resource",
                        resource=TextContent(
                            uri=part.file_data.file_uri,
                            mimeType=part.file_data.mime_type,
                            text=f"[Resource: {part.file_data.file_uri}, MIME: {part.file_data.mime_type}]",  # Placeholder text
                        ),
                    )
                )

        fast_agent_role = (
            "user" if content.role == "user" else "assistant"
        )  # Default to assistant for "model" or "tool" roles not caught above
        return PromptMessageMultipart(role=fast_agent_role, content=fast_agent_parts)
