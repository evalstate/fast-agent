import json
from typing import Any, Dict, List, Optional, Union

from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import (
    get_text,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

_logger = get_logger(__name__)


class TensorZeroConverter:
    """Converts MCP message types to/from TensorZero API format."""

    @staticmethod
    def _convert_content_part(
        part: Union[TextContent, ImageContent, EmbeddedResource],
    ) -> Optional[Dict[str, Any]]:
        """Converts a single MCP content part to a T0 content block dictionary."""
        if isinstance(part, TextContent):
            text = get_text(part)
            if text is not None:
                return {"type": "text", "text": text}
        elif isinstance(part, ImageContent):
            # Handle Base64: needs data, mimeType (and mimeType must not be empty)
            if hasattr(part, "data") and part.data and hasattr(part, "mimeType") and part.mimeType:
                _logger.debug(
                    f"Converting ImageContent as base64 for T0 native: mime={part.mimeType}, data_len={len(part.data) if isinstance(part.data, str) else 'N/A'}"
                )
                supported_mime_types = ["image/jpeg", "image/png", "image/webp"]
                mime_type = getattr(part, "mimeType", "")

                # Use the provided mime_type if supported, otherwise default to png
                if mime_type not in supported_mime_types:
                    _logger.warning(
                        f"Unsupported mimeType '{mime_type}' for T0 base64 image, defaulting to image/png."
                    )
                    mime_type = "image/png"

                return {
                    "type": "image",
                    "mime_type": mime_type,  # Note: T0 uses mime_type, not media_type
                    "data": getattr(part, "data", ""),  # Data is direct property
                }
            else:
                # Log cases where it's an ImageContent but doesn't fit Base64 criteria
                _logger.warning(
                    f"Skipping ImageContent: Missing required base64 fields (mimeType/data), or mimeType is empty: {part}"
                )

        elif isinstance(part, EmbeddedResource):
            _logger.warning(f"Skipping EmbeddedResource, T0 conversion not implemented: {part}")
        else:
            _logger.error(
                f"Unsupported content part type for T0 conversion: {type(part)}"
            )  # Changed to error

        return None  # Return None if no block was successfully created

    @staticmethod
    def _get_text_from_call_tool_result(result: CallToolResult) -> str:
        """Helper to extract combined text from a CallToolResult's content list."""
        texts = []
        if result.content:
            for part in result.content:
                text = get_text(part)
                if text:
                    texts.append(text)
        return "\n".join(texts)

    @staticmethod
    def convert_tool_results_to_t0_user_message(
        results: List[CallToolResult],
    ) -> Optional[Dict[str, Any]]:
        """Formats CallToolResult list into T0's tool_result blocks within a user message dict."""
        t0_tool_result_blocks = []
        for result in results:
            tool_use_id = getattr(result, "_t0_tool_use_id_temp", None)
            tool_name = getattr(result, "_t0_tool_name_temp", None)

            if tool_use_id and tool_name:
                result_content_str = TensorZeroConverter._get_text_from_call_tool_result(result)
                try:
                    # Attempt to treat result as JSON if possible, else use raw string
                    try:
                        json_result = json.loads(result_content_str)
                    except json.JSONDecodeError:
                        json_result = result_content_str  # Fallback to string if not valid JSON
                except Exception as e:
                    _logger.error(f"Unexpected error processing tool result content: {e}")
                    json_result = str(result_content_str)  # Safest fallback

                t0_block = {
                    "type": "tool_result",
                    "id": tool_use_id,
                    "name": tool_name,
                    "result": json_result,  # T0 expects the result directly
                }
                t0_tool_result_blocks.append(t0_block)

                # Clean up temporary attributes
                try:
                    delattr(result, "_t0_tool_use_id_temp")
                    delattr(result, "_t0_tool_name_temp")
                    if hasattr(result, "_t0_is_error_temp"):
                        delattr(result, "_t0_is_error_temp")
                except AttributeError:
                    pass
            else:
                _logger.warning(
                    f"Could not find id/name temp attributes for CallToolResult: {result}"
                )

        if not t0_tool_result_blocks:
            return None

        return {"role": "user", "content": t0_tool_result_blocks}

    @staticmethod
    def convert_mcp_to_t0_message(msg: PromptMessageMultipart) -> Optional[Dict[str, Any]]:
        """
        Converts a single PromptMessageMultipart to a T0 API message dictionary.
        Handles Text, Image, and embedded CallToolResult content.
        Skips system messages.
        """
        if msg.role == "system":
            return None

        t0_content_blocks = []
        contains_tool_result = False

        for part in msg.content:
            # Use the corrected _convert_content_part
            converted_block = TensorZeroConverter._convert_content_part(part)
            if converted_block:
                t0_content_blocks.append(converted_block)
            elif isinstance(part, CallToolResult):
                # Existing logic for handling embedded CallToolResult (seems compatible with T0 tool_result spec)
                contains_tool_result = True
                tool_use_id = getattr(part, "_t0_tool_use_id_temp", None)
                tool_name = getattr(part, "_t0_tool_name_temp", None)
                if tool_use_id and tool_name:
                    result_content_str = TensorZeroConverter._get_text_from_call_tool_result(part)
                    # Try to format result as JSON object/string
                    try:
                        json_result = json.loads(result_content_str)
                    except json.JSONDecodeError:
                        json_result = result_content_str  # Fallback
                    except Exception as e:
                        _logger.error(f"Error processing embedded tool result: {e}")
                        json_result = str(result_content_str)

                    t0_content_blocks.append(
                        {
                            "type": "tool_result",
                            "id": tool_use_id,
                            "name": tool_name,
                            "result": json_result,
                        }
                    )
                    # Clean up temp attributes
                    try:
                        delattr(part, "_t0_tool_use_id_temp")
                        delattr(part, "_t0_tool_name_temp")
                    except AttributeError:
                        pass
                else:
                    _logger.warning(
                        f"Found embedded CallToolResult without required temp attributes: {part}"
                    )
            # Note: The _convert_content_part handles logging for other skipped/unsupported types

        if not t0_content_blocks:
            return None

        # Determine role - logic remains the same
        valid_role = msg.role if msg.role in ["user", "assistant"] else "user"
        if contains_tool_result and all(
            block.get("type") == "tool_result" for block in t0_content_blocks
        ):
            final_role = "user"
            if valid_role != final_role:
                _logger.debug(f"Overriding role to '{final_role}' for tool result message.")
        else:
            final_role = valid_role
            if valid_role != msg.role:
                _logger.warning(f"Mapping message role '{msg.role}' to '{valid_role}' for T0.")

        return {"role": final_role, "content": t0_content_blocks}

    # Add methods here if needed to convert *from* T0 format back to MCP types
    # e.g., adapt_t0_response_to_mcp(...) - this logic stays in the LLM class for now
