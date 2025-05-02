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
            # Check for data attribute *after* confirming it's ImageContent
            if hasattr(part, "data") and part.data:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": getattr(part, "mimeType", "image/png"),
                        "data": getattr(part, "data", ""),  # Safe now
                    },
                }
            else:
                _logger.warning(f"Skipping ImageContent without data: {part}")
        elif isinstance(part, EmbeddedResource):
            # TODO: Add handling for EmbeddedResource if T0 supports documents/other files
            _logger.warning(f"Skipping EmbeddedResource, T0 conversion not implemented: {part}")
        else:
            # This case handles potential future types or unexpected input
            _logger.warning(f"Unsupported content part type for T0 conversion: {type(part)}")
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
                    json_result = json.dumps(result_content_str)
                except TypeError as json_err:
                    _logger.error(
                        f"Failed to JSON encode tool result string: {result_content_str} - {json_err}"
                    )
                    json_result = json.dumps(str(result_content_str))  # Fallback

                t0_block = {
                    "type": "tool_result",
                    "id": tool_use_id,
                    "name": tool_name,
                    # Assign the JSON encoded string
                    "result": json_result,
                }
                t0_tool_result_blocks.append(t0_block)

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
            if isinstance(part, TextContent):
                t0_content_blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ImageContent):
                if hasattr(part, "data") and part.data:
                    t0_content_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": getattr(part, "mimeType", "image/png"),
                                "data": getattr(part, "data", ""),
                            },
                        }
                    )
                else:
                    _logger.warning(f"Skipping ImageContent without data: {part}")
            elif isinstance(part, CallToolResult):
                # Format embedded tool results
                contains_tool_result = True
                tool_use_id = getattr(part, "_t0_tool_use_id_temp", None)
                tool_name = getattr(part, "_t0_tool_name_temp", None)
                if tool_use_id and tool_name:
                    result_content_str = TensorZeroConverter._get_text_from_call_tool_result(part)
                    # Format result as a JSON object before dumping
                    result_object = {"text_result": result_content_str}
                    try:
                        json_result = json.dumps(result_object)
                    except TypeError as json_err:
                        _logger.error(
                            f"Failed to JSON encode tool result object: {result_object} - {json_err}"
                        )
                        json_result = json.dumps(str(result_content_str))

                    t0_content_blocks.append(
                        {
                            "type": "tool_result",
                            "id": tool_use_id,
                            "name": tool_name,
                            "result": json_result,
                        }
                    )
                    try:
                        delattr(part, "_t0_tool_use_id_temp")
                    except AttributeError:
                        pass
                    try:
                        delattr(part, "_t0_tool_name_temp")
                    except AttributeError:
                        pass
                else:
                    _logger.warning(
                        f"Found CallToolResult without required temp attributes: {part}"
                    )
            elif isinstance(part, EmbeddedResource):
                _logger.warning(f"Skipping EmbeddedResource, T0 conversion not implemented: {part}")
            else:
                _logger.warning(f"Unsupported content part type for T0 conversion: {type(part)}")

        if not t0_content_blocks:
            return None

        # Determine role - if content ONLY contains tool results, role must be 'user'
        # Otherwise, use the original role.
        valid_role = msg.role if msg.role in ["user", "assistant"] else "user"
        if contains_tool_result and all(
            block.get("type") == "tool_result" for block in t0_content_blocks
        ):
            final_role = "user"  # Force role to user if only tool results present
            if valid_role != final_role:
                _logger.debug(f"Overriding role to '{final_role}' for tool result message.")
        else:
            final_role = valid_role
            if valid_role != msg.role:
                _logger.warning(f"Mapping message role '{msg.role}' to '{valid_role}' for T0.")

        return {"role": final_role, "content": t0_content_blocks}

    # Add methods here if needed to convert *from* T0 format back to MCP types
    # e.g., adapt_t0_response_to_mcp(...) - this logic stays in the LLM class for now
