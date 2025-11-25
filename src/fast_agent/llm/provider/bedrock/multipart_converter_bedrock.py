from typing import Any

from fast_agent.types import PromptMessageExtended

# Bedrock message format types
BedrockMessageParam = dict[str, Any]


class BedrockConverter:
    """Converts MCP message types to Bedrock API format."""

    @staticmethod
    def convert_to_bedrock(multipart_msg: PromptMessageExtended) -> BedrockMessageParam:
        """
        Convert a PromptMessageExtended message to Bedrock API format.

        This is a wrapper around the instance method _convert_multipart_to_bedrock_message
        to provide a static interface similar to AnthropicConverter.

        Args:
            multipart_msg: The PromptMessageExtended message to convert

        Returns:
            A Bedrock API message parameter dictionary
        """
        # Simple conversion without needing BedrockLLM instance
        bedrock_msg = {"role": multipart_msg.role, "content": []}

        # Handle tool results first (if present)
        if multipart_msg.tool_results:
            import json

            from mcp.types import TextContent

            # Check if any tool ID indicates system prompt format
            has_system_prompt_tools = any(
                tool_id.startswith("system_prompt_") for tool_id in multipart_msg.tool_results.keys()
            )

            if has_system_prompt_tools:
                # For system prompt models: format as human-readable text
                tool_result_parts = []
                for tool_id, tool_result in multipart_msg.tool_results.items():
                    result_text = "".join(
                        part.text for part in tool_result.content if isinstance(part, TextContent)
                    )
                    result_payload = {
                        "tool_name": tool_id,
                        "status": "error" if tool_result.isError else "success",
                        "result": result_text,
                    }
                    tool_result_parts.append(json.dumps(result_payload))

                if tool_result_parts:
                    full_result_text = f"Tool Results:\n{', '.join(tool_result_parts)}"
                    bedrock_msg["content"].append({"type": "text", "text": full_result_text})
            else:
                # For Nova/Anthropic models: use structured tool_result format
                for tool_id, tool_result in multipart_msg.tool_results.items():
                    result_content_blocks = []
                    if tool_result.content:
                        for part in tool_result.content:
                            if isinstance(part, TextContent):
                                result_content_blocks.append({"text": part.text})

                    if not result_content_blocks:
                        result_content_blocks.append({"text": "[No content in tool result]"})

                    bedrock_msg["content"].append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result_content_blocks,
                            "status": "error" if tool_result.isError else "success",
                        }
                    )

        # Handle regular content
        from mcp.types import TextContent
        for content_item in multipart_msg.content:
            if isinstance(content_item, TextContent):
                bedrock_msg["content"].append({"type": "text", "text": content_item.text})

        return bedrock_msg
