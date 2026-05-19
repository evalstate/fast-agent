import base64
import unittest
from typing import TYPE_CHECKING

from mcp.types import (
    CallToolResult,
    ImageContent,
    TextContent,
)

from fast_agent.llm.provider.google.google_converter import GoogleConverter

if TYPE_CHECKING:
    from google.genai.types import Content


class TestOpenAIToolConverter(unittest.TestCase):
    """Test cases for conversion of tool results to OpenAI tool messages."""

    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a tool result"
        self.converter = GoogleConverter()

    def test_tool_result_conversion(self):
        """Test conversion of CallToolResult to OpenAI tool message."""
        # Create a tool result with text content
        text_content = TextContent(type="text", text=self.sample_text)
        tool_result = CallToolResult(content=[text_content], isError=False)

        # Create a tool call ID
        #        tool_call_id = "call_abc123"

        # Convert directly to OpenAI tool message
        converted: list[Content] = self.converter.convert_function_results_to_google(
            [("test", None, tool_result)]
        )
        assert 1 == len(converted)
        assert "user" == converted[0].role
        parts = converted[0].parts
        assert parts is not None
        fn_resp = parts[0].function_response
        assert fn_resp is not None
        response = fn_resp.response
        assert isinstance(response, dict)
        assert self.sample_text == response["result"]

    def test_multiple_tool_results_with_mixed_content(self):
        """Test conversion of multiple tool results with different content types."""
        # Create first tool result with text only
        text_result = CallToolResult(
            content=[TextContent(type="text", text="Text-only result")], isError=False
        )

        # Create second tool result with image
        image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")
        image_content = ImageContent(type="image", data=image_base64, mimeType="image/jpeg")
        image_result = CallToolResult(
            content=[TextContent(type="text", text="Here's the image:"), image_content],
            isError=False,
        )

        # Create tool names/ids
        tool_name1 = "text_tool"
        tool_name2 = "image_tool"
        tool_call_id1 = "call_text_only"
        tool_call_id2 = "call_with_image"

        results: list[tuple[str, str | None, CallToolResult]] = [
            (tool_name1, tool_call_id1, text_result),
            (tool_name2, tool_call_id2, image_result),
        ]

        # Convert to OpenAI tool messages
        converted: list[Content] = self.converter.convert_function_results_to_google(results)

        # Assertions
        assert 1 == len(converted)
        content = converted[0]
        assert content.role == "user"
        parts = content.parts
        assert parts is not None
        assert 2 == len(parts)

        # First function response part (Text Only)
        fn_resp1 = parts[0].function_response
        assert fn_resp1 is not None
        assert fn_resp1.id == tool_call_id1
        assert fn_resp1.name == tool_name1
        assert isinstance(fn_resp1.response, dict)
        assert fn_resp1.response["result"] == "Text-only result"

        # Second function response part (With image)
        fn_resp2 = parts[1].function_response
        assert fn_resp2 is not None
        assert fn_resp2.id == tool_call_id2
        assert fn_resp2.name == tool_name2
        second_response = fn_resp2.response
        assert isinstance(second_response, dict)
        assert second_response["result"] == "Here's the image:"
        assert fn_resp2.parts is not None
        assert len(fn_resp2.parts) == 1


#        assert self.sample_text == converted[0].parts[0].function_response.response["text"][0]

# # Check first tool message (text only)
# self.assertEqual(tool_messages[0]["role"], "tool")
# self.assertEqual(tool_messages[0]["tool_call_id"], tool_call_id1)
# self.assertEqual(tool_messages[0]["content"], "Text-only result")

# # Check second tool message (with image)
# self.assertEqual(tool_messages[1]["role"], "tool")
# self.assertEqual(tool_messages[1]["tool_call_id"], tool_call_id2)
# self.assertEqual(tool_messages[1]["content"], "Here's the image:")
# self.assertEqual(tool_messages[2]["role"], "user")
# self.assertEqual(tool_messages[2]["content"][0]["type"], "image_url")
