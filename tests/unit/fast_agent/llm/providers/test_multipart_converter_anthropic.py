import base64
import unittest

from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import (
    AnthropicConverter,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from mcp_agent.mcp.resource_utils import normalize_uri

PDF_BASE64 = base64.b64encode(b"fake_pdf_data").decode("utf-8")


def create_pdf_resource(pdf_base64) -> EmbeddedResource:
    pdf_resource: BlobResourceContents = BlobResourceContents(
        uri="test://example.com/document.pdf",
        mimeType="application/pdf",
        blob=pdf_base64,
    )
    return EmbeddedResource(type="resource", resource=pdf_resource)


class TestAnthropicUserConverter(unittest.TestCase):
    """Test cases for conversion from user role MCP message types to Anthropic API."""

    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a test message"
        self.sample_image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")

    def test_text_content_conversion(self):
        """Test conversion of TextContent to Anthropic text block."""
        # Create a text content message
        text_content = TextContent(type="text", text=self.sample_text)
        multipart = PromptMessageExtended(role="user", content=[text_content])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access, not attribute access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)

    def test_image_content_conversion(self):
        """Test conversion of ImageContent to Anthropic image block."""
        # Create an image content message
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        multipart = PromptMessageExtended(role="user", content=[image_content])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "image")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "base64")
        self.assertEqual(anthropic_msg["content"][0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], self.sample_image_base64)

    def test_embedded_resource_text_conversion(self):
        """Test conversion of text-based EmbeddedResource to Anthropic document block."""
        # Create a text resource
        text_resource = TextResourceContents(
            uri="test://example.com/document.txt",
            mimeType="text/plain",
            text=self.sample_text,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=text_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "document")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["title"], "document.txt")
        self.assertEqual(anthropic_msg["content"][0]["source"]["media_type"], "text/plain")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], self.sample_text)

    def test_embedded_resource_pdf_conversion(self):
        """Test conversion of PDF EmbeddedResource to Anthropic document block."""
        # Create a PDF resource
        pdf_resource = create_pdf_resource(PDF_BASE64)
        multipart = PromptMessageExtended(role="user", content=[pdf_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "document")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "base64")
        self.assertEqual(anthropic_msg["content"][0]["source"]["media_type"], "application/pdf")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], PDF_BASE64)

    def test_embedded_resource_image_url_conversion(self):
        """Test conversion of image URL in EmbeddedResource to Anthropic image block."""
        # Create an image resource with URL
        image_resource = BlobResourceContents(
            uri="https://example.com/image.jpg",
            mimeType="image/jpeg",
            blob=self.sample_image_base64,  # This should be ignored for URL
        )
        embedded_resource = EmbeddedResource(type="resource", resource=image_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "image")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "url")
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["url"],
            "https://example.com/image.jpg",
        )

    def test_assistant_role_restrictions(self):
        """Test that assistant messages can only contain text blocks."""
        # Create mixed content for assistant
        text_content = TextContent(type="text", text=self.sample_text)
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        multipart = PromptMessageExtended(role="assistant", content=[text_content, image_content])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - only text should remain
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)

    def test_multiple_content_blocks(self):
        """Test conversion of messages with multiple content blocks."""
        # Create multiple content blocks
        text_content1 = TextContent(type="text", text="First text")
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        text_content2 = TextContent(type="text", text="Second text")

        multipart = PromptMessageExtended(
            role="user", content=[text_content1, image_content, text_content2]
        )

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 3)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], "First text")
        self.assertEqual(anthropic_msg["content"][1]["type"], "image")
        self.assertEqual(anthropic_msg["content"][2]["type"], "text")
        self.assertEqual(anthropic_msg["content"][2]["text"], "Second text")

    def test_unsupported_mime_type_handling(self):
        """Test handling of unsupported MIME types."""
        # Create an image with unsupported mime type
        image_content = ImageContent(
            type="image",
            data=self.sample_image_base64,
            mimeType="image/bmp",  # Unsupported in Anthropic API
        )
        text_content = TextContent(type="text", text="This is some text")
        multipart = PromptMessageExtended(role="user", content=[text_content, image_content])

        # Convert to Anthropic format - should convert unsupported image to text fallback
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have kept the text content and added a fallback text for the image
        self.assertEqual(len(anthropic_msg["content"]), 2)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], "This is some text")
        self.assertEqual(anthropic_msg["content"][1]["type"], "text")
        self.assertIn(
            "Image with unsupported format 'image/bmp'",
            anthropic_msg["content"][1]["text"],
        )

    def test_svg_resource_conversion(self):
        """Test handling of SVG resources - should convert to code block."""
        # Create an embedded SVG resource
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
        svg_resource = TextResourceContents(
            uri="test://example.com/image.svg",
            mimeType="image/svg+xml",
            text=svg_content,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=svg_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to Anthropic format - should extract SVG as text
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should be converted to a text block with the SVG code
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertIn("```xml", anthropic_msg["content"][0]["text"])
        self.assertIn(svg_content, anthropic_msg["content"][0]["text"])

    def test_empty_content_list(self):
        """Test conversion with empty content list."""
        multipart = PromptMessageExtended(role="user", content=[])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have empty content list
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 0)

    def test_embedded_resource_pdf_url_conversion(self):
        """Test conversion of PDF URL in EmbeddedResource to Anthropic document block."""
        # Create a PDF resource with URL
        pdf_resource = BlobResourceContents(
            uri="https://example.com/document.pdf",
            mimeType="application/pdf",
            blob=base64.b64encode(b"fake_pdf_data").decode("utf-8"),
        )
        embedded_resource = EmbeddedResource(type="resource", resource=pdf_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions - using dictionary access
        self.assertEqual(anthropic_msg["content"][0]["type"], "document")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "url")
        self.assertEqual(
            anthropic_msg["content"][0]["source"]["url"],
            "https://example.com/document.pdf",
        )

    def test_mixed_content_with_unsupported_formats(self):
        """Test conversion of mixed content where some items are unsupported."""
        # Create mixed content with supported and unsupported items
        text_content = TextContent(type="text", text=self.sample_text)
        unsupported_image = ImageContent(
            type="image",
            data=self.sample_image_base64,
            mimeType="image/bmp",  # Unsupported
        )
        supported_image = ImageContent(
            type="image",
            data=self.sample_image_base64,
            mimeType="image/jpeg",  # Supported
        )

        multipart = PromptMessageExtended(
            role="user", content=[text_content, unsupported_image, supported_image]
        )

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have kept the text, created fallback for unsupported, and kept supported image
        self.assertEqual(len(anthropic_msg["content"]), 3)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)
        self.assertEqual(
            anthropic_msg["content"][1]["type"], "text"
        )  # Fallback text for unsupported
        self.assertEqual(anthropic_msg["content"][2]["type"], "image")  # Supported image kept
        self.assertEqual(anthropic_msg["content"][2]["source"]["media_type"], "image/jpeg")

    def test_multipart_with_tool_results_and_content(self):
        """Test conversion of PromptMessageExtended with both tool_results and content."""
        # Create tool results
        tool_result = CallToolResult(
            content=[TextContent(type="text", text="Tool execution result")], isError=False
        )

        # Create additional content
        additional_text = TextContent(type="text", text="What should I do next?")

        # Create multipart message with both tool_results and content
        multipart = PromptMessageExtended(
            role="user", content=[additional_text], tool_results={"tool_id_1": tool_result}
        )

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 2)

        # First block should be tool_result (must come first per Anthropic API)
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(anthropic_msg["content"][0]["tool_use_id"], "tool_id_1")
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["text"], "Tool execution result")

        # Second block should be the additional content
        self.assertEqual(anthropic_msg["content"][1]["type"], "text")
        self.assertEqual(anthropic_msg["content"][1]["text"], "What should I do next?")

    def test_code_file_as_text_document_with_filename(self):
        """Test handling of code files using a simple filename."""
        code_text = "def hello_world():\n    print('Hello, world!')"

        # Use the helper function with simple filename
        code_resource = create_text_resource(
            text=code_text, filename_or_uri="example.py", mime_type="text/x-python"
        )

        embedded_resource = EmbeddedResource(type="resource", resource=code_resource)

        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Check that title is set correctly
        self.assertEqual(anthropic_msg["content"][0]["title"], "example.py")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], code_text)
        self.assertEqual(anthropic_msg["content"][0]["source"]["media_type"], "text/plain")

    def test_code_file_as_text_document_with_uri(self):
        """Test handling of code files using a proper URI."""
        code_text = "def hello_world():\n    print('Hello, world!')"

        # Use the helper function with full URI
        code_resource = create_text_resource(
            text=code_text,
            filename_or_uri="file:///projects/example.py",
            mime_type="text/x-python",
        )

        embedded_resource = EmbeddedResource(type="resource", resource=code_resource)

        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should extract just the filename from the path
        self.assertEqual(anthropic_msg["content"][0]["title"], "example.py")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], code_text)

    def test_unsupported_binary_resource_conversion(self):
        """Test handling of unsupported binary resource types."""
        # Create an embedded resource with binary data
        binary_data = base64.b64encode(b"This is binary data").decode("utf-8")  # 20 bytes of data
        binary_resource = BlobResourceContents(
            uri="test://example.com/data.bin",
            mimeType="application/octet-stream",
            blob=binary_data,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=binary_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to Anthropic format - should create text fallback
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have a fallback text block
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")

        # Check that the content describes it as unsupported format
        fallback_text = anthropic_msg["content"][0]["text"]
        self.assertIn(
            "Embedded Resource test://example.com/data.bin with unsupported format application/octet-stream (28 characters)",
            fallback_text,
        )


class TestAnthropicToolConverter(unittest.TestCase):
    """Test cases for conversion of tool results to Anthropic API format."""

    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a tool result"
        self.sample_image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")
        self.tool_use_id = "toolu_01D7FLrfh4GYq7yT1ULFeyMV"

    def test_pdf_result_conversion(self):
        """Test conversion places PDF inside the tool_result content."""
        # Create a tool result with text and PDF content
        text_content = TextContent(type="text", text=self.sample_text)
        pdf_content = create_pdf_resource(PDF_BASE64)
        tool_result = CallToolResult(content=[text_content, pdf_content], isError=False)

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.create_tool_results_message(
            [(self.tool_use_id, tool_result)]
        )

        # Assertions
        self.assertEqual(anthropic_msg["role"], "user")
        # Now a single tool_result block that contains both text and the document
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(anthropic_msg["content"][0]["tool_use_id"], self.tool_use_id)
        self.assertEqual(len(anthropic_msg["content"][0]["content"]), 2)
        # First inner block: text
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["text"], self.sample_text)
        # Second inner block: document (PDF)
        self.assertEqual(anthropic_msg["content"][0]["content"][1]["type"], "document")
        self.assertEqual(anthropic_msg["content"][0]["content"][1]["source"]["type"], "base64")
        self.assertEqual(
            anthropic_msg["content"][0]["content"][1]["source"]["media_type"], "application/pdf"
        )
        self.assertEqual(anthropic_msg["content"][0]["content"][1]["source"]["data"], PDF_BASE64)

    def test_binary_only_tool_result_conversion(self):
        """Binary-only tool result should be a single tool_result with a document inside."""
        # Create a PDF embedded resource with no text content
        pdf_content = create_pdf_resource(PDF_BASE64)
        tool_result = CallToolResult(content=[pdf_content], isError=False)

        # Test the message creation with this result
        anthropic_msg = AnthropicConverter.create_tool_results_message(
            [(self.tool_use_id, tool_result)]
        )

        # Should have a single tool_result block with the PDF document inside
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(len(anthropic_msg["content"][0]["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "document")

    def test_create_tool_results_message(self):
        """Test creation of user message with multiple tool results."""
        # Create two tool results
        text_content = TextContent(type="text", text=self.sample_text)
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )

        tool_result1 = CallToolResult(content=[text_content], isError=False)

        tool_result2 = CallToolResult(content=[image_content], isError=False)

        tool_use_id1 = "tool_id_1"
        tool_use_id2 = "tool_id_2"

        # Create tool results list
        tool_results = [(tool_use_id1, tool_result1), (tool_use_id2, tool_result2)]

        # Convert to Anthropic message
        anthropic_msg = AnthropicConverter.create_tool_results_message(tool_results)

        # Assertions
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 2)

        # Check first tool result
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(anthropic_msg["content"][0]["tool_use_id"], tool_use_id1)
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["text"], self.sample_text)

        # Check second tool result
        self.assertEqual(anthropic_msg["content"][1]["type"], "tool_result")
        self.assertEqual(anthropic_msg["content"][1]["tool_use_id"], tool_use_id2)
        self.assertEqual(anthropic_msg["content"][1]["content"][0]["type"], "image")

    def test_create_tool_results_message_with_error(self):
        """Test creation of tool results message with error flag."""
        # Create a tool result with error flag set
        error_content = TextContent(type="text", text="Error: Something went wrong")
        tool_result = CallToolResult(content=[error_content], isError=True)
        tool_use_id = "tool_error_id"

        # Convert to Anthropic message
        anthropic_msg = AnthropicConverter.create_tool_results_message([(tool_use_id, tool_result)])

        # Assertions
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(anthropic_msg["content"][0]["tool_use_id"], tool_use_id)
        self.assertEqual(anthropic_msg["content"][0]["is_error"], True)
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "text")
        self.assertEqual(
            anthropic_msg["content"][0]["content"][0]["text"], "Error: Something went wrong"
        )

    def test_create_tool_results_message_with_empty_content(self):
        """Test creation of tool results message with empty content."""
        # Create a tool result with no content
        tool_result = CallToolResult(content=[], isError=False)
        tool_use_id = "tool_empty_id"

        # Convert to Anthropic message
        anthropic_msg = AnthropicConverter.create_tool_results_message([(tool_use_id, tool_result)])

        # Should have a placeholder text block
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(anthropic_msg["content"][0]["tool_use_id"], tool_use_id)
        self.assertEqual(len(anthropic_msg["content"][0]["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "text")
        self.assertEqual(
            anthropic_msg["content"][0]["content"][0]["text"], "[No content in tool result]"
        )

    def test_create_tool_results_message_with_unsupported_image(self):
        """Test handling of unsupported image format in tool results message."""
        # Create a tool result with unsupported image format
        unsupported_image = ImageContent(
            type="image",
            data=self.sample_image_base64,
            mimeType="image/bmp",  # Unsupported
        )
        tool_result = CallToolResult(content=[unsupported_image], isError=False)
        tool_use_id = "tool_unsupported_id"

        # Convert to Anthropic message
        anthropic_msg = AnthropicConverter.create_tool_results_message([(tool_use_id, tool_result)])

        # Unsupported image should be converted to text
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(len(anthropic_msg["content"][0]["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "text")
        self.assertIn(
            "Image with unsupported format 'image/bmp'",
            anthropic_msg["content"][0]["content"][0]["text"],
        )

    def test_create_tool_results_message_with_mixed_content(self):
        """Test creation of tool results message with mixed text and image content."""
        # Create a tool result with text and image content
        text_content = TextContent(type="text", text=self.sample_text)
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        tool_result = CallToolResult(content=[text_content, image_content], isError=False)
        tool_use_id = "tool_mixed_id"

        # Convert to Anthropic message
        anthropic_msg = AnthropicConverter.create_tool_results_message([(tool_use_id, tool_result)])

        # Assertions
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(anthropic_msg["content"][0]["tool_use_id"], tool_use_id)
        self.assertEqual(len(anthropic_msg["content"][0]["content"]), 2)
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["text"], self.sample_text)
        self.assertEqual(anthropic_msg["content"][0]["content"][1]["type"], "image")

    def test_create_tool_results_message_with_text_resource(self):
        """Test creation of tool results message with text resource (markdown)."""
        markdown_content = EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=AnyUrl("resource://test/content"),
                mimeType="text/markdown",
                text="markdown text",
            ),
        )
        tool_result = CallToolResult(content=[markdown_content], isError=False)
        tool_use_id = "tool_markdown_id"

        # Convert to Anthropic message
        anthropic_msg = AnthropicConverter.create_tool_results_message([(tool_use_id, tool_result)])

        # Text resources should be included in tool result as text blocks
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "tool_result")
        self.assertEqual(anthropic_msg["content"][0]["tool_use_id"], tool_use_id)
        self.assertEqual(len(anthropic_msg["content"][0]["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["content"][0]["text"], "markdown text")


def create_text_resource(
    text: str, filename_or_uri: str, mime_type: str = None
) -> TextResourceContents:
    """
    Helper function to create a TextResourceContents with proper URI handling.

    Args:
        text: The text content
        filename_or_uri: A filename or URI
        mime_type: Optional MIME type

    Returns:
        A properly configured TextResourceContents
    """
    # Normalize the URI
    uri = normalize_uri(filename_or_uri)

    return TextResourceContents(uri=uri, mimeType=mime_type, text=text)


class TestAnthropicAssistantConverter(unittest.TestCase):
    """Test cases for conversion from assistant role MCP message types to Anthropic API."""

    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a response from the assistant"

    def test_assistant_text_content_conversion(self):
        """Test conversion of assistant TextContent to Anthropic text block."""
        # Create a text content message from assistant
        text_content = TextContent(type="text", text=self.sample_text)
        multipart = PromptMessageExtended(role="assistant", content=[text_content])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)

    def test_convert_prompt_message_to_anthropic(self):
        """Test conversion of a standard PromptMessage to Anthropic format."""
        # Create a PromptMessage with TextContent
        text_content = TextContent(type="text", text=self.sample_text)
        prompt_message = PromptMessage(role="assistant", content=text_content)

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_prompt_message_to_anthropic(prompt_message)

        # Assertions
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)

    def test_convert_prompt_message_image_to_anthropic(self):
        """Test conversion of a PromptMessage with image content to Anthropic format."""
        # Create a PromptMessage with ImageContent
        image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")
        image_content = ImageContent(type="image", data=image_base64, mimeType="image/jpeg")
        prompt_message = PromptMessage(role="user", content=image_content)

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_prompt_message_to_anthropic(prompt_message)

        # Assertions
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "image")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "base64")
        self.assertEqual(anthropic_msg["content"][0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], image_base64)

    def test_convert_prompt_message_embedded_resource_to_anthropic(self):
        """Test conversion of a PromptMessage with embedded resource to Anthropic format."""
        # Create a PromptMessage with embedded text resource
        text_resource = TextResourceContents(
            uri="test://example.com/document.txt",
            mimeType="text/plain",
            text="This is a text resource",
        )
        embedded_resource = EmbeddedResource(type="resource", resource=text_resource)
        prompt_message = PromptMessage(role="user", content=embedded_resource)

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_prompt_message_to_anthropic(prompt_message)

        # Assertions
        self.assertEqual(anthropic_msg["role"], "user")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "document")
        self.assertEqual(anthropic_msg["content"][0]["source"]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["title"], "document.txt")
        self.assertEqual(anthropic_msg["content"][0]["source"]["data"], "This is a text resource")

    def test_assistant_multiple_text_blocks(self):
        """Test conversion of assistant messages with multiple text blocks."""
        # Create multiple text content blocks
        text_content1 = TextContent(type="text", text="First part of response")
        text_content2 = TextContent(type="text", text="Second part of response")

        multipart = PromptMessageExtended(role="assistant", content=[text_content1, text_content2])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Assertions
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertEqual(len(anthropic_msg["content"]), 2)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], "First part of response")
        self.assertEqual(anthropic_msg["content"][1]["type"], "text")
        self.assertEqual(anthropic_msg["content"][1]["text"], "Second part of response")

    def test_assistant_non_text_content_stripped(self):
        """Test that non-text content is stripped from assistant messages."""
        # Create a mixed content message with text and image
        text_content = TextContent(type="text", text=self.sample_text)
        image_content = ImageContent(
            type="image",
            data=base64.b64encode(b"fake_image_data").decode("utf-8"),
            mimeType="image/jpeg",
        )

        multipart = PromptMessageExtended(role="assistant", content=[text_content, image_content])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Only text should remain, image should be filtered out
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)

    def test_assistant_embedded_resource_stripped(self):
        """Test that embedded resources are stripped from assistant messages."""
        # Create a message with text and embedded resource
        text_content = TextContent(type="text", text=self.sample_text)

        resource_content = TextResourceContents(
            uri="test://example.com/document.txt",
            mimeType="text/plain",
            text="Some document content",
        )
        embedded_resource = EmbeddedResource(type="resource", resource=resource_content)

        multipart = PromptMessageExtended(
            role="assistant", content=[text_content, embedded_resource]
        )

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Only text should remain, resource should be filtered out
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertEqual(len(anthropic_msg["content"]), 1)
        self.assertEqual(anthropic_msg["content"][0]["type"], "text")
        self.assertEqual(anthropic_msg["content"][0]["text"], self.sample_text)

    def test_assistant_empty_content(self):
        """Test conversion with empty content from assistant."""
        multipart = PromptMessageExtended(role="assistant", content=[])

        # Convert to Anthropic format
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart)

        # Should have empty content list
        self.assertEqual(anthropic_msg["role"], "assistant")
        self.assertEqual(len(anthropic_msg["content"]), 0)
