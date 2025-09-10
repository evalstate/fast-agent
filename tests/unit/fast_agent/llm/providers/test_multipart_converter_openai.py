import base64
import unittest

from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from fast_agent.llm.provider.openai import llm_openai
from fast_agent.llm.provider.openai.multipart_converter_openai import (
    OpenAIConverter,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class TestOpenAIUserConverter(unittest.TestCase):
    """Test cases for conversion from user role MCP message types to OpenAI API."""

    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a test message"
        self.sample_image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")

    def test_text_content_conversion(self):
        """Test conversion of TextContent to OpenAI text content."""
        # Create a text content message
        text_content = TextContent(type="text", text=self.sample_text)
        multipart = PromptMessageExtended(role="user", content=[text_content])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(openai_msg["content"], self.sample_text)

    def test_image_content_conversion(self):
        """Test conversion of ImageContent to OpenAI image block."""
        # Create an image content message
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        multipart = PromptMessageExtended(role="user", content=[image_content])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "image_url")
        self.assertEqual(
            openai_msg["content"][0]["image_url"]["url"],
            f"data:image/jpeg;base64,{self.sample_image_base64}",
        )

    def test_embedded_resource_text_conversion(self):
        """Test conversion of text-based EmbeddedResource to OpenAI text content with fastagent:file tags."""
        # Create a text resource
        text_resource = TextResourceContents(
            uri="test://example.com/document.txt",
            mimeType="text/plain",
            text=self.sample_text,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=text_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertIn("<fastagent:file", openai_msg["content"][0]["text"])
        self.assertIn('title="document.txt"', openai_msg["content"][0]["text"])
        self.assertIn('mimetype="text/plain"', openai_msg["content"][0]["text"])
        self.assertIn(self.sample_text, openai_msg["content"][0]["text"])
        self.assertIn("</fastagent:file>", openai_msg["content"][0]["text"])

    def test_embedded_resource_pdf_conversion(self):
        """Test conversion of PDF EmbeddedResource to OpenAI file part."""
        # Create a PDF resource
        pdf_base64 = base64.b64encode(b"fake_pdf_data").decode("utf-8")
        pdf_resource = BlobResourceContents(
            uri="test://example.com/document.pdf",
            mimeType="application/pdf",
            blob=pdf_base64,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=pdf_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "file")
        self.assertEqual(openai_msg["content"][0]["file"]["filename"], "document.pdf")
        self.assertEqual(
            openai_msg["content"][0]["file"]["file_data"],
            f"data:application/pdf;base64,{pdf_base64}",
        )

    def test_embedded_resource_image_url_conversion(self):
        """Test conversion of image URL in EmbeddedResource to OpenAI image block."""
        # Create an image resource with URL
        image_resource = BlobResourceContents(
            uri="https://example.com/image.jpg",
            mimeType="image/jpeg",
            blob=self.sample_image_base64,  # This would be ignored for URL in OpenAI
        )
        embedded_resource = EmbeddedResource(type="resource", resource=image_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "image_url")
        self.assertEqual(
            openai_msg["content"][0]["image_url"]["url"],
            "https://example.com/image.jpg",
        )

    def test_linked_resource_conversion(self):
        """Test conversion of text-based EmbeddedResource to OpenAI text content with fastagent:file tags."""
        # Create a text resource
        resource_link = ResourceLink(
            uri=AnyUrl("test://example.com/document.txt"),
            type="resource_link",
            mimeType="text/plain",
            description="Some description",
            name="some name",
        )
        multipart = PromptMessageExtended(role="user", content=[resource_link])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertIn("Some description", openai_msg["content"][0]["text"])
        self.assertIn("some name", openai_msg["content"][0]["text"])
        self.assertIn("text/plain", openai_msg["content"][0]["text"])
        self.assertIn("test://example.com/document.txt", openai_msg["content"][0]["text"])

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

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 3)
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertEqual(openai_msg["content"][0]["text"], "First text")
        self.assertEqual(openai_msg["content"][1]["type"], "image_url")
        self.assertEqual(openai_msg["content"][2]["type"], "text")
        self.assertEqual(openai_msg["content"][2]["text"], "Second text")

    def test_svg_resource_conversion(self):
        """Test handling of SVG resources - should convert to text with fastagent:file tags for OpenAI."""
        # Create an embedded SVG resource
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
        svg_resource = TextResourceContents(
            uri="test://example.com/image.svg",
            mimeType="image/svg+xml",
            text=svg_content,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=svg_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Should be converted to a text block with the SVG in fastagent:file tags
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertIn("<fastagent:file", openai_msg["content"][0]["text"])
        self.assertIn('title="image.svg"', openai_msg["content"][0]["text"])
        self.assertIn('mimetype="image/svg+xml"', openai_msg["content"][0]["text"])
        self.assertIn(svg_content, openai_msg["content"][0]["text"])
        self.assertIn("</fastagent:file>", openai_msg["content"][0]["text"])

    def test_empty_content_list(self):
        """Test conversion with empty content list."""
        multipart = PromptMessageExtended(role="user", content=[])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Should have empty content
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(openai_msg["content"], "")

    def test_code_file_conversion(self):
        """Test handling of code files as text with fastagent:file tags."""
        code_text = "def hello_world():\n    print('Hello, world!')"

        # Create a code resource
        code_resource = TextResourceContents(
            uri="test://example.com/example.py",
            mimeType="text/x-python",
            text=code_text,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=code_resource)

        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to OpenAI format (now returns a list)
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Check that proper fastagent:file tags are used
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertIn("<fastagent:file", openai_msg["content"][0]["text"])
        self.assertIn('title="example.py"', openai_msg["content"][0]["text"])
        self.assertIn('mimetype="text/x-python"', openai_msg["content"][0]["text"])
        self.assertIn(code_text, openai_msg["content"][0]["text"])
        self.assertIn("</fastagent:file>", openai_msg["content"][0]["text"])


class TestOpenAIAssistantConverter(unittest.TestCase):
    """Test cases for conversion from assistant role MCP message types to OpenAI API."""

    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a response from the assistant"

    def test_assistant_text_content_conversion(self):
        """Test conversion of assistant TextContent to OpenAI string content."""
        # Create a text content message from assistant
        text_content = TextContent(type="text", text=self.sample_text)
        multipart = PromptMessageExtended(role="assistant", content=[text_content])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions - assistant should have string content in OpenAI
        self.assertEqual(openai_msg["role"], "assistant")
        self.assertEqual(openai_msg["content"], self.sample_text)

    def test_convert_prompt_message_to_openai_assistant(self):
        """Test conversion of a standard PromptMessage with assistant role to OpenAI format."""
        # Create a PromptMessage with TextContent
        text_content = TextContent(type="text", text=self.sample_text)
        prompt_message = PromptMessage(role="assistant", content=text_content)

        # Convert to OpenAI format
        openai_msg = OpenAIConverter.convert_prompt_message_to_openai(prompt_message)

        # Assertions - assistant should have string content in OpenAI
        self.assertEqual(openai_msg["role"], "assistant")
        self.assertEqual(openai_msg["content"], self.sample_text)

    def test_convert_prompt_message_to_openai_user_text(self):
        """Test conversion of a standard PromptMessage with user role and text content."""
        # Create a PromptMessage with TextContent
        text_content = TextContent(type="text", text="User message")
        prompt_message = PromptMessage(role="user", content=text_content)

        # Convert to OpenAI format
        openai_msg = OpenAIConverter.convert_prompt_message_to_openai(prompt_message)

        # Assertions - user should have array content in OpenAI
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(openai_msg["content"], "User message")

    def test_convert_prompt_message_to_openai_user_image(self):
        """Test conversion of a PromptMessage with image content to OpenAI format."""
        # Create a PromptMessage with ImageContent
        image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")
        image_content = ImageContent(type="image", data=image_base64, mimeType="image/jpeg")
        prompt_message = PromptMessage(role="user", content=image_content)

        # Convert to OpenAI format
        openai_msg = OpenAIConverter.convert_prompt_message_to_openai(prompt_message)

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertIsInstance(openai_msg["content"], list)
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "image_url")
        self.assertEqual(
            openai_msg["content"][0]["image_url"]["url"],
            f"data:image/jpeg;base64,{image_base64}",
        )

    def test_convert_prompt_message_embedded_resource_to_openai(self):
        """Test conversion of a PromptMessage with embedded resource to OpenAI format."""
        # Create a PromptMessage with embedded text resource
        text_resource = TextResourceContents(
            uri="test://example.com/document.txt",
            mimeType="text/plain",
            text="This is a text resource",
        )
        embedded_resource = EmbeddedResource(type="resource", resource=text_resource)
        prompt_message = PromptMessage(role="user", content=embedded_resource)

        # Convert to OpenAI format
        openai_msg = OpenAIConverter.convert_prompt_message_to_openai(prompt_message)

        # Assertions
        self.assertEqual(openai_msg["role"], "user")
        self.assertIsInstance(openai_msg["content"], list)
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertIn("<fastagent:file", openai_msg["content"][0]["text"])
        self.assertIn("This is a text resource", openai_msg["content"][0]["text"])

    def test_empty_assistant_message(self):
        """Test conversion of empty assistant message."""
        # Create an assistant message with empty content
        multipart = PromptMessageExtended(role="assistant", content=[])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions - should have empty content
        self.assertEqual(openai_msg["role"], "assistant")
        self.assertEqual(openai_msg["content"], "")


class TestOpenAIToolConverter(unittest.TestCase):
    """Test cases for conversion of tool results to OpenAI tool messages."""

    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a tool result"

    def test_tool_result_conversion(self):
        """Test conversion of CallToolResult to OpenAI tool message."""
        # Create a tool result with text content
        text_content = TextContent(type="text", text=self.sample_text)
        tool_result = CallToolResult(content=[text_content], isError=False)

        # Create a tool call ID
        tool_call_id = "call_abc123"

        # Convert directly to OpenAI tool message
        tool_message = OpenAIConverter.convert_tool_result_to_openai(
            tool_result=tool_result, tool_call_id=tool_call_id
        )

        # Assertions
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], tool_call_id)
        self.assertEqual(tool_message["content"], self.sample_text)

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

        # Create tool call IDs
        tool_call_id1 = "call_text_only"
        tool_call_id2 = "call_with_image"

        # Create a dict of tool_call_id -> result
        results = {tool_call_id1: text_result, tool_call_id2: image_result}

        # Convert to OpenAI tool messages
        tool_messages = OpenAIConverter.convert_function_results_to_openai(results)

        # Assertions
        self.assertEqual(len(tool_messages), 3)

        # Check first tool message (text only)
        self.assertEqual(tool_messages[0]["role"], "tool")
        self.assertEqual(tool_messages[0]["tool_call_id"], tool_call_id1)
        self.assertEqual(tool_messages[0]["content"], "Text-only result")

        # Check second tool message (with image)
        self.assertEqual(tool_messages[1]["role"], "tool")
        self.assertEqual(tool_messages[1]["tool_call_id"], tool_call_id2)
        self.assertEqual(tool_messages[1]["content"], "Here's the image:")
        self.assertEqual(tool_messages[2]["role"], "user")
        self.assertEqual(tool_messages[2]["content"][0]["type"], "image_url")

    def test_tool_result_with_mixed_content(self):
        """Test conversion of tool result with mixed content types."""
        # Create a tool result with text, image, and embedded resource
        text_content = TextContent(type="text", text="Here's the analysis:")

        # Add an image
        image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")
        image_content = ImageContent(type="image", data=image_base64, mimeType="image/jpeg")

        # Add a PDF file
        pdf_base64 = base64.b64encode(b"fake_pdf_data").decode("utf-8")
        pdf_resource = BlobResourceContents(
            uri="test://example.com/document.pdf",
            mimeType="application/pdf",
            blob=pdf_base64,
        )
        pdf_embedded = EmbeddedResource(type="resource", resource=pdf_resource)

        # Create the tool result with all content types
        tool_result = CallToolResult(
            content=[text_content, image_content, pdf_embedded], isError=False
        )

        # Create a tool call ID
        tool_call_id = "call_mixed_content"

        # Convert to OpenAI tool message
        tool_message = OpenAIConverter.convert_tool_result_to_openai(
            tool_result=tool_result, tool_call_id=tool_call_id
        )

        self.assertEqual(len(tool_message), 2)
        self.assertEqual(tool_message[0]["role"], "tool")
        self.assertEqual(tool_message[0]["tool_call_id"], tool_call_id)
        self.assertEqual(tool_message[0]["content"], "Here's the analysis:")

        self.assertEqual(tool_message[1][0]["role"], "user")
        self.assertEqual(tool_message[1][0]["content"][0]["type"], "image_url")

        self.assertEqual(tool_message[1][0]["content"][1]["type"], "file")
        self.assertEqual(
            tool_message[1][0]["content"][1]["file"]["file_data"],
            f"data:application/pdf;base64,{pdf_base64}",
        )

    def test_empty_schema_behavior(self):
        """Test adjustment of parameters for empty schema."""
        inputSchema = {
            "type": "object",
        }
        an_llm = llm_openai.OpenAILLM(Provider.OPENAI)
        adjusted = an_llm.adjust_schema(inputSchema)
        assert adjusted["properties"] == {}


class TestTextConcatenation(unittest.TestCase):
    """Test cases for concatenating adjacent text blocks."""

    def test_adjacent_text_blocks_concatenation(self):
        """Test that adjacent text blocks are concatenated when requested."""
        # Create multiple adjacent text blocks
        text1 = TextContent(type="text", text="First sentence.")
        text2 = TextContent(type="text", text="Second sentence.")
        text3 = TextContent(type="text", text="Third sentence.")

        multipart = PromptMessageExtended(role="user", content=[text1, text2, text3])

        # Convert with concatenation enabled
        openai_msgs = OpenAIConverter.convert_to_openai(multipart, concatenate_text_blocks=True)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions - should have combined all text blocks
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertEqual(
            openai_msg["content"][0]["text"],
            "First sentence. Second sentence. Third sentence.",
        )

    def test_mixed_content_with_concatenation(self):
        """Test concatenation with mixed content types."""
        # Create content with text blocks separated by an image
        text1 = TextContent(type="text", text="Text before image.")

        image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")
        image = ImageContent(type="image", data=image_base64, mimeType="image/jpeg")

        text2 = TextContent(type="text", text="Text after image.")
        text3 = TextContent(type="text", text="More text after image.")

        multipart = PromptMessageExtended(role="user", content=[text1, image, text2, text3])

        # Convert with concatenation enabled
        openai_msgs = OpenAIConverter.convert_to_openai(multipart, concatenate_text_blocks=True)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions - should have concatenated adjacent text blocks but kept them separate from image
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 3)

        # First block should be the first text
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertEqual(openai_msg["content"][0]["text"], "Text before image.")

        # Second block should be the image
        self.assertEqual(openai_msg["content"][1]["type"], "image_url")

        # Third block should be combined text2 and text3
        self.assertEqual(openai_msg["content"][2]["type"], "text")
        self.assertEqual(
            openai_msg["content"][2]["text"], "Text after image. More text after image."
        )

    def test_tool_result_with_concatenation(self):
        """Test that tool results can use text concatenation."""
        # Create a tool result with multiple text blocks
        text1 = TextContent(type="text", text="First part of result.")
        text2 = TextContent(type="text", text="Second part of result.")

        tool_result = CallToolResult(content=[text1, text2], isError=False)

        # Convert with concatenation enabled
        tool_message = OpenAIConverter.convert_tool_result_to_openai(
            tool_result=tool_result,
            tool_call_id="call_123",
            concatenate_text_blocks=True,
        )

        # Assertions - should have concatenated the text blocks
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_123")
        self.assertTrue(isinstance(tool_message["content"], str))
        self.assertEqual(tool_message["content"], "First part of result. Second part of result.")

    def test_convert_unsupported_binary_format(self):
        """Test handling of unsupported binary formats."""
        # Create a binary resource with an unsupported format
        binary_base64 = base64.b64encode(b"fake_binary_data").decode("utf-8")
        binary_resource = BlobResourceContents(
            uri="test://example.com/data.bin",
            mimeType="application/octet-stream",
            blob=binary_base64,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=binary_resource)
        multipart = PromptMessageExtended(role="user", content=[embedded_resource])

        # Convert to OpenAI format
        openai_msgs = OpenAIConverter.convert_to_openai(multipart)
        self.assertEqual(len(openai_msgs), 1)
        openai_msg = openai_msgs[0]

        # Assertions - should create a text message mentioning the resource
        self.assertEqual(openai_msg["role"], "user")
        self.assertEqual(len(openai_msg["content"]), 1)
        self.assertEqual(openai_msg["content"][0]["type"], "text")
        self.assertIn("Binary resource", openai_msg["content"][0]["text"])
        self.assertIn("data.bin", openai_msg["content"][0]["text"])
