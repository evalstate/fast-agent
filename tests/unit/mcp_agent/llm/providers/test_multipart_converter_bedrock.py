"""
Unit tests for the BedrockConverter class.
"""

import base64
import unittest
from typing import List

from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from mcp_agent.llm.providers.multipart_converter_bedrock import (
    BedrockConverter,
    ModelFamily,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.resource_utils import normalize_uri

# Sample data for tests
PDF_BASE64 = base64.b64encode(b"fake_pdf_data").decode("utf-8")


def create_pdf_resource(pdf_base64) -> EmbeddedResource:
    """Helper to create a PDF resource."""
    pdf_resource: BlobResourceContents = BlobResourceContents(
        uri="test://example.com/document.pdf",
        mimeType="application/pdf",
        blob=pdf_base64,
    )
    return EmbeddedResource(type="resource", resource=pdf_resource)


class TestBedrockModelFamilyDetection(unittest.TestCase):
    """Test cases for model family detection."""
    
    def test_detect_claude_model(self):
        """Test detection of Claude models."""
        # Test with standard Claude model ID
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        family = BedrockConverter.detect_model_family(model_id)
        self.assertEqual(family, ModelFamily.CLAUDE)
        
        # Test with non-standard ID containing 'claude'
        model_id = "custom.claude-model"
        family = BedrockConverter.detect_model_family(model_id)
        self.assertEqual(family, ModelFamily.CLAUDE)
    
    def test_detect_nova_model(self):
        """Test detection of Amazon Nova models."""
        # Test with standard Nova model ID
        model_id = "amazon.nova-text-v1"
        family = BedrockConverter.detect_model_family(model_id)
        self.assertEqual(family, ModelFamily.NOVA)
        
        # Test with non-standard ID containing 'nova'
        model_id = "custom.nova-model"
        family = BedrockConverter.detect_model_family(model_id)
        self.assertEqual(family, ModelFamily.NOVA)
    
    def test_detect_meta_model(self):
        """Test detection of Meta Llama models."""
        # Test with standard Meta model ID
        model_id = "meta.llama3-70b-instruct-v1:0"
        family = BedrockConverter.detect_model_family(model_id)
        self.assertEqual(family, ModelFamily.META)
        
        # Test with non-standard ID containing 'llama'
        model_id = "custom.llama-model"
        family = BedrockConverter.detect_model_family(model_id)
        self.assertEqual(family, ModelFamily.META)
    
    def test_unknown_model(self):
        """Test detection of unknown model families."""
        model_id = "unknown.model-v1"
        family = BedrockConverter.detect_model_family(model_id)
        self.assertEqual(family, ModelFamily.UNKNOWN)


class TestBedrockConverter(unittest.TestCase):
    """Test cases for conversion from MCP message types to Bedrock API format."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a test message"
        self.sample_image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")
    
    def test_claude_format_user_message(self):
        """Test conversion of user message to Claude format."""
        # Create a text content message
        text_content = TextContent(type="text", text=self.sample_text)
        multipart = PromptMessageMultipart(role="user", content=[text_content])
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Assertions
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 1)
        self.assertEqual(bedrock_msg["content"][0]["type"], "text")
        self.assertEqual(bedrock_msg["content"][0]["text"], self.sample_text)
    
    def test_claude_format_assistant_message(self):
        """Test conversion of assistant message to Claude format."""
        # Create a text content message
        text_content = TextContent(type="text", text=self.sample_text)
        multipart = PromptMessageMultipart(role="assistant", content=[text_content])
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Assertions
        self.assertEqual(bedrock_msg["role"], "assistant")
        self.assertEqual(len(bedrock_msg["content"]), 1)
        self.assertEqual(bedrock_msg["content"][0]["type"], "text")
        self.assertEqual(bedrock_msg["content"][0]["text"], self.sample_text)
    
    def test_claude_format_system_message(self):
        """Test conversion of system message to Claude format."""
        # Create a system message
        text_content = TextContent(type="text", text="You are a helpful assistant")
        multipart = PromptMessageMultipart(role="system", content=[text_content])
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Assertions - Claude uses a special format for system messages
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 1)
        self.assertEqual(bedrock_msg["content"][0]["type"], "text")
        self.assertIn("<admin>", bedrock_msg["content"][0]["text"])
        self.assertIn("You are a helpful assistant", bedrock_msg["content"][0]["text"])
        self.assertIn("</admin>", bedrock_msg["content"][0]["text"])
    
    def test_claude_format_image_content(self):
        """Test conversion of image content to Claude format."""
        # Create an image content message
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        multipart = PromptMessageMultipart(role="user", content=[image_content])
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Assertions
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 1)
        self.assertEqual(bedrock_msg["content"][0]["type"], "image")
        self.assertEqual(bedrock_msg["content"][0]["source"]["type"], "base64")
        self.assertEqual(bedrock_msg["content"][0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(bedrock_msg["content"][0]["source"]["data"], self.sample_image_base64)
    
    def test_claude_format_pdf_resource(self):
        """Test conversion of PDF resource to Claude format."""
        # Create a PDF resource
        pdf_resource = create_pdf_resource(PDF_BASE64)
        multipart = PromptMessageMultipart(role="user", content=[pdf_resource])
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Assertions - Bedrock doesn't support document blocks like direct Anthropic API
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 1)
        self.assertEqual(bedrock_msg["content"][0]["type"], "text")
        self.assertIn("[PDF document:", bedrock_msg["content"][0]["text"])
    
    def test_claude_format_empty_content(self):
        """Test conversion of empty content list to Claude format."""
        multipart = PromptMessageMultipart(role="user", content=[])
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Should have empty content list
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 0)
    
    def test_nova_format_user_message(self):
        """Test conversion of user message to Nova format."""
        # Create a text content message
        text_content = TextContent(type="text", text=self.sample_text)
        multipart = PromptMessageMultipart(role="user", content=[text_content])
        
        # Convert to Nova format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.NOVA)
        
        # Assertions - Nova has a different format
        self.assertEqual(bedrock_msg["text"], self.sample_text)
    
    def test_nova_format_system_message(self):
        """Test conversion of system message to Nova format."""
        # Create a system message
        text_content = TextContent(type="text", text="You are a helpful assistant")
        multipart = PromptMessageMultipart(role="system", content=[text_content])
        
        # Convert to Nova format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.NOVA)
        
        # Assertions - Nova uses a different format for system messages
        self.assertIn("<<SYS>>", bedrock_msg["text"])
        self.assertIn("You are a helpful assistant", bedrock_msg["text"])
        self.assertIn("<</SYS>>", bedrock_msg["text"])
    
    def test_meta_format_messages(self):
        """Test conversion of messages to Meta Llama format."""
        # Create messages with different roles
        user_content = TextContent(type="text", text="Hello AI")
        user_msg = PromptMessageMultipart(role="user", content=[user_content])
        
        assistant_content = TextContent(type="text", text="Hello human")
        assistant_msg = PromptMessageMultipart(role="assistant", content=[assistant_content])
        
        system_content = TextContent(type="text", text="You are an AI assistant")
        system_msg = PromptMessageMultipart(role="system", content=[system_content])
        
        # Convert to Meta format
        user_bedrock_msg = BedrockConverter.convert_to_bedrock(user_msg, ModelFamily.META)
        assistant_bedrock_msg = BedrockConverter.convert_to_bedrock(assistant_msg, ModelFamily.META)
        system_bedrock_msg = BedrockConverter.convert_to_bedrock(system_msg, ModelFamily.META)
        
        # Assertions - Meta format has role and content fields
        self.assertEqual(user_bedrock_msg["role"], "user")
        self.assertEqual(user_bedrock_msg["content"], "Hello AI")
        
        self.assertEqual(assistant_bedrock_msg["role"], "assistant")
        self.assertEqual(assistant_bedrock_msg["content"], "Hello human")
        
        self.assertEqual(system_bedrock_msg["role"], "system")
        self.assertEqual(system_bedrock_msg["content"], "You are an AI assistant")
    
    def test_supported_image_types(self):
        """Test detection of supported image types."""
        # Supported MIME types
        self.assertTrue(BedrockConverter._is_supported_image_type("image/jpeg"))
        self.assertTrue(BedrockConverter._is_supported_image_type("image/png"))
        self.assertTrue(BedrockConverter._is_supported_image_type("image/gif"))
        self.assertTrue(BedrockConverter._is_supported_image_type("image/webp"))
        
        # Unsupported MIME types
        self.assertFalse(BedrockConverter._is_supported_image_type("image/bmp"))
        self.assertFalse(BedrockConverter._is_supported_image_type("image/svg+xml"))
        self.assertFalse(BedrockConverter._is_supported_image_type("image/tiff"))
    
    def test_fallback_for_unsupported_image(self):
        """Test fallback for unsupported image formats."""
        # Create an image with unsupported MIME type
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/bmp"
        )
        multipart = PromptMessageMultipart(role="user", content=[image_content])
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Assertions - should create a text fallback
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 1)
        self.assertEqual(bedrock_msg["content"][0]["type"], "text")
        self.assertIn("unsupported format 'image/bmp'", bedrock_msg["content"][0]["text"])
    
    def test_svg_resource_conversion(self):
        """Test conversion of SVG resource."""
        # Create an SVG resource
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'
        svg_resource = TextResourceContents(
            uri="test://example.com/image.svg",
            mimeType="image/svg+xml",
            text=svg_content,
        )
        embedded_resource = EmbeddedResource(type="resource", resource=svg_resource)
        multipart = PromptMessageMultipart(role="user", content=[embedded_resource])
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Assertions - should create a text block with XML code formatting
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 1)
        self.assertEqual(bedrock_msg["content"][0]["type"], "text")
        self.assertIn("```xml", bedrock_msg["content"][0]["text"])
        self.assertIn(svg_content, bedrock_msg["content"][0]["text"])
    
    def test_multiple_content_blocks(self):
        """Test conversion of multiple content blocks."""
        # Create multiple content blocks
        text_content1 = TextContent(type="text", text="First text")
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        text_content2 = TextContent(type="text", text="Second text")
        
        multipart = PromptMessageMultipart(
            role="user", content=[text_content1, image_content, text_content2]
        )
        
        # Convert to Claude format
        bedrock_msg = BedrockConverter.convert_to_bedrock(multipart, ModelFamily.CLAUDE)
        
        # Assertions
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 3)
        self.assertEqual(bedrock_msg["content"][0]["type"], "text")
        self.assertEqual(bedrock_msg["content"][0]["text"], "First text")
        self.assertEqual(bedrock_msg["content"][1]["type"], "image")
        self.assertEqual(bedrock_msg["content"][2]["type"], "text")
        self.assertEqual(bedrock_msg["content"][2]["text"], "Second text")


class TestBedrockToolConverter(unittest.TestCase):
    """Test cases for conversion of tool results to Bedrock API format."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_text = "This is a tool result"
        self.sample_image_base64 = base64.b64encode(b"fake_image_data").decode("utf-8")
        self.tool_use_id = "toolu_01D7FLrfh4GYq7yT1ULFeyMV"
    
    def test_text_tool_result_conversion(self):
        """Test conversion of text tool result to Bedrock format."""
        # Create a tool result with text content
        text_content = TextContent(type="text", text=self.sample_text)
        tool_result = CallToolResult(content=[text_content], isError=False)
        
        # Convert to Bedrock format
        bedrock_block = BedrockConverter.convert_tool_result_to_bedrock(
            tool_result, self.tool_use_id, ModelFamily.CLAUDE
        )
        
        # Assertions
        self.assertEqual(bedrock_block["type"], "tool_result")
        self.assertEqual(bedrock_block["tool_use_id"], self.tool_use_id)
        self.assertEqual(bedrock_block["is_error"], False)
        self.assertEqual(len(bedrock_block["content"]), 1)
        self.assertEqual(bedrock_block["content"][0]["type"], "text")
        self.assertEqual(bedrock_block["content"][0]["text"], self.sample_text)
    
    def test_image_tool_result_conversion(self):
        """Test conversion of image tool result to Bedrock format."""
        # Create a tool result with image content
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        tool_result = CallToolResult(content=[image_content], isError=False)
        
        # Convert to Bedrock format
        bedrock_block = BedrockConverter.convert_tool_result_to_bedrock(
            tool_result, self.tool_use_id, ModelFamily.CLAUDE
        )
        
        # Assertions
        self.assertEqual(bedrock_block["type"], "tool_result")
        self.assertEqual(bedrock_block["tool_use_id"], self.tool_use_id)
        self.assertEqual(bedrock_block["is_error"], False)
        self.assertEqual(len(bedrock_block["content"]), 1)
        self.assertEqual(bedrock_block["content"][0]["type"], "image")
        self.assertEqual(bedrock_block["content"][0]["source"]["type"], "base64")
        self.assertEqual(bedrock_block["content"][0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(bedrock_block["content"][0]["source"]["data"], self.sample_image_base64)
    
    def test_mixed_tool_result_conversion(self):
        """Test conversion of mixed content tool result to Bedrock format."""
        # Create a tool result with text and image content
        text_content = TextContent(type="text", text=self.sample_text)
        image_content = ImageContent(
            type="image", data=self.sample_image_base64, mimeType="image/jpeg"
        )
        tool_result = CallToolResult(content=[text_content, image_content], isError=False)
        
        # Convert to Bedrock format
        bedrock_block = BedrockConverter.convert_tool_result_to_bedrock(
            tool_result, self.tool_use_id, ModelFamily.CLAUDE
        )
        
        # Assertions
        self.assertEqual(bedrock_block["type"], "tool_result")
        self.assertEqual(bedrock_block["tool_use_id"], self.tool_use_id)
        self.assertEqual(len(bedrock_block["content"]), 2)
        self.assertEqual(bedrock_block["content"][0]["type"], "text")
        self.assertEqual(bedrock_block["content"][0]["text"], self.sample_text)
        self.assertEqual(bedrock_block["content"][1]["type"], "image")
    
    def test_error_tool_result_conversion(self):
        """Test conversion of error tool result to Bedrock format."""
        # Create a tool result with error flag set
        text_content = TextContent(type="text", text="Error: Something went wrong")
        tool_result = CallToolResult(content=[text_content], isError=True)
        
        # Convert to Bedrock format
        bedrock_block = BedrockConverter.convert_tool_result_to_bedrock(
            tool_result, self.tool_use_id, ModelFamily.CLAUDE
        )
        
        # Assertions
        self.assertEqual(bedrock_block["type"], "tool_result")
        self.assertEqual(bedrock_block["tool_use_id"], self.tool_use_id)
        self.assertEqual(bedrock_block["is_error"], True)
        self.assertEqual(len(bedrock_block["content"]), 1)
        self.assertEqual(bedrock_block["content"][0]["type"], "text")
        self.assertEqual(bedrock_block["content"][0]["text"], "Error: Something went wrong")
    
    def test_empty_tool_result_conversion(self):
        """Test conversion of empty tool result to Bedrock format."""
        # Create a tool result with no content
        tool_result = CallToolResult(content=[], isError=False)
        
        # Convert to Bedrock format
        bedrock_block = BedrockConverter.convert_tool_result_to_bedrock(
            tool_result, self.tool_use_id, ModelFamily.CLAUDE
        )
        
        # Should have a placeholder text block
        self.assertEqual(bedrock_block["type"], "tool_result")
        self.assertEqual(len(bedrock_block["content"]), 1)
        self.assertEqual(bedrock_block["content"][0]["type"], "text")
        self.assertEqual(bedrock_block["content"][0]["text"], "[No content in tool result]")
    
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
        
        # Convert to Bedrock message
        bedrock_msg = BedrockConverter.create_tool_results_message(
            tool_results, ModelFamily.CLAUDE
        )
        
        # Assertions
        self.assertEqual(bedrock_msg["role"], "user")
        self.assertEqual(len(bedrock_msg["content"]), 2)
        
        # Check first tool result
        self.assertEqual(bedrock_msg["content"][0]["type"], "tool_result")
        self.assertEqual(bedrock_msg["content"][0]["tool_use_id"], tool_use_id1)
        self.assertEqual(bedrock_msg["content"][0]["content"][0]["type"], "text")
        self.assertEqual(bedrock_msg["content"][0]["content"][0]["text"], self.sample_text)
        
        # Check second tool result
        self.assertEqual(bedrock_msg["content"][1]["type"], "tool_result")
        self.assertEqual(bedrock_msg["content"][1]["tool_use_id"], tool_use_id2)
        self.assertEqual(bedrock_msg["content"][1]["content"][0]["type"], "image")


def create_text_resource(
    text: str, filename_or_uri: str, mime_type: str = None
) -> TextResourceContents:
    """
    Helper function to create a TextResourceContents with proper URI handling.
    """
    # Normalize the URI
    uri = normalize_uri(filename_or_uri)
    
    return TextResourceContents(uri=uri, mimeType=mime_type, text=text)