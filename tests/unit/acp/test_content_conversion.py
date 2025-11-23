"""
Unit tests for ACP content block conversion to MCP format.
"""

from __future__ import annotations

import base64

from acp.schema import (
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    TextContentBlock,
    TextResourceContents,
)
from mcp.types import (
    BlobResourceContents as MCPBlobResourceContents,
)
from mcp.types import (
    EmbeddedResource as MCPEmbeddedResource,
)
from mcp.types import (
    ImageContent as MCPImageContent,
)
from mcp.types import (
    TextContent as MCPTextContent,
)
from mcp.types import (
    TextResourceContents as MCPTextResourceContents,
)

from fast_agent.acp.content_conversion import (
    convert_acp_content_to_mcp,
    convert_acp_prompt_to_mcp_content_blocks,
)


class TestTextContentConversion:
    """Test conversion of TextContentBlock."""

    def test_basic_text_conversion(self):
        """Test basic text content conversion."""
        acp_text = TextContentBlock(
            type="text",
            text="Hello, world!",
        )

        mcp_content = convert_acp_content_to_mcp(acp_text)

        assert isinstance(mcp_content, MCPTextContent)
        assert mcp_content.type == "text"
        assert mcp_content.text == "Hello, world!"
        assert mcp_content.annotations is None

    def test_text_with_annotations(self):
        """Test text content with annotations."""
        # Note: annotations are optional and may not be present
        acp_text = TextContentBlock(
            type="text",
            text="Important message",
        )

        mcp_content = convert_acp_content_to_mcp(acp_text)

        assert isinstance(mcp_content, MCPTextContent)
        assert mcp_content.text == "Important message"


class TestImageContentConversion:
    """Test conversion of ImageContentBlock."""

    def test_basic_image_conversion(self):
        """Test basic image content conversion."""
        # Create a simple base64 encoded image data
        image_data = base64.b64encode(b"fake-image-data").decode("utf-8")

        acp_image = ImageContentBlock(
            type="image",
            data=image_data,
            mimeType="image/png",
        )

        mcp_content = convert_acp_content_to_mcp(acp_image)

        assert isinstance(mcp_content, MCPImageContent)
        assert mcp_content.type == "image"
        assert mcp_content.data == image_data
        assert mcp_content.mimeType == "image/png"

    def test_image_with_uri(self):
        """Test image content with URI (should be preserved)."""
        image_data = base64.b64encode(b"fake-image-data").decode("utf-8")

        acp_image = ImageContentBlock(
            type="image",
            data=image_data,
            mimeType="image/jpeg",
            uri="file:///path/to/image.jpg",
        )

        mcp_content = convert_acp_content_to_mcp(acp_image)

        assert isinstance(mcp_content, MCPImageContent)
        assert mcp_content.data == image_data
        assert mcp_content.mimeType == "image/jpeg"


class TestEmbeddedResourceConversion:
    """Test conversion of EmbeddedResourceContentBlock."""

    def test_text_resource_conversion(self):
        """Test conversion of embedded text resource."""
        acp_resource = EmbeddedResourceContentBlock(
            type="resource",
            resource=TextResourceContents(
                uri="file:///path/to/file.py",
                mimeType="text/x-python",
                text="def hello():\n    print('Hello')",
            ),
        )

        mcp_content = convert_acp_content_to_mcp(acp_resource)

        assert isinstance(mcp_content, MCPEmbeddedResource)
        assert mcp_content.type == "resource"
        assert isinstance(mcp_content.resource, MCPTextResourceContents)
        assert str(mcp_content.resource.uri) == "file:///path/to/file.py"
        assert mcp_content.resource.mimeType == "text/x-python"
        assert mcp_content.resource.text == "def hello():\n    print('Hello')"

    def test_blob_resource_conversion(self):
        """Test conversion of embedded blob resource."""
        blob_data = base64.b64encode(b"fake-binary-data").decode("utf-8")

        acp_resource = EmbeddedResourceContentBlock(
            type="resource",
            resource=BlobResourceContents(
                uri="file:///path/to/file.pdf",
                mimeType="application/pdf",
                blob=blob_data,
            ),
        )

        mcp_content = convert_acp_content_to_mcp(acp_resource)

        assert isinstance(mcp_content, MCPEmbeddedResource)
        assert mcp_content.type == "resource"
        assert isinstance(mcp_content.resource, MCPBlobResourceContents)
        assert str(mcp_content.resource.uri) == "file:///path/to/file.pdf"
        assert mcp_content.resource.mimeType == "application/pdf"
        assert mcp_content.resource.blob == blob_data

    def test_text_resource_without_mimetype(self):
        """Test text resource without MIME type."""
        acp_resource = EmbeddedResourceContentBlock(
            type="resource",
            resource=TextResourceContents(
                uri="file:///path/to/file.txt",
                text="Hello, world!",
            ),
        )

        mcp_content = convert_acp_content_to_mcp(acp_resource)

        assert isinstance(mcp_content, MCPEmbeddedResource)
        assert isinstance(mcp_content.resource, MCPTextResourceContents)
        assert str(mcp_content.resource.uri) == "file:///path/to/file.txt"
        assert mcp_content.resource.mimeType is None
        assert mcp_content.resource.text == "Hello, world!"


class TestPromptConversion:
    """Test conversion of complete ACP prompts to MCP content blocks."""

    def test_mixed_content_prompt(self):
        """Test conversion of prompt with mixed content types."""
        image_data = base64.b64encode(b"fake-image").decode("utf-8")

        acp_prompt = [
            TextContentBlock(type="text", text="Please analyze this code:"),
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///main.py",
                    mimeType="text/x-python",
                    text="print('hello')",
                ),
            ),
            TextContentBlock(type="text", text="And this screenshot:"),
            ImageContentBlock(
                type="image",
                data=image_data,
                mimeType="image/png",
            ),
        ]

        mcp_blocks = convert_acp_prompt_to_mcp_content_blocks(acp_prompt)

        assert len(mcp_blocks) == 4
        assert isinstance(mcp_blocks[0], MCPTextContent)
        assert mcp_blocks[0].text == "Please analyze this code:"
        assert isinstance(mcp_blocks[1], MCPEmbeddedResource)
        assert str(mcp_blocks[1].resource.uri) == "file:///main.py"
        assert isinstance(mcp_blocks[2], MCPTextContent)
        assert mcp_blocks[2].text == "And this screenshot:"
        assert isinstance(mcp_blocks[3], MCPImageContent)
        assert mcp_blocks[3].data == image_data

    def test_empty_prompt(self):
        """Test conversion of empty prompt."""
        acp_prompt = []
        mcp_blocks = convert_acp_prompt_to_mcp_content_blocks(acp_prompt)
        assert mcp_blocks == []

    def test_text_only_prompt(self):
        """Test conversion of text-only prompt."""
        acp_prompt = [
            TextContentBlock(type="text", text="First message"),
            TextContentBlock(type="text", text="Second message"),
        ]

        mcp_blocks = convert_acp_prompt_to_mcp_content_blocks(acp_prompt)

        assert len(mcp_blocks) == 2
        assert all(isinstance(block, MCPTextContent) for block in mcp_blocks)
        assert mcp_blocks[0].text == "First message"
        assert mcp_blocks[1].text == "Second message"


class TestUnsupportedContent:
    """Test handling of unsupported content types."""

    def test_unsupported_content_returns_none(self):
        """Test that unsupported content types return None."""

        # Create a mock unsupported content type
        class UnsupportedContent:
            type = "audio"
            data = "base64-audio-data"

        result = convert_acp_content_to_mcp(UnsupportedContent())
        assert result is None

    def test_prompt_with_unsupported_content_skips_it(self):
        """Test that unsupported content is skipped in prompt conversion."""

        class UnsupportedContent:
            type = "audio"

        acp_prompt = [
            TextContentBlock(type="text", text="Hello"),
            UnsupportedContent(),
            TextContentBlock(type="text", text="World"),
        ]

        mcp_blocks = convert_acp_prompt_to_mcp_content_blocks(acp_prompt)

        # Should only have the two text blocks
        assert len(mcp_blocks) == 2
        assert all(isinstance(block, MCPTextContent) for block in mcp_blocks)
        assert mcp_blocks[0].text == "Hello"
        assert mcp_blocks[1].text == "World"
