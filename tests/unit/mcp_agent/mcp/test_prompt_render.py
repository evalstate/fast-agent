"""
Unit tests for the prompt rendering utilities.
"""

from mcp.types import (
    TextContent,
)

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from mcp_agent.mcp.prompt_render import render_multipart_message
from mcp_agent.mcp.resource_utils import (
    create_blob_resource,
    create_image_content,
    create_text_resource,
)


class TestPromptRender:
    """Tests for prompt rendering utilities."""

    def test_render_text_only_message(self):
        """Test rendering a message with only text content."""
        # Create a simple multipart message with text content
        message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Hello, world!"),
            ],
        )

        # Render the message
        result = render_multipart_message(message)

        # Check the rendered output
        assert result == "Hello, world!"

    def test_render_multiple_text_contents(self):
        """Test rendering a message with multiple text contents."""
        # Create a multipart message with multiple text contents
        message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Hello, world!"),
                TextContent(type="text", text="How are you today?"),
            ],
        )

        # Render the message
        result = render_multipart_message(message)

        # Check the rendered output (should join with newlines)
        assert result == "Hello, world!\nHow are you today?"

    def test_render_with_image_content(self):
        """Test rendering a message with image content."""
        # Create sample base64 data
        sample_base64 = (
            "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"  # 1x1 transparent GIF
        )

        # Create a multipart message with both text and image
        message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Look at this image:"),
                create_image_content(sample_base64, "image/png"),
            ],
        )

        # Render the message
        result = render_multipart_message(message)

        # Check the rendered output (should show image info)
        assert "Look at this image:" in result
        assert "[IMAGE: image/png" in result
        assert f"{len(sample_base64)} bytes" in result

    def test_render_with_embedded_text_resource(self):
        """Test rendering a message with embedded text resource."""
        # Create a multipart message with embedded text resource
        resource_text = "This is the content of the resource."
        message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Here's a text resource:"),
                create_text_resource("resource://test/sample.txt", resource_text, "text/plain"),
            ],
        )

        # Render the message
        result = render_multipart_message(message)

        # Check the rendered output (should include resource info and preview)
        assert "Here's a text resource:" in result
        assert "[EMBEDDED TEXT RESOURCE: text/plain" in result
        assert "resource://test/sample.txt" in result
        assert f"{len(resource_text)} chars" in result
        assert resource_text in result

    def test_render_with_long_embedded_text_resource(self):
        """Test rendering a message with a long embedded text resource (>300 chars)."""
        # Create a long text (over 300 characters)
        long_text = "A" * 400

        # Create a multipart message with embedded long text resource
        message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Here's a long text resource:"),
                create_text_resource("resource://test/long_sample.txt", long_text, "text/plain"),
            ],
        )

        # Render the message
        result = render_multipart_message(message)

        # Check the rendered output (should truncate with ellipsis)
        assert "Here's a long text resource:" in result
        assert "[EMBEDDED TEXT RESOURCE: text/plain" in result
        assert "resource://test/long_sample.txt" in result
        assert "400 chars" in result
        assert long_text[:300] in result
        assert "..." in result

        # Check that we are only displaying 300 characters plus ellipsis
        lines = result.splitlines()

        # Find the content line (should be the last line)
        text_section = lines[-1]
        assert len(text_section) == 303  # 300 chars + 3 for the ellipsis

    def test_render_with_blob_resource(self):
        """Test rendering a message with a binary blob resource."""
        # Create sample binary data (base64 encoded)
        sample_blob = "SGVsbG8sIHRoaXMgaXMgYSBiaW5hcnkgYmxvYiE="  # "Hello, this is a binary blob!"

        # Create a multipart message with a blob resource
        message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Here's a binary blob:"),
                create_blob_resource(
                    "resource://test/sample.bin", sample_blob, "application/octet-stream"
                ),
            ],
        )

        # Render the message
        result = render_multipart_message(message)

        # Check the rendered output (should show blob info)
        assert "Here's a binary blob:" in result
        assert "[EMBEDDED BLOB RESOURCE: application/octet-stream" in result
        assert "resource://test/sample.bin" in result
        assert f"{len(sample_blob)} bytes" in result
