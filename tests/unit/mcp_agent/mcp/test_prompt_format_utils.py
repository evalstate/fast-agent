"""
Tests for the prompt_format_utils module, focusing on resource handling.
"""

import os
import tempfile
from pathlib import Path

import pytest
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from mcp_agent.mcp.prompt_serialization import (
    delimited_format_to_extended_messages,
    load_messages_from_delimited_file,
    multipart_messages_to_delimited_format,
    save_messages_to_delimited_file,
)


class TestPromptFormatUtils:
    """Tests for the prompt_format_utils module."""

    def test_multipart_with_resources_to_delimited(self):
        """Test converting multipart messages with resources to delimited format."""
        # Create messages with resources
        messages = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="Here's a code sample:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://code.py",
                            mimeType="text/x-python",
                            text='print("Hello, World!")',
                        ),
                    ),
                ],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[
                    TextContent(
                        type="text",
                        text="I've analyzed your code and made improvements:",
                    ),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://improved_code.py",
                            mimeType="text/x-python",
                            text='def main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()',
                        ),
                    ),
                ],
            ),
        ]

        # Convert to delimited format
        delimited = multipart_messages_to_delimited_format(
            messages,
            user_delimiter="---USER",
            assistant_delimiter="---ASSISTANT",
            resource_delimiter="---RESOURCE",
        )

        # Verify structure
        assert len(delimited) == 8  # 2 role delimiters + 2 text blocks + 4 resource-related entries

        # First message (user)
        assert delimited[0] == "---USER"
        assert "Here's a code sample:" in delimited[1]
        assert delimited[2] == "---RESOURCE"

        # User resource in JSON format
        user_resource_json = delimited[3]
        assert "type" in user_resource_json
        assert "resource" in user_resource_json
        assert "code.py" in user_resource_json
        assert "print" in user_resource_json

        # Second message (assistant)
        assert delimited[4] == "---ASSISTANT"
        assert "I've analyzed your code" in delimited[5]
        assert delimited[6] == "---RESOURCE"

        # Assistant resource in JSON format
        assistant_resource_json = delimited[7]
        assert "type" in assistant_resource_json
        assert "resource" in assistant_resource_json
        assert "improved_code.py" in assistant_resource_json
        assert "def main()" in assistant_resource_json

    def test_delimited_with_resources_to_extended(self):
        """Test converting delimited format with resources to multipart messages."""
        # Create delimited content with resources in JSON format
        delimited_content = """---USER
Here's a CSS file I want to improve:

---RESOURCE
{
  "type": "resource",
  "resource": {
    "uri": "resource://styles.css",
    "mimeType": "text/css",
    "text": "body { color: black; }"
  }
}

---ASSISTANT
I've reviewed your CSS and made it more efficient:

---RESOURCE
{
  "type": "resource",
  "resource": {
    "uri": "resource://improved_styles.css",
    "mimeType": "text/css",
    "text": "body { color: #000; }"
  }
}"""

        # Convert to multipart messages
        messages = delimited_format_to_extended_messages(
            delimited_content, resource_delimiter="---RESOURCE"
        )

        # Verify structure
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert len(messages[0].content) == 2  # Text and resource
        assert messages[0].content[0].type == "text"
        assert "Here's a CSS file" in messages[0].content[0].text
        assert messages[0].content[1].type == "resource"
        assert str(messages[0].content[1].resource.uri) == "resource://styles.css"
        assert messages[0].content[1].resource.mimeType == "text/css"
        assert messages[0].content[1].resource.text == "body { color: black; }"

        assert messages[1].role == "assistant"
        assert len(messages[1].content) == 2  # Text and resource
        assert messages[1].content[0].type == "text"
        assert "I've reviewed your CSS" in messages[1].content[0].text
        assert messages[1].content[1].type == "resource"
        assert str(messages[1].content[1].resource.uri) == "resource://improved_styles.css"
        assert messages[1].content[1].resource.mimeType == "text/css"
        assert messages[1].content[1].resource.text == "body { color: #000; }"

    def test_multiple_resources_in_one_message(self):
        """Test handling multiple resources in a single message."""
        # Create a message with multiple resources
        message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="I need to analyze these files:"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="resource://data1.csv",
                        mimeType="text/csv",
                        text="id,name,value\n1,A,10\n2,B,20",
                    ),
                ),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="resource://data2.csv",
                        mimeType="text/csv",
                        text="id,name,value\n3,C,30\n4,D,40",
                    ),
                ),
            ],
        )

        # Convert to delimited format
        delimited = multipart_messages_to_delimited_format([message])

        # Verify structure - should have user delimiter, text, and two resource JSON blocks
        assert len(delimited) == 6
        assert delimited[0] == "---USER"
        assert "I need to analyze these files:" in delimited[1]
        assert delimited[2] == "---RESOURCE"

        # First resource JSON
        first_resource_json = delimited[3]
        assert "type" in first_resource_json
        assert "resource" in first_resource_json
        assert "data1.csv" in first_resource_json
        assert "text/csv" in first_resource_json
        assert "id,name,value" in first_resource_json

        assert delimited[4] == "---RESOURCE"

        # Second resource JSON
        second_resource_json = delimited[5]
        assert "type" in second_resource_json
        assert "resource" in second_resource_json
        assert "data2.csv" in second_resource_json
        assert "text/csv" in second_resource_json
        assert "id,name,value" in second_resource_json

        # Convert back to multipart
        messages = delimited_format_to_extended_messages("\n".join(delimited))

        # Verify round-trip conversion
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert len(messages[0].content) == 3  # Text and two resources
        assert messages[0].content[0].type == "text"
        assert messages[0].content[1].type == "resource"
        assert messages[0].content[2].type == "resource"

        # Verify resource content is preserved
        assert str(messages[0].content[1].resource.uri) == "resource://data1.csv"
        assert messages[0].content[1].resource.mimeType == "text/csv"
        assert "id,name,value" in messages[0].content[1].resource.text

        assert str(messages[0].content[2].resource.uri) == "resource://data2.csv"
        assert messages[0].content[2].resource.mimeType == "text/csv"
        assert "id,name,value" in messages[0].content[2].resource.text

    def test_image_handling(self):
        """Test handling image content in multipart messages."""
        # Create a message with an image
        message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(type="text", text="Look at this image:"),
                ImageContent(type="image", data="base64EncodedImageData", mimeType="image/png"),
            ],
        )

        # Convert to delimited format
        delimited = multipart_messages_to_delimited_format([message])

        # In the new implementation, images are serialized as JSON
        assert len(delimited) == 4
        assert delimited[0] == "---USER"
        assert "Look at this image:" in delimited[1]
        assert delimited[2] == "---RESOURCE"

        # Image JSON contains the image data
        image_json = delimited[3]
        assert "type" in image_json
        assert "image" in image_json
        assert "data" in image_json
        assert "base64EncodedImageData" in image_json
        assert "mimeType" in image_json
        assert "image/png" in image_json

    @pytest.fixture
    def temp_resource_file(self):
        """Create a temporary file for testing resource handling."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
            tf.write("""---USER
Here's a file with resources:

---RESOURCE
file1.js

---RESOURCE
file2.css

---ASSISTANT
I've analyzed both files.

---RESOURCE
analysis.md""")
            tf_path = Path(tf.name)

        yield tf_path

        # Cleanup
        os.unlink(tf_path)

    def test_save_and_load_with_resources(self, temp_resource_file):
        """Test saving and loading multipart messages with resources."""
        # Create messages with resources
        messages = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="Check this JSON file:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://config.json",
                            mimeType="application/json",
                            text='{"key": "value"}',
                        ),
                    ),
                ],
            )
        ]

        # Save to file
        save_messages_to_delimited_file(messages, str(temp_resource_file))

        # Load from file
        loaded_messages = load_messages_from_delimited_file(str(temp_resource_file))

        # Verify structure
        assert len(loaded_messages) == 1
        assert loaded_messages[0].role == "user"
        assert len(loaded_messages[0].content) == 2  # Text and resource
        assert loaded_messages[0].content[0].type == "text"
        assert loaded_messages[0].content[1].type == "resource"
        assert str(loaded_messages[0].content[1].resource.uri) == "resource://config.json"

    def test_round_trip_with_mime_types(self):
        """Test round-trip conversion preserving MIME type information."""
        # Original message with different MIME types
        original_messages = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="Here are some files:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://script.js",
                            mimeType="application/javascript",
                            text="function hello() { return 'Hello!'; }",
                        ),
                    ),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://style.css",
                            mimeType="text/css",
                            text="body { color: blue; }",
                        ),
                    ),
                ],
            )
        ]

        # Convert to delimited format
        delimited_content = multipart_messages_to_delimited_format(original_messages)
        delimited_text = "\n".join(delimited_content)

        # Convert back to multipart
        result_messages = delimited_format_to_extended_messages(delimited_text)

        # Verify structure
        assert len(result_messages) == 1
        assert result_messages[0].role == "user"
        assert len(result_messages[0].content) == 3  # Text and two resources

        # The resource URIs should be preserved
        resources = [
            content for content in result_messages[0].content if content.type == "resource"
        ]
        assert len(resources) == 2

        # Resource URIs should be preserved
        resource_uris = [str(resource.resource.uri) for resource in resources]
        assert "resource://script.js" in resource_uris
        assert "resource://style.css" in resource_uris
