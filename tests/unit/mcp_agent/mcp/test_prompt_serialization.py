"""
Tests for serializing PromptMessageExtended objects to delimited format.
"""

from mcp.types import EmbeddedResource, ImageContent, TextContent, TextResourceContents

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from mcp_agent.mcp.prompt_serialization import (
    json_to_extended_messages,
    multipart_messages_to_delimited_format,
    multipart_messages_to_json,
)


class TestPromptSerialization:
    """Tests for prompt serialization and delimited format conversion."""

    def test_json_serialization_and_deserialization(self):
        """Test the new JSON serialization and deserialization approach."""
        # Create multipart messages with various content types
        original_messages = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="Here's a resource:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://data.json",
                            mimeType="application/json",
                            text='{"key": "value"}',
                        ),
                    ),
                ],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[
                    TextContent(type="text", text="I've processed your resource."),
                    ImageContent(type="image", data="base64EncodedImage", mimeType="image/jpeg"),
                ],
            ),
        ]

        # Convert to JSON
        json_str = multipart_messages_to_json(original_messages)

        # Verify JSON contains expected elements
        assert "user" in json_str
        assert "assistant" in json_str
        assert "resource://data.json" in json_str
        assert "application/json" in json_str
        assert "base64EncodedImage" in json_str
        assert "image/jpeg" in json_str

        # Convert back from JSON
        parsed_messages = json_to_extended_messages(json_str)

        # Verify round-trip conversion
        assert len(parsed_messages) == len(original_messages)
        assert parsed_messages[0].role == original_messages[0].role
        assert parsed_messages[1].role == original_messages[1].role

        # Check first message
        assert len(parsed_messages[0].content) == 2
        assert parsed_messages[0].content[0].type == "text"
        assert parsed_messages[0].content[0].text == "Here's a resource:"
        assert parsed_messages[0].content[1].type == "resource"
        assert str(parsed_messages[0].content[1].resource.uri) == "resource://data.json"
        assert parsed_messages[0].content[1].resource.mimeType == "application/json"
        assert parsed_messages[0].content[1].resource.text == '{"key": "value"}'

        # Check second message
        assert len(parsed_messages[1].content) == 2
        assert parsed_messages[1].content[0].type == "text"
        assert parsed_messages[1].content[0].text == "I've processed your resource."
        assert parsed_messages[1].content[1].type == "image"
        assert parsed_messages[1].content[1].data == "base64EncodedImage"
        assert parsed_messages[1].content[1].mimeType == "image/jpeg"

    def test_multipart_to_delimited_format(self):
        """Test converting PromptMessageExtended to delimited format for saving."""
        # Create multipart messages
        multipart_messages = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="Hello!"),
                    TextContent(type="text", text="Can you help me?"),
                ],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[
                    TextContent(type="text", text="Sure, I'd be happy to help."),
                    TextContent(type="text", text="What do you need assistance with?"),
                ],
            ),
        ]

        # Convert to delimited format
        delimited_content = multipart_messages_to_delimited_format(multipart_messages)

        # Verify results
        assert len(delimited_content) == 4
        assert delimited_content[0] == "---USER"
        assert delimited_content[1] == "Hello!\n\nCan you help me?"
        assert delimited_content[2] == "---ASSISTANT"
        assert (
            delimited_content[3]
            == "Sure, I'd be happy to help.\n\nWhat do you need assistance with?"
        )

    def test_multipart_with_resources_to_delimited_format(self):
        """Test converting PromptMessageExtended with resources to delimited format."""
        # Create multipart messages with resources
        multipart_messages = [
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="Check this code:"),
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri="resource://example.py",
                            mimeType="text/x-python",
                            text="def hello():\n    print('Hello, world!')",
                        ),
                    ),
                ],
            ),
        ]

        # Convert to delimited format
        delimited_content = multipart_messages_to_delimited_format(multipart_messages)

        # Verify results
        assert len(delimited_content) == 4
        assert delimited_content[0] == "---USER"
        assert "Check this code:" in delimited_content[1]
        assert delimited_content[2] == "---RESOURCE"

        # Resource is now in JSON format
        resource_json = delimited_content[3]
        assert "type" in resource_json
        assert "resource" in resource_json
        assert "uri" in resource_json.lower()
        assert "example.py" in resource_json
        assert "def hello()" in resource_json

    def test_multi_role_messages_to_delimited_format(self):
        """Test converting a list of PromptMessageExtended objects with different roles to delimited format."""
        # Create multipart messages with different roles
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="You are a helpful assistant.")],
            ),
            PromptMessageExtended(
                role="user",
                content=[
                    TextContent(type="text", text="Hello!"),
                    TextContent(type="text", text="Can you help me?"),
                ],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[
                    TextContent(type="text", text="I'd be happy to help."),
                    TextContent(type="text", text="What can I assist you with today?"),
                ],
            ),
        ]

        # Convert to delimited format
        delimited = multipart_messages_to_delimited_format(messages)

        # Verify results
        assert len(delimited) == 6  # 3 delimiters + 3 content blocks
        assert delimited[0] == "---USER"
        assert delimited[1] == "You are a helpful assistant."
        assert delimited[2] == "---USER"
        assert delimited[3] == "Hello!\n\nCan you help me?"
        assert delimited[4] == "---ASSISTANT"
        assert delimited[5] == "I'd be happy to help.\n\nWhat can I assist you with today?"
