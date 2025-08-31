"""
Unit tests for the PromptMessageMultipart class.
"""

from mcp.types import (
    GetPromptResult,
    ImageContent,
    PromptMessage,
    TextContent,
)

from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class TestPromptMessageMultipart:
    """Tests for the PromptMessageMultipart class."""

    def test_from_prompt_messages_with_single_role(self):
        """Test converting a sequence of PromptMessages with the same role."""
        # Create test messages
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="user", content=TextContent(type="text", text="How are you?")),
        ]

        # Convert to PromptMessageMultipart
        result = PromptMessageMultipart.to_multipart(messages)

        # Verify results
        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].content) == 2
        assert result[0].content[0].text == "Hello"
        assert result[0].content[1].text == "How are you?"

    def test_from_prompt_messages_with_multiple_roles(self):
        """Test converting a sequence of PromptMessages with different roles."""
        # Create test messages with alternating roles
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="assistant", content=TextContent(type="text", text="Hi there!")),
            PromptMessage(role="user", content=TextContent(type="text", text="How are you?")),
        ]

        # Convert to PromptMessageMultipart
        result = PromptMessageMultipart.to_multipart(messages)

        # Verify results
        assert len(result) == 3
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[2].role == "user"
        assert len(result[0].content) == 1
        assert len(result[1].content) == 1
        assert len(result[2].content) == 1
        assert result[0].content[0].text == "Hello"
        assert result[1].content[0].text == "Hi there!"
        assert result[2].content[0].text == "How are you?"

    def test_from_prompt_messages_with_mixed_content_types(self):
        """Test converting messages with mixed content types (text and image)."""
        # Create a message with an image content
        image_content = ImageContent(
            type="image", data="base64_encoded_image_data", mimeType="image/png"
        )

        messages = [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text="Look at this image:"),
            ),
            PromptMessage(role="user", content=image_content),
        ]

        # Convert to PromptMessageMultipart
        result = PromptMessageMultipart.to_multipart(messages)

        # Verify results
        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].content) == 2
        assert result[0].content[0].text == "Look at this image:"
        assert result[0].content[1].type == "image"
        assert result[0].content[1].data == "base64_encoded_image_data"
        assert result[0].content[1].mimeType == "image/png"

    def test_to_prompt_messages(self):
        """Test converting a PromptMessageMultipart back to PromptMessages."""
        # Create a multipart message
        multipart = PromptMessageMultipart(
            role="user",
            content=[
                TextContent(type="text", text="Hello"),
                TextContent(type="text", text="How are you?"),
            ],
        )

        # Convert back to PromptMessages
        result = multipart.from_multipart()

        # Verify results
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "user"
        assert result[0].content.text == "Hello"
        assert result[1].content.text == "How are you?"

    def test_parse_get_prompt_result(self):
        """Test parsing a GetPromptResult into PromptMessageMultipart objects."""
        # Create test messages
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="assistant", content=TextContent(type="text", text="Hi there!")),
            PromptMessage(role="user", content=TextContent(type="text", text="How are you?")),
        ]

        # Create a GetPromptResult
        result = GetPromptResult(messages=messages)

        # Parse into PromptMessageMultipart objects
        multiparts = PromptMessageMultipart.parse_get_prompt_result(result)

        # Verify results
        assert len(multiparts) == 3
        assert multiparts[0].role == "user"
        assert multiparts[1].role == "assistant"
        assert multiparts[2].role == "user"
        assert len(multiparts[0].content) == 1
        assert len(multiparts[1].content) == 1
        assert len(multiparts[2].content) == 1
        assert multiparts[0].content[0].text == "Hello"
        assert multiparts[1].content[0].text == "Hi there!"
        assert multiparts[2].content[0].text == "How are you?"

    def test_empty_messages(self):
        """Test handling of empty message lists."""
        # Convert an empty list
        result = PromptMessageMultipart.to_multipart([])

        # Should return an empty list
        assert result == []

    def test_round_trip_conversion(self):
        """Test round-trip conversion from PromptMessages to Multipart and back."""
        # Original messages
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="user", content=TextContent(type="text", text="How are you?")),
            PromptMessage(
                role="assistant",
                content=TextContent(type="text", text="I'm doing well, thanks!"),
            ),
        ]

        # Convert to multipart
        multiparts = PromptMessageMultipart.to_multipart(messages)

        # Convert back to regular messages
        result = []
        for mp in multiparts:
            result.extend(mp.from_multipart())

        # Verify the result matches the original
        assert len(result) == len(messages)
        for i in range(len(messages)):
            assert result[i].role == messages[i].role
            assert result[i].content.text == messages[i].content.text

    def test_from_get_prompt_result(self):
        """Test from_get_prompt_result method with error handling."""
        # Test with valid GetPromptResult
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="assistant", content=TextContent(type="text", text="Hi there!")),
        ]
        result = GetPromptResult(messages=messages)

        multiparts = PromptMessageMultipart.from_get_prompt_result(result)
        assert len(multiparts) == 2
        assert multiparts[0].role == "user"
        assert multiparts[1].role == "assistant"

        # Test with None
        multiparts = PromptMessageMultipart.from_get_prompt_result(None)
        assert multiparts == []

        # Test with empty result
        empty_result = GetPromptResult(messages=[])
        multiparts = PromptMessageMultipart.from_get_prompt_result(empty_result)
        assert multiparts == []

    def test_getting_last_text_empty(self):
        """Test from_get_prompt_result method with error handling."""
        # Test with valid GetPromptResult
        assert None is Prompt.user().last_text()
        assert "last" == Prompt.user("first", "last").last_text()

    def test_convenience_add_text(self):
        """Test from_get_prompt_result method with error handling."""
        # Test with valid GetPromptResult
        multipart = Prompt.user("hello", "world")
        assert 2 == len(multipart.content)

        multipart.add_text("foo")
        assert 3 == len(multipart.content)
        assert "foo" == multipart.last_text()
        assert isinstance(multipart.content[2], TextContent)
