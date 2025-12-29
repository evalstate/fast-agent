"""
Unit tests for the PromptMessageExtended class.
"""

from mcp.types import (
    GetPromptResult,
    ImageContent,
    PromptMessage,
    TextContent,
)

from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def _text(block: object) -> TextContent:
    assert isinstance(block, TextContent)
    return block


def _image(block: object) -> ImageContent:
    assert isinstance(block, ImageContent)
    return block


class TestPromptMessageExtended:
    """Tests for the PromptMessageExtended class."""

    def test_from_prompt_messages_with_single_role(self):
        """Test converting a sequence of PromptMessages with the same role."""
        # Create test messages
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="user", content=TextContent(type="text", text="How are you?")),
        ]

        # Convert to PromptMessageExtended
        result = PromptMessageExtended.to_extended(messages)

        # Verify results
        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].content) == 2
        assert _text(result[0].content[0]).text == "Hello"
        assert _text(result[0].content[1]).text == "How are you?"

    def test_from_prompt_messages_with_multiple_roles(self):
        """Test converting a sequence of PromptMessages with different roles."""
        # Create test messages with alternating roles
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="assistant", content=TextContent(type="text", text="Hi there!")),
            PromptMessage(role="user", content=TextContent(type="text", text="How are you?")),
        ]

        # Convert to PromptMessageExtended
        result = PromptMessageExtended.to_extended(messages)

        # Verify results
        assert len(result) == 3
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[2].role == "user"
        assert len(result[0].content) == 1
        assert len(result[1].content) == 1
        assert len(result[2].content) == 1
        assert _text(result[0].content[0]).text == "Hello"
        assert _text(result[1].content[0]).text == "Hi there!"
        assert _text(result[2].content[0]).text == "How are you?"

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

        # Convert to PromptMessageExtended
        result = PromptMessageExtended.to_extended(messages)

        # Verify results
        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].content) == 2
        assert _text(result[0].content[0]).text == "Look at this image:"
        image_block = _image(result[0].content[1])
        assert image_block.type == "image"
        assert image_block.data == "base64_encoded_image_data"
        assert image_block.mimeType == "image/png"

    def test_to_prompt_messages(self):
        """Test converting a PromptMessageExtended back to PromptMessages."""
        # Create a multipart message
        multipart = PromptMessageExtended(
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
        assert _text(result[0].content).text == "Hello"
        assert _text(result[1].content).text == "How are you?"

    def test_parse_get_prompt_result(self):
        """Test parsing a GetPromptResult into PromptMessageExtended objects."""
        # Create test messages
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="assistant", content=TextContent(type="text", text="Hi there!")),
            PromptMessage(role="user", content=TextContent(type="text", text="How are you?")),
        ]

        # Create a GetPromptResult
        result = GetPromptResult(messages=messages)

        # Parse into PromptMessageExtended objects
        multiparts = PromptMessageExtended.parse_get_prompt_result(result)

        # Verify results
        assert len(multiparts) == 3
        assert multiparts[0].role == "user"
        assert multiparts[1].role == "assistant"
        assert multiparts[2].role == "user"
        assert len(multiparts[0].content) == 1
        assert len(multiparts[1].content) == 1
        assert len(multiparts[2].content) == 1
        assert _text(multiparts[0].content[0]).text == "Hello"
        assert _text(multiparts[1].content[0]).text == "Hi there!"
        assert _text(multiparts[2].content[0]).text == "How are you?"

    def test_empty_messages(self):
        """Test handling of empty message lists."""
        # Convert an empty list
        result = PromptMessageExtended.to_extended([])

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
        multiparts = PromptMessageExtended.to_extended(messages)

        # Convert back to regular messages
        result = []
        for mp in multiparts:
            result.extend(mp.from_multipart())

        # Verify the result matches the original
        assert len(result) == len(messages)
        for i in range(len(messages)):
            assert result[i].role == messages[i].role
            assert _text(result[i].content).text == _text(messages[i].content).text

    def test_from_get_prompt_result(self):
        """Test from_get_prompt_result method with error handling."""
        # Test with valid GetPromptResult
        messages = [
            PromptMessage(role="user", content=TextContent(type="text", text="Hello")),
            PromptMessage(role="assistant", content=TextContent(type="text", text="Hi there!")),
        ]
        result = GetPromptResult(messages=messages)

        multiparts = PromptMessageExtended.from_get_prompt_result(result)
        assert len(multiparts) == 2
        assert multiparts[0].role == "user"
        assert multiparts[1].role == "assistant"

        # Test with None
        multiparts = PromptMessageExtended.from_get_prompt_result(None)
        assert multiparts == []

        # Test with empty result
        empty_result = GetPromptResult(messages=[])
        multiparts = PromptMessageExtended.from_get_prompt_result(empty_result)
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
