"""
Tests for the Prompt class.
"""

import base64
import os
import tempfile
from pathlib import Path

from mcp.types import EmbeddedResource, ImageContent, PromptMessage, TextContent

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from mcp_agent.core.prompt import Prompt


def test_user_method():
    """Test the Prompt.user method."""
    # Test with simple text
    message = Prompt.user("Hello, world!")

    assert isinstance(message, PromptMessageExtended)
    assert message.role == "user"
    assert len(message.content) == 1
    assert isinstance(message.content[0], TextContent)
    assert message.content[0].text == "Hello, world!"

    # Test with multiple items
    message = Prompt.user("Hello,", "How are you?")

    assert isinstance(message, PromptMessageExtended)
    assert message.role == "user"
    assert len(message.content) == 2
    assert message.content[0].text == "Hello,"
    assert message.content[1].text == "How are you?"

    # Test with PromptMessage
    prompt_message = PromptMessage(
        role="assistant", content=TextContent(type="text", text="I'm a PromptMessage")
    )
    message = Prompt.user(prompt_message)

    assert isinstance(message, PromptMessageExtended)
    assert message.role == "user"  # Role should be changed to user
    assert len(message.content) == 1
    assert message.content[0].text == "I'm a PromptMessage"

    # Test with PromptMessageExtended
    multipart = Prompt.assistant("I'm a multipart message")
    message = Prompt.user(multipart)

    assert isinstance(message, PromptMessageExtended)
    assert message.role == "user"  # Role should be changed to user
    assert len(message.content) == 1
    assert message.content[0].text == "I'm a multipart message"


def test_assistant_method():
    """Test the Prompt.assistant method."""
    # Test with simple text
    message = Prompt.assistant("I'm doing well, thanks!")

    assert isinstance(message, PromptMessageExtended)
    assert message.role == "assistant"
    assert len(message.content) == 1
    assert isinstance(message.content[0], TextContent)
    assert message.content[0].text == "I'm doing well, thanks!"


def test_message_method():
    """Test the Prompt.message method."""
    # Test with user role (default)
    message = Prompt.message("Hello")

    assert isinstance(message, PromptMessageExtended)
    assert message.role == "user"

    # Test with assistant role
    message = Prompt.message("Hello", role="assistant")

    assert isinstance(message, PromptMessageExtended)
    assert message.role == "assistant"


def test_with_file_paths():
    """Test the Prompt class with file paths."""
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as text_file:
        text_file.write(b"Hello, world!")
        text_path = text_file.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
        image_file.write(b"fake image data")
        image_path = image_file.name

    try:
        # Test with text file
        message = Prompt.user("Check this file:", Path(text_path))

        assert message.role == "user"
        assert len(message.content) == 2
        assert message.content[0].text == "Check this file:"
        assert isinstance(message.content[1], EmbeddedResource)
        assert message.content[1].resource.text == "Hello, world!"

        # Test with image file
        message = Prompt.assistant("Here's the image:", Path(image_path))

        assert message.role == "assistant"
        assert len(message.content) == 2
        assert message.content[0].text == "Here's the image:"
        assert isinstance(message.content[1], ImageContent)

        # Decode the base64 data
        decoded = base64.b64decode(message.content[1].data)
        assert decoded == b"fake image data"

        # Test with ResourceContents and EmbeddedResource
        from mcp.types import ReadResourceResult, TextResourceContents
        from pydantic import AnyUrl

        # Create a TextResourceContents
        text_resource = TextResourceContents(
            uri=AnyUrl("file:///test/example.txt"), text="Sample text", mimeType="text/plain"
        )

        # Test with ResourceContent
        message = Prompt.user("Check this resource:", text_resource)
        assert message.role == "user"
        assert len(message.content) == 2
        assert isinstance(message.content[1], EmbeddedResource)
        assert message.content[1].resource == text_resource

        # Test with EmbeddedResource
        embedded = EmbeddedResource(type="resource", resource=text_resource)
        message = Prompt.user("Another resource:", embedded)
        assert message.role == "user"
        assert len(message.content) == 2
        # Using dictionary comparison because the objects might not be identity-equal
        assert message.content[1].type == embedded.type
        assert message.content[1].resource.text == embedded.resource.text

        # Test with ReadResourceResult
        resource_result = ReadResourceResult(contents=[text_resource])
        message = Prompt.user("Resource result:", resource_result)
        assert message.role == "user"
        assert len(message.content) > 1  # Should have text + resource
        assert message.content[0].text == "Resource result:"
        assert isinstance(message.content[1], EmbeddedResource)

        # Test with direct TextContent
        text_content = TextContent(type="text", text="Direct text content")
        message = Prompt.user(text_content)
        assert message.role == "user"
        assert len(message.content) == 1
        assert message.content[0] == text_content

        # Test with direct ImageContent
        image_content = ImageContent(
            type="image", data="ZmFrZSBpbWFnZSBkYXRh", mimeType="image/png"
        )
        message = Prompt.assistant(image_content)
        assert message.role == "assistant"
        assert len(message.content) == 1
        assert message.content[0] == image_content

        # Test with mixed content including direct content types
        message = Prompt.user("Text followed by:", text_content, "And an image:", image_content)
        assert message.role == "user"
        assert len(message.content) == 4
        assert message.content[0].text == "Text followed by:"
        assert message.content[1] == text_content
        assert message.content[2].text == "And an image:"
        assert message.content[3] == image_content

    finally:
        # Clean up
        os.unlink(text_path)
        os.unlink(image_path)


def test_conversation_method():
    """Test the Prompt.conversation method."""
    # Create conversation from PromptMessageExtended objects
    user_msg = Prompt.user("Hello")
    assistant_msg = Prompt.assistant("Hi there!")

    conversation = Prompt.conversation(user_msg, assistant_msg)

    assert len(conversation) == 2
    assert all(isinstance(msg, PromptMessage) for msg in conversation)
    assert conversation[0].role == "user"
    assert conversation[1].role == "assistant"

    # Test with mixed inputs
    mixed_conversation = Prompt.conversation(
        user_msg,
        {"role": "assistant", "content": TextContent(type="text", text="Direct dict!")},
        Prompt.user("Another message"),
    )

    assert len(mixed_conversation) == 3
    assert mixed_conversation[0].role == "user"
    assert mixed_conversation[1].role == "assistant"
    assert mixed_conversation[1].content.text == "Direct dict!"
    assert mixed_conversation[2].role == "user"


def test_from_multipart_method():
    """Test the Prompt.from_multipart method."""
    # Create a list of multipart messages
    multipart_msgs = [
        Prompt.user("Hello"),
        Prompt.assistant("Hi there!"),
        Prompt.user("How are you?"),
    ]

    # Convert to PromptMessages
    messages = Prompt.from_multipart(multipart_msgs)

    assert len(messages) == 3
    assert all(isinstance(msg, PromptMessage) for msg in messages)
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[2].role == "user"

    # Test with PromptMessageExtended instances containing multiple content items
    complex_multipart = [
        Prompt.user("Hello,", "How are you?"),
        Prompt.assistant("I'm fine,", "Thanks for asking!"),
    ]

    messages = Prompt.from_multipart(complex_multipart)

    assert len(messages) == 4  # 2 content items in each multipart = 4 total messages
    assert messages[0].role == "user"
    assert messages[1].role == "user"
    assert messages[2].role == "assistant"
    assert messages[3].role == "assistant"
