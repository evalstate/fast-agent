"""
Tests for prompt helper functions.
"""

import pytest
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)
from pydantic.networks import AnyUrl

from fast_agent.mcp.helpers.content_helpers import (
    get_image_data,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
    split_thinking_content,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompts.prompt_helpers import MessageContent


# Test fixture data
@pytest.fixture
def text_content():
    return TextContent(type="text", text="Hello, world!")


@pytest.fixture
def image_content():
    return ImageContent(type="image", data="base64data", mimeType="image/png")


@pytest.fixture
def text_embedded_resource():
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("file:///example.txt"), text="Resource content", mimeType="text/plain"
        ),
    )


@pytest.fixture
def blob_resource():
    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("file:///example.png"), blob="base64blobdata", mimeType="image/png"
        ),
    )


@pytest.fixture
def text_resource():
    return TextResourceContents(text="text_resource", uri=AnyUrl("file://example.txt"))


# Test content type extraction
def test_get_text(
    text_content, image_content, text_embedded_resource, blob_resource, text_resource
):
    assert get_text(text_content) == "Hello, world!"
    assert get_text(text_embedded_resource) == "Resource content"
    assert get_text(text_resource) == "text_resource"
    assert get_text(image_content) is None
    assert get_text(blob_resource) is None
    assert get_text(object()) is None


def test_get_image_data(text_content, image_content, text_embedded_resource, blob_resource):
    assert get_image_data(image_content) == "base64data"
    assert get_image_data(blob_resource) == "base64blobdata"
    assert get_image_data(text_content) is None
    assert get_image_data(text_embedded_resource) is None
    assert get_image_data(object()) is None


def test_get_resource_uri(text_content, image_content, text_embedded_resource, blob_resource):
    assert get_resource_uri(text_embedded_resource) == "file:///example.txt"
    assert get_resource_uri(blob_resource) == "file:///example.png"
    assert get_resource_uri(text_content) is None
    assert get_resource_uri(image_content) is None
    assert get_resource_uri(object()) is None


def test_is_text_content(text_content, image_content, text_embedded_resource, text_resource):
    assert is_text_content(text_content) is True
    assert is_text_content(text_resource) is True
    assert is_text_content(image_content) is False
    assert is_text_content(text_embedded_resource) is False


def test_is_image_content(text_content, image_content, text_embedded_resource):
    assert is_image_content(image_content) is True
    assert is_image_content(text_content) is False
    assert is_image_content(text_embedded_resource) is False


def test_is_resource_content(text_content, image_content, text_embedded_resource):
    assert is_resource_content(text_embedded_resource) is True
    assert is_resource_content(text_content) is False
    assert is_resource_content(image_content) is False


# Test MessageContent helper class with PromptMessage
def test_message_content_with_prompt_message(text_content, image_content, text_embedded_resource):
    text_msg = PromptMessage(role="user", content=text_content)
    image_msg = PromptMessage(role="user", content=image_content)
    resource_msg = PromptMessage(role="user", content=text_embedded_resource)

    # Test text extraction
    assert MessageContent.get_first_text(text_msg) == "Hello, world!"
    assert MessageContent.get_first_text(image_msg) is None
    assert MessageContent.get_first_text(resource_msg) == "Resource content"


# Test MessageContent helper class with PromptMessageExtended
def test_message_content_with_multipart(
    text_content, image_content, text_embedded_resource, blob_resource
):
    # Create a multipart message with both text and image content
    multipart_msg = PromptMessageExtended(
        role="user", content=[text_content, image_content, text_embedded_resource, blob_resource]
    )

    assert MessageContent.get_first_text(multipart_msg) == "Hello, world!"

    # Test with empty multipart message
    empty_msg = PromptMessageExtended(role="user", content=[])
    assert MessageContent.get_first_text(empty_msg) is None


# Test split_thinking_content function
def test_split_thinking_content():
    # Test with thinking block
    message_with_thinking = "<think>This is my thought process</think>This is the actual content"
    thinking, content = split_thinking_content(message_with_thinking)
    assert thinking == "This is my thought process"
    assert content == "This is the actual content"

    # Test with multiline thinking block
    multiline_message = """<think>
    Line 1 of thinking
    Line 2 of thinking
    </think>
    The main content here"""
    thinking, content = split_thinking_content(multiline_message)
    assert thinking == "Line 1 of thinking\n    Line 2 of thinking"
    assert content == "The main content here"

    # Test without thinking block
    plain_message = "Just regular content without thinking"
    thinking, content = split_thinking_content(plain_message)
    assert thinking is None
    assert content == "Just regular content without thinking"

    # Test with malformed thinking tag (no closing tag)
    malformed_message = "<think>Unclosed thinking block\nSome content"
    thinking, content = split_thinking_content(malformed_message)
    assert thinking is None
    assert content == malformed_message

    # Test with empty thinking block
    empty_thinking = "<think></think>Main content"
    thinking, content = split_thinking_content(empty_thinking)
    assert thinking == ""
    assert content == "Main content"

    # Test with thinking block not at start
    middle_thinking = "Some text <think>thoughts</think> more text"
    thinking, content = split_thinking_content(middle_thinking)
    assert thinking is None
    assert content == middle_thinking

    # Test with consecutive thinking blocks
    chained_thinking = "<think>first</think><think>second</think>Main content"
    thinking, content = split_thinking_content(chained_thinking)
    assert thinking == "first\nsecond"
    assert content == "Main content"
