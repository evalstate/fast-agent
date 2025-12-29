import base64

from google.genai import types
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from fast_agent.llm.provider.google.google_converter import GoogleConverter
from fast_agent.types import (
    PromptMessageExtended,
    audio_link,
    image_link,
    resource_link,
    video_link,
)


def test_convert_function_results_to_google_text_only():
    converter = GoogleConverter()

    # Create a simple text-only tool result
    result = CallToolResult(
        content=[TextContent(type="text", text="Weather is sunny")], isError=False
    )

    contents = converter.convert_function_results_to_google([("weather", result)])

    # One google Content with role 'tool'
    assert isinstance(contents, list)
    assert len(contents) == 1
    content = contents[0]
    assert isinstance(content, types.Content)
    assert content.role == "tool"
    parts = content.parts
    assert parts is not None
    # First part should be a function response named 'weather'
    fn_resp = parts[0].function_response
    assert fn_resp is not None
    assert fn_resp.name == "weather"
    assert isinstance(fn_resp.response, dict)
    assert fn_resp.response.get("tool_name") == "weather"


def test_clean_schema_for_google_const_string_to_enum():
    converter = GoogleConverter()
    schema = {"type": "string", "const": "all"}
    cleaned = converter._clean_schema_for_google(schema)
    # Expect const rewritten to enum ["all"]
    assert "const" not in cleaned
    assert cleaned.get("enum") == ["all"]


def test_clean_schema_for_google_const_non_string_dropped():
    converter = GoogleConverter()
    schema_bool = {"type": "boolean", "const": True}
    cleaned_bool = converter._clean_schema_for_google(schema_bool)
    # Non-string const dropped
    assert "const" not in cleaned_bool
    assert "enum" not in cleaned_bool

    schema_num = {"type": "number", "const": 3.14}
    cleaned_num = converter._clean_schema_for_google(schema_num)
    assert "const" not in cleaned_num
    assert "enum" not in cleaned_num


def test_convert_video_resource():
    converter = GoogleConverter()
    
    # Create a mock video resource
    video_bytes = b"fake_video_bytes"
    encoded_video = base64.b64encode(video_bytes).decode("utf-8")
    
    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("file:///path/to/video.mp4"),
            mimeType="video/mp4",
            blob=encoded_video
        )
    )
    
    # Wrap in PromptMessageExtended
    message = PromptMessageExtended(
        role="user",
        content=[resource]
    )
    
    # Convert - pass as a list!
    contents = converter.convert_to_google_content([message])
    
    # Verify
    assert isinstance(contents, list)
    assert len(contents) == 1
    content = contents[0]
    
    assert isinstance(content, types.Content)
    parts = content.parts
    assert parts is not None
    assert len(parts) == 1
    part = parts[0]
    
    # Check if it's an inline data part
    assert part.inline_data is not None
    assert part.inline_data.mime_type == "video/mp4"
    assert part.inline_data.data == video_bytes


def test_convert_mixed_content_video_text():
    converter = GoogleConverter()
    
    # Video resource
    video_bytes = b"video_data"
    encoded_video = base64.b64encode(video_bytes).decode("utf-8")
    video_resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("file:///video.mp4"),
            mimeType="video/mp4",
            blob=encoded_video
        )
    )
    
    # Text content
    text_content = TextContent(type="text", text="Describe this video")
    
    # Mixed message
    message = PromptMessageExtended(
        role="user",
        content=[video_resource, text_content]
    )
    
    # Convert - pass as a list!
    contents = converter.convert_to_google_content([message])
    
    # Verify
    assert len(contents) == 1
    content = contents[0]
    parts = content.parts
    assert parts is not None
    assert len(parts) == 2
    
    # First part should be video
    assert parts[0].inline_data is not None
    assert parts[0].inline_data.mime_type == "video/mp4"
    
    # Second part should be text
    assert parts[1].text == "Describe this video"


def test_convert_youtube_url_video():
    converter = GoogleConverter()

    # Create a YouTube URL video resource (TextResourceContents, not BlobResourceContents)
    youtube_resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            mimeType="video/mp4",
            text="YouTube video"
        )
    )
    
    message = PromptMessageExtended(
        role="user",
        content=[youtube_resource]
    )
    
    # Convert - pass as a list!
    contents = converter.convert_to_google_content([message])
    
    # Verify
    assert len(contents) == 1
    content = contents[0]
    parts = content.parts
    assert parts is not None
    assert len(parts) == 1
    part = parts[0]
    
    # Should use file_data for YouTube URLs
    assert part.file_data is not None
    assert part.file_data.file_uri == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert part.file_data.mime_type == "video/mp4"


def test_convert_resource_link_video():
    """Test that video ResourceLink uses Part.from_uri()"""
    converter = GoogleConverter()

    link = video_link("https://example.com/video.mp4", name="video_resource")

    message = PromptMessageExtended(role="user", content=[link])

    contents = converter.convert_to_google_content([message])

    assert len(contents) == 1
    content = contents[0]
    parts = content.parts
    assert parts is not None
    assert len(parts) == 1
    part = parts[0]

    # Should use file_data for video ResourceLink
    assert part.file_data is not None
    assert part.file_data.file_uri == "https://example.com/video.mp4"
    assert part.file_data.mime_type == "video/mp4"


def test_convert_resource_link_image():
    """Test that image ResourceLink uses Part.from_uri()"""
    converter = GoogleConverter()

    link = image_link("https://example.com/photo.png", name="image_resource")

    message = PromptMessageExtended(role="user", content=[link])

    contents = converter.convert_to_google_content([message])

    assert len(contents) == 1
    content = contents[0]
    parts = content.parts
    assert parts is not None
    assert len(parts) == 1
    part = parts[0]

    # Should use file_data for image ResourceLink
    assert part.file_data is not None
    assert part.file_data.file_uri == "https://example.com/photo.png"
    assert part.file_data.mime_type == "image/png"


def test_convert_resource_link_audio():
    """Test that audio ResourceLink uses Part.from_uri()"""
    converter = GoogleConverter()

    link = audio_link("https://example.com/audio.mp3", name="audio_resource")

    message = PromptMessageExtended(role="user", content=[link])

    contents = converter.convert_to_google_content([message])

    assert len(contents) == 1
    content = contents[0]
    parts = content.parts
    assert parts is not None
    assert len(parts) == 1
    part = parts[0]

    # Should use file_data for audio ResourceLink
    assert part.file_data is not None
    assert part.file_data.file_uri == "https://example.com/audio.mp3"
    assert part.file_data.mime_type == "audio/mpeg"


def test_convert_resource_link_text_fallback():
    """Test that non-media ResourceLink falls back to text representation"""
    converter = GoogleConverter()

    link = resource_link(
        "https://example.com/document.json",
        name="document_resource",
        description="A JSON config file",
    )

    message = PromptMessageExtended(role="user", content=[link])

    contents = converter.convert_to_google_content([message])

    assert len(contents) == 1
    content = contents[0]
    parts = content.parts
    assert parts is not None
    assert len(parts) == 1
    part = parts[0]

    # Should use text for non-media ResourceLink
    assert part.text is not None
    assert "document_resource" in part.text
    assert "https://example.com/document.json" in part.text
    assert "application/json" in part.text


def test_convert_resource_link_in_tool_result():
    """Test ResourceLink in tool results"""
    converter = GoogleConverter()

    # Create a tool result with a video ResourceLink
    link = video_link("https://storage.example.com/output.mp4", name="generated_video")

    result = CallToolResult(content=[link], isError=False)

    contents = converter.convert_function_results_to_google([("video_generator", result)])

    assert len(contents) == 1
    content = contents[0]
    assert content.role == "tool"

    # Should have function response part and media part
    parts = content.parts
    assert parts is not None
    assert len(parts) >= 1

    # Check for the media part (video)
    media_parts = [p for p in parts if p.file_data is not None]
    assert len(media_parts) == 1
    assert media_parts[0].file_data is not None
    assert media_parts[0].file_data.file_uri == "https://storage.example.com/output.mp4"
    assert media_parts[0].file_data.mime_type == "video/mp4"


def test_convert_resource_link_text_in_tool_result():
    """Test non-media ResourceLink in tool results falls back to text"""
    converter = GoogleConverter()

    # Create a tool result with a text ResourceLink (YAML is not a media type)
    link = resource_link(
        "https://example.com/config.yaml",
        name="config_file",
        mime_type="application/yaml",
    )

    result = CallToolResult(content=[link], isError=False)

    contents = converter.convert_function_results_to_google([("config_reader", result)])

    assert len(contents) == 1
    content = contents[0]
    assert content.role == "tool"

    # Should have function response part with text content
    parts = content.parts
    assert parts is not None
    fn_resp = parts[0].function_response
    assert fn_resp is not None
    response = fn_resp.response
    assert isinstance(response, dict)
    assert "text_content" in response
    assert "config_file" in response["text_content"]
