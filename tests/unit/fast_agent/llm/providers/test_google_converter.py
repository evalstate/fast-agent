import base64
from google.genai import types
from mcp.types import CallToolResult, TextContent, EmbeddedResource, BlobResourceContents, TextResourceContents

from fast_agent.llm.provider.google.google_converter import GoogleConverter
from fast_agent.types import PromptMessageExtended


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
    assert content.parts
    # First part should be a function response named 'weather'
    fn_resp = content.parts[0].function_response
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
            uri="file:///path/to/video.mp4",
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
    assert len(content.parts) == 1
    part = content.parts[0]
    
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
            uri="file:///video.mp4",
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
    assert len(content.parts) == 2
    
    # First part should be video
    assert content.parts[0].inline_data is not None
    assert content.parts[0].inline_data.mime_type == "video/mp4"
    
    # Second part should be text
    assert content.parts[1].text == "Describe this video"


def test_convert_youtube_url_video():
    converter = GoogleConverter()
    
    # Create a YouTube URL video resource (TextResourceContents, not BlobResourceContents)
    from mcp.types import TextResourceContents
    from pydantic import AnyUrl
    
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
    assert len(content.parts) == 1
    part = content.parts[0]
    
    # Should use file_data for YouTube URLs
    assert part.file_data is not None
    assert part.file_data.file_uri == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert part.file_data.mime_type == "video/mp4"
