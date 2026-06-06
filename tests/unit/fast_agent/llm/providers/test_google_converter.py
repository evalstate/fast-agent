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

    contents = converter.convert_function_results_to_google([("weather", "call_123", result)])

    # One google Content with user role, per Gemini function-response protocol.
    assert isinstance(contents, list)
    assert len(contents) == 1
    content = contents[0]
    assert isinstance(content, types.Content)
    assert content.role == "user"
    parts = content.parts
    assert parts is not None
    # First part should be a function response named 'weather'
    fn_resp = parts[0].function_response
    assert fn_resp is not None
    assert fn_resp.name == "weather"
    assert fn_resp.id == "call_123"
    assert isinstance(fn_resp.response, dict)
    assert fn_resp.response.get("result") == "Weather is sunny"


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


def test_clean_schema_for_google_resolves_escaped_refs_and_preserves_siblings():
    converter = GoogleConverter()
    schema = {
        "$defs": {
            "path/with~tilde": {
                "type": "string",
                "minLength": 3,
            }
        },
        "type": "object",
        "properties": {
            "name": {
                "$ref": "#/$defs/path~1with~0tilde",
                "description": "Display name",
                "default": "Ada",
            }
        },
    }

    cleaned = converter._clean_schema_for_google(schema)

    assert "$defs" not in cleaned
    assert cleaned["properties"]["name"] == {
        "type": "string",
        "minLength": 3,
        "description": "Display name",
        "default": "Ada",
    }


def test_convert_from_google_content_handles_missing_content_parts():
    converter = GoogleConverter()

    assert converter.convert_from_google_content(None) == []
    assert converter.convert_from_google_content(types.Content(role="model", parts=None)) == []


def test_convert_from_google_content_list_handles_missing_content_parts():
    converter = GoogleConverter()

    missing_content = converter._convert_from_google_content(None)
    missing_parts = converter.convert_from_google_content_list(
        [types.Content(role="model", parts=None)]
    )[0]

    assert missing_content.content == []
    assert missing_parts.content == []


def test_convert_video_resource():
    converter = GoogleConverter()

    # Create a mock video resource
    video_bytes = b"fake_video_bytes"
    encoded_video = base64.b64encode(video_bytes).decode("utf-8")

    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("file:///path/to/video.mp4"), mimeType="video/mp4", blob=encoded_video
        ),
    )

    # Wrap in PromptMessageExtended
    message = PromptMessageExtended(role="user", content=[resource])

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
            uri=AnyUrl("file:///video.mp4"), mimeType="video/mp4", blob=encoded_video
        ),
    )

    # Text content
    text_content = TextContent(type="text", text="Describe this video")

    # Mixed message
    message = PromptMessageExtended(role="user", content=[video_resource, text_content])

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


def test_convert_audio_blob_resource():
    converter = GoogleConverter()

    audio_bytes = b"audio_data"
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("file:///audio.mp3"),
            mimeType="audio/mpeg",
            blob=encoded_audio,
        ),
    )

    contents = converter.convert_to_google_content(
        [PromptMessageExtended(role="user", content=[audio_resource])]
    )

    assert len(contents) == 1
    parts = contents[0].parts
    assert parts is not None
    assert len(parts) == 1
    assert parts[0].inline_data is not None
    assert parts[0].inline_data.mime_type == "audio/mpeg"
    assert parts[0].inline_data.data == audio_bytes


def test_convert_youtube_url_video():
    converter = GoogleConverter()

    # Create a YouTube URL video resource (TextResourceContents, not BlobResourceContents)
    youtube_resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            mimeType="video/mp4",
            text="YouTube video",
        ),
    )

    message = PromptMessageExtended(role="user", content=[youtube_resource])

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

    contents = converter.convert_function_results_to_google([("video_generator", None, result)])

    assert len(contents) == 1
    content = contents[0]
    assert content.role == "user"

    # Media must live inside the function response, not alongside it.
    parts = content.parts
    assert parts is not None
    assert len(parts) == 1

    fn_resp = parts[0].function_response
    assert fn_resp is not None
    response_parts = fn_resp.parts or []
    media_parts = [p for p in response_parts if p.file_data is not None]
    assert len(media_parts) == 1
    assert media_parts[0].file_data is not None
    assert media_parts[0].file_data.file_uri == "https://storage.example.com/output.mp4"
    assert media_parts[0].file_data.mime_type == "video/mp4"


def test_convert_video_blob_in_tool_result():
    """Test embedded video blobs in tool results become inline media parts."""
    converter = GoogleConverter()

    video_bytes = b"video_data"
    encoded_video = base64.b64encode(video_bytes).decode("utf-8")
    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("file:///video.mp4"),
            mimeType="video/mp4",
            blob=encoded_video,
        ),
    )
    result = CallToolResult(content=[resource], isError=False)

    contents = converter.convert_function_results_to_google([("attach_media", None, result)])

    assert len(contents) == 1
    parts = contents[0].parts
    assert parts is not None
    fn_resp = parts[0].function_response
    assert fn_resp is not None
    response_parts = fn_resp.parts or []
    media_parts = [part for part in response_parts if part.inline_data is not None]
    assert len(media_parts) == 1
    assert media_parts[0].inline_data is not None
    assert media_parts[0].inline_data.mime_type == "video/mp4"
    assert media_parts[0].inline_data.data == video_bytes


def test_convert_audio_blob_in_tool_result():
    """Test embedded audio blobs in tool results become inline media parts."""
    converter = GoogleConverter()

    audio_bytes = b"audio_data"
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=AnyUrl("file:///audio.mp3"),
            mimeType="audio/mpeg",
            blob=encoded_audio,
        ),
    )
    result = CallToolResult(content=[resource], isError=False)

    contents = converter.convert_function_results_to_google([("attach_media", None, result)])

    assert len(contents) == 1
    parts = contents[0].parts
    assert parts is not None
    fn_resp = parts[0].function_response
    assert fn_resp is not None
    response_parts = fn_resp.parts or []
    media_parts = [part for part in response_parts if part.inline_data is not None]
    assert len(media_parts) == 1
    assert media_parts[0].inline_data is not None
    assert media_parts[0].inline_data.mime_type == "audio/mpeg"
    assert media_parts[0].inline_data.data == audio_bytes


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

    contents = converter.convert_function_results_to_google([("config_reader", None, result)])

    assert len(contents) == 1
    content = contents[0]
    assert content.role == "user"

    # Should have function response part with text content
    parts = content.parts
    assert parts is not None
    fn_resp = parts[0].function_response
    assert fn_resp is not None
    response = fn_resp.response
    assert isinstance(response, dict)
    assert "result" in response
    assert "config_file" in response["result"]


def test_gemini3_removes_sampling_parameters_and_budget():
    """Test that temperature, top_p, top_k, and raw thinking_budget are stripped/handled for Gemini 3.x."""
    from fast_agent.types import RequestParams

    converter = GoogleConverter()
    params = RequestParams(
        model=" GEMINI-3.5-FLASH ",
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )
    # Generate content config with thinking_budget set
    config = converter.convert_request_params_to_google_config(
        params,
        thinking_budget=5000,
    )

    # Temperature, top_p, and top_k must be stripped as per Gemini 3.5 API guidance.
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None

    # raw thinking_budget must be omitted, and mapped to thinking_level MEDIUM instead.
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == "MEDIUM"
    assert config.thinking_config.thinking_budget is None


def test_convert_multiple_function_results_into_single_content():
    """Test that multiple tool results are combined into a single Content object."""
    converter = GoogleConverter()

    result1 = CallToolResult(content=[TextContent(type="text", text="Output 1")], isError=False)
    result2 = CallToolResult(content=[TextContent(type="text", text="Output 2")], isError=False)

    contents = converter.convert_function_results_to_google(
        [
            ("tool_one", "id_1", result1),
            ("tool_two", "id_2", result2),
        ]
    )

    assert isinstance(contents, list)
    assert len(contents) == 1
    content = contents[0]
    assert content.role == "user"
    assert content.parts is not None
    assert len(content.parts) == 2

    part1 = content.parts[0].function_response
    assert part1 is not None
    assert part1.name == "tool_one"
    assert part1.id == "id_1"
    assert isinstance(part1.response, dict)
    assert part1.response.get("result") == "Output 1"

    part2 = content.parts[1].function_response
    assert part2 is not None
    assert part2.name == "tool_two"
    assert part2.id == "id_2"
    assert isinstance(part2.response, dict)
    assert part2.response.get("result") == "Output 2"
