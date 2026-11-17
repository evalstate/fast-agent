import base64

import pytest
from google.genai import types
from mcp import Tool
from mcp_types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    TextContent,
    TextResourceContents,
)

from fast_agent.llm.provider.google.google_converter import GoogleConverter
from fast_agent.types import (
    PromptMessageExtended,
    RequestParams,
    audio_link,
    image_link,
    resource_link,
    video_link,
)


def test_convert_function_results_to_google_text_only():
    converter = GoogleConverter()

    # Create a simple text-only tool result
    result = CallToolResult(
        content=[TextContent(type="text", text="Weather is sunny")], is_error=False
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


def test_google_tools_preserve_mcp_json_schema():
    # Recursive tool parameters cannot be flattened into an inline Schema.
    schema = {
        "$defs": {
            "node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/node"}},
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/node"}},
    }
    (tool,) = GoogleConverter().convert_to_google_tools([Tool(name="tree", input_schema=schema)])
    assert tool.function_declarations
    declaration = tool.function_declarations[0]
    assert declaration.parameters_json_schema == schema
    assert declaration.parameters is None


def test_convert_from_google_content_handles_missing_content_parts():
    converter = GoogleConverter()

    assert converter.convert_from_google_content(None) == []
    assert converter.convert_from_google_content(types.Content(role="model", parts=None)) == []


def test_convert_video_resource():
    converter = GoogleConverter()

    # Create a mock video resource
    video_bytes = b"fake_video_bytes"
    encoded_video = base64.b64encode(video_bytes).decode("utf-8")

    resource = EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri="file:///path/to/video.mp4", mime_type="video/mp4", blob=encoded_video
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
            uri="file:///video.mp4", mime_type="video/mp4", blob=encoded_video
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
            uri="file:///audio.mp3",
            mime_type="audio/mpeg",
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
            uri="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            mime_type="video/mp4",
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

    result = CallToolResult(content=[link], is_error=False)

    contents = converter.convert_function_results_to_google(
        [("video_generator", "call_video", result)]
    )

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
            uri="file:///video.mp4",
            mime_type="video/mp4",
            blob=encoded_video,
        ),
    )
    result = CallToolResult(content=[resource], is_error=False)

    contents = converter.convert_function_results_to_google(
        [("attach_media", "call_video", result)]
    )

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
            uri="file:///audio.mp3",
            mime_type="audio/mpeg",
            blob=encoded_audio,
        ),
    )
    result = CallToolResult(content=[resource], is_error=False)

    contents = converter.convert_function_results_to_google(
        [("attach_media", "call_audio", result)]
    )

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

    result = CallToolResult(content=[link], is_error=False)

    contents = converter.convert_function_results_to_google(
        [("config_reader", "call_config", result)]
    )

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


@pytest.mark.parametrize("model", ["gemini-3.7-flash", "gemini-3.8-flash"])
def test_current_flash_uses_level_only_config_and_omits_candidate_count(model: str) -> None:
    converter = GoogleConverter()
    params = RequestParams(
        model=model,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
    )

    config = converter.convert_request_params_to_google_config(
        params,
        thinking_budget=0,
    )

    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.candidate_count is None
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == "MEDIUM"
    assert config.thinking_config.thinking_budget is None


@pytest.mark.parametrize(
    ("tool_name", "tool_call_id", "message"),
    [
        ("", "call_123", "function name"),
        ("lookup", "", "function call ID"),
    ],
)
def test_google_function_response_requires_name_and_call_id(
    tool_name: str,
    tool_call_id: str,
    message: str,
) -> None:
    converter = GoogleConverter()
    result = CallToolResult(content=[TextContent(type="text", text="result")])

    with pytest.raises(ValueError, match=message):
        converter.convert_function_results_to_google([(tool_name, tool_call_id, result)])


def test_convert_multiple_function_results_into_single_content():
    """Test that multiple tool results are combined into a single Content object."""
    converter = GoogleConverter()

    result1 = CallToolResult(content=[TextContent(type="text", text="Output 1")], is_error=False)
    result2 = CallToolResult(content=[TextContent(type="text", text="Output 2")], is_error=False)

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


@pytest.mark.parametrize(
    ("model", "default_level"),
    [
        ("gemini-4-carbon", types.ThinkingLevel.MEDIUM),
        ("gemini-4-argon", types.ThinkingLevel.MEDIUM),
        ("gemini-4-hydrogen", types.ThinkingLevel.HIGH),
        ("models/gemini-4-hydrogen", types.ThinkingLevel.HIGH),
        (
            "projects/p/locations/global/publishers/google/models/gemini-4-carbon",
            types.ThinkingLevel.MEDIUM,
        ),
    ],
)
@pytest.mark.parametrize("budget", [-1, 0, 8192])
def test_gemini4_uses_fixed_sampling_and_model_default_thinking(
    model: str, default_level: types.ThinkingLevel, budget: int
) -> None:
    config = GoogleConverter().convert_request_params_to_google_config(
        RequestParams(
            model=model,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=131072,
            system_prompt="Be concise.",
        ),
        thinking_budget=budget,
    )
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.candidate_count is None
    assert config.max_output_tokens == 131072
    assert config.system_instruction == "Be concise."
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == default_level
    assert config.thinking_config.thinking_budget is None


@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
def test_gemini4_explicit_thinking_level_takes_precedence_over_budget(level: str) -> None:
    config = GoogleConverter().convert_request_params_to_google_config(
        RequestParams(model="gemini-4-hydrogen"),
        thinking_level=level,
        thinking_budget=8192,
    )
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == types.ThinkingLevel(level.upper())
    assert config.thinking_config.thinking_budget is None
    assert config.thinking_config.include_thoughts is True


def test_gemini4_without_thinking_override_preserves_server_default() -> None:
    config = GoogleConverter().convert_request_params_to_google_config(
        RequestParams(model="gemini-4-hydrogen")
    )
    assert config.thinking_config is None
