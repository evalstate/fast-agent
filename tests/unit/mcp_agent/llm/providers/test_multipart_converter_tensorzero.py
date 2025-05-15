from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from mcp_agent.llm.providers.multipart_converter_tensorzero import TensorZeroConverter
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def test_convert_mcp_user_text_message():
    mcp_msg = PromptMessageMultipart(role="user", content=[TextContent(type="text", text="Hi.")])
    expected_t0_msg = {"role": "user", "content": [{"type": "text", "text": "Hi."}]}
    assert TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg) == expected_t0_msg


def test_convert_mcp_assistant_text_message():
    mcp_msg = PromptMessageMultipart(
        role="assistant", content=[TextContent(type="text", text="Hello there!")]
    )
    expected_t0_msg = {"role": "assistant", "content": [{"type": "text", "text": "Hello there!"}]}
    assert TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg) == expected_t0_msg


def test_convert_tool_results_to_t0_user_message():
    tool_result_1 = CallToolResult(content=[TextContent(type="text", text="!dlrow ,olleH")])
    setattr(tool_result_1, "_t0_tool_use_id_temp", "toolu_01")
    setattr(tool_result_1, "_t0_tool_name_temp", "tester-example_tool")

    tool_result_2 = CallToolResult(content=[TextContent(type="text", text='{"status": "ok"}')])
    setattr(tool_result_2, "_t0_tool_use_id_temp", "toolu_02")
    setattr(tool_result_2, "_t0_tool_name_temp", "another_tool")

    results = [tool_result_1, tool_result_2]
    expected_t0_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "id": "toolu_01",
                "name": "tester-example_tool",
                "result": "!dlrow ,olleH",  # String result
            },
            {
                "type": "tool_result",
                "id": "toolu_02",
                "name": "another_tool",
                "result": {"status": "ok"},  # JSON object result
            },
        ],
    }
    assert TensorZeroConverter.convert_tool_results_to_t0_user_message(results) == expected_t0_msg
    assert not hasattr(tool_result_1, "_t0_tool_use_id_temp")
    assert not hasattr(tool_result_2, "_t0_tool_use_id_temp")


def test_convert_tool_results_empty_list():
    assert TensorZeroConverter.convert_tool_results_to_t0_user_message([]) is None


def test_convert_mcp_image_message():
    """Test conversion of ImageContent."""
    # Valid PNG
    img_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    mcp_msg_png = PromptMessageMultipart(
        role="user", content=[ImageContent(type="image", mimeType="image/png", data=img_data)]
    )
    expected_png = {
        "role": "user",
        "content": [{"type": "image", "mime_type": "image/png", "data": img_data}],
    }
    assert TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg_png) == expected_png

    # Valid JPEG
    mcp_msg_jpeg = PromptMessageMultipart(
        role="user",
        content=[ImageContent(type="image", mimeType="image/jpeg", data="/9j/...")],
    )
    expected_jpeg = {
        "role": "user",
        "content": [{"type": "image", "mime_type": "image/jpeg", "data": "/9j/..."}],
    }
    assert TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg_jpeg) == expected_jpeg

    # Unsupported mime type (should default to png)
    mcp_msg_unsupported = PromptMessageMultipart(
        role="user", content=[ImageContent(type="image", mimeType="image/gif", data=img_data)]
    )
    expected_unsupported = {
        "role": "user",
        "content": [
            {"type": "image", "mime_type": "image/png", "data": img_data}
        ],  # Defaults to png
    }
    assert (
        TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg_unsupported) == expected_unsupported
    )

    # Missing data (should skip)
    mcp_msg_no_data = PromptMessageMultipart(
        role="user", content=[ImageContent(type="image", mimeType="image/png", data="")]
    )
    assert TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg_no_data) is None

    # Missing mimeType (should skip)
    mcp_msg_no_mime = PromptMessageMultipart(
        role="user",
        content=[ImageContent(type="image", mimeType="", data=img_data)],  # Empty mimeType
    )
    assert TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg_no_mime) is None


def test_convert_mcp_embedded_resource_skipped():
    """Test that EmbeddedResource is skipped."""
    # Create a valid dummy URL for TextResourceContents
    url_str = "https://example.com/resource"
    # Create minimal required fields for EmbeddedResource test
    dummy_resource = TextResourceContents(text="dummy content", uri=AnyUrl(url_str))
    mcp_msg = PromptMessageMultipart(
        role="user",
        content=[EmbeddedResource(type="resource", resource=dummy_resource)],
    )
    assert TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg) is None


def test_convert_mcp_mixed_content():
    """Test conversion of message with mixed Text and Image content."""
    img_data = "base64data"
    mcp_msg = PromptMessageMultipart(
        role="user",
        content=[
            TextContent(type="text", text="Describe this:"),
            ImageContent(type="image", mimeType="image/png", data=img_data),
            TextContent(type="text", text="and this text."),
        ],
    )
    expected_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this:"},
            {"type": "image", "mime_type": "image/png", "data": img_data},
            {"type": "text", "text": "and this text."},
        ],
    }
    assert TensorZeroConverter.convert_mcp_to_t0_message(mcp_msg) == expected_msg
