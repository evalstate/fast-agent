import base64
import json
from io import BytesIO

import pytest
from mcp.types import CallToolResult, ImageContent
from PIL import Image

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.provider.openai.llm_deepseek import DeepSeekResponsesLLM
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def image_content(size: tuple[int, int], image_format: str = "PNG") -> ImageContent:
    output = BytesIO()
    with Image.new("RGB", size, (40, 100, 180)) as image:
        image.save(output, format=image_format)
    return ImageContent(
        type="image",
        mime_type=Image.MIME[image_format],
        data=base64.b64encode(output.getvalue()).decode("ascii"),
    )


def llm() -> DeepSeekResponsesLLM:
    return DeepSeekResponsesLLM(context=Context(config=Settings()), model="deepseek-flash")


@pytest.mark.parametrize("size", [(8268, 4092), (4092, 8268)])
def test_oversize_image_resizes_at_request_boundary(size: tuple[int, int]) -> None:
    provider = llm()
    image = image_content(size)
    items = provider._convert_extended_messages_to_provider(
        [PromptMessageExtended(role="user", content=[image])]
    )
    before = json.dumps(items)
    args = provider._build_response_args(items, provider.default_request_params, None)
    url = json.loads(json.dumps(args))["input"][0]["content"][0]["image_url"]
    assert url.startswith("data:image/png;base64,")
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as resized:
        assert max(resized.size) == 8192
        assert abs(resized.width / resized.height - size[0] / size[1]) < 0.001
        assert resized.getpixel((100, 100)) == (40, 100, 180)
    assert json.dumps(items) == before


@pytest.mark.parametrize("count", [14, 15])
@pytest.mark.parametrize("oversize_in_tool", [False, True])
def test_threshold_counts_retained_and_tool_output_images(
    count: int, oversize_in_tool: bool
) -> None:
    provider = llm()
    retained = image_content((5000, 100))
    small = image_content((2, 2))
    user_image = small if oversize_in_tool else retained
    tool_images = [retained if oversize_in_tool else small] + [small] * (count - 2)
    messages = [
        PromptMessageExtended(role="user", content=[user_image]),
        PromptMessageExtended(
            role="user",
            tool_results={"call_image": CallToolResult(content=tool_images)},
        ),
    ]
    items = provider._convert_extended_messages_to_provider(messages)
    assert items[1]["type"] == "function_call_output"
    args = provider._build_response_args(items, provider.default_request_params, None)
    part = args["input"][1]["output"][0] if oversize_in_tool else args["input"][0]["content"][0]
    url = part["image_url"]
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as resized:
        assert resized.width == (5000 if count == 14 else 4096)
    unchanged_index = 0 if oversize_in_tool else 1
    assert args["input"][unchanged_index] == items[unchanged_index]
    assert messages[0].content == [user_image]


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP", "GIF"])
def test_within_bounds_preserves_mime_bytes_and_pixels(image_format: str) -> None:
    provider = llm()
    image = image_content((32, 17), image_format)
    items = provider._convert_extended_messages_to_provider(
        [PromptMessageExtended(role="user", content=[image])]
    )
    args = provider._build_response_args(items, provider.default_request_params, None)
    wire = json.loads(json.dumps(args))["input"]
    assert wire == items
    assert wire[0]["content"][0]["image_url"] == f"data:{image.mime_type};base64,{image.data}"


def test_remote_and_file_ids_count_without_fetching_or_changing() -> None:
    provider = llm()
    image = image_content((5000, 100))
    items = provider._convert_extended_messages_to_provider(
        [PromptMessageExtended(role="user", content=[image])]
    )
    references = [
        {"type": "input_image", "image_url": "https://invalid.example/private-image"},
        {"type": "input_image", "file_id": "file-private"},
    ] * 7
    items.append({"role": "user", "content": references})
    args = provider._build_response_args(items, provider.default_request_params, None)
    assert args["input"][1]["content"] == references
    url = args["input"][0]["content"][0]["image_url"]
    with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as resized:
        assert resized.width == 4096


def test_invalid_inline_image_has_payload_free_actionable_error() -> None:
    provider = llm()
    items = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64,PRIVATE"}],
        }
    ]
    with pytest.raises(ValueError, match="8192px per side") as error:
        provider._build_response_args(items, provider.default_request_params, None)
    assert "PRIVATE" not in str(error.value)
