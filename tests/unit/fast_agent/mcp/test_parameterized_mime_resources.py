from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from mcp_types import EmbeddedResource, TextResourceContents

from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.mcp import mime_utils
from fast_agent.mcp.mcp_content import MCPFile
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.mark.parametrize(
    "mime_type",
    [
        "application/xml",
        "application/xml; charset=utf-8",
        ' APPLICATION/XML ; charset="UTF-8" ',
        "application/atom+xml; charset=utf-8",
    ],
)
def test_xml_attachment_preserves_text_for_provider(tmp_path: Path, mime_type: str) -> None:
    path = tmp_path / "document.xml"
    text = "<document>Source material: café</document>"
    path.write_text(text, encoding="utf-8")

    attachment = MCPFile(path, mime_type=mime_type)["content"]

    assert isinstance(attachment, EmbeddedResource)
    assert isinstance(attachment.resource, TextResourceContents)
    assert attachment.resource.text == text
    assert attachment.resource.mime_type == mime_type

    converted = AnthropicConverter.convert_to_anthropic(
        PromptMessageExtended(role="user", content=[attachment])
    )
    blocks = converted["content"]
    assert not isinstance(blocks, str)
    block = cast("Mapping[str, object]", list(blocks)[0])
    assert block["type"] == "document"
    assert block["source"] == {"type": "text", "media_type": "text/plain", "data": text}


def test_parameterized_mime_classification_preserves_media_categories() -> None:
    assert mime_utils.is_text_mime_type("application/json; charset=utf-8")
    assert mime_utils.is_text_mime_type("image/svg+xml; charset=utf-8")
    assert not mime_utils.is_image_mime_type("image/svg+xml; charset=utf-8")
    assert mime_utils.is_document_mime_type("application/pdf; version=1.7")
    assert mime_utils.normalize_mime_type(' IMAGE/JPG ; name="photo;2026.jpg" ') == "image/jpeg"
    assert mime_utils.is_binary_content("application/octet-stream; name=data.bin")
