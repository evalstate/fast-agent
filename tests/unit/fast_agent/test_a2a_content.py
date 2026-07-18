import base64

from mcp.types import AudioContent, EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from fast_agent.a2a.content import part_from_content


def test_part_from_content_converts_audio() -> None:
    part = part_from_content(
        AudioContent(
            type="audio",
            data=base64.b64encode(b"audio bytes").decode(),
            mimeType="audio/wav",
        )
    )

    assert part is not None
    assert part.raw == b"audio bytes"
    assert part.media_type == "audio/wav"


def test_part_from_content_preserves_text_resource_metadata() -> None:
    part = part_from_content(
        EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=AnyUrl("resource:///reports/final%20summary.txt"),
                mimeType="text/markdown",
                text="# Summary",
            ),
        )
    )

    assert part is not None
    assert part.text == "# Summary"
    assert part.media_type == "text/markdown"
    assert part.filename == "final summary.txt"
