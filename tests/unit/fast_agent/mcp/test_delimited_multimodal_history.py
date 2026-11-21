"""Round-trip supported multimodal blocks through saved conversation files."""

from pathlib import Path

import pytest
from mcp_types import (
    Annotations,
    AudioContent,
    ContentBlock,
    ImageContent,
    ResourceLink,
    TextContent,
)

from fast_agent.mcp.message_roles import MessageRole
from fast_agent.mcp.prompt_serialization import load_messages, save_messages
from fast_agent.types import PromptMessageExtended


@pytest.fixture(params=["audio", "resource-link", "mixed"])
def multimodal_content(request: pytest.FixtureRequest) -> list[ContentBlock]:
    audio = AudioContent(
        type="audio",
        data="YXVkaW8=",
        mime_type="audio/wav",
        annotations=Annotations(audience=["user"], priority=0.5),
        meta={"source": "microphone"},
    )
    link = ResourceLink(
        type="resource_link",
        name="transcript.txt",
        uri="resource://recording/transcript",
        description="Recording transcript",
        mime_type="text/plain",
        size=12,
        annotations=Annotations(priority=0.7),
        meta={"recording": "sample"},
    )
    if request.param == "audio":
        return [audio]
    if request.param == "resource-link":
        return [link]
    return [
        TextContent(type="text", text="Recording and transcript:"),
        audio,
        ImageContent(type="image", data="aW1hZ2U=", mime_type="image/png"),
        link,
    ]


@pytest.mark.parametrize("suffix", [".json", ".md"])
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_saved_history_preserves_multimodal_content(
    tmp_path: Path, multimodal_content: list[ContentBlock], suffix: str, role: MessageRole
) -> None:
    messages = [
        PromptMessageExtended(role=role, content=multimodal_content),
        PromptMessageExtended(
            role="assistant", content=[TextContent(type="text", text="Follow-up message")]
        ),
    ]
    target = str(tmp_path / f"history{suffix}")

    save_messages(messages, target)
    restored = load_messages(target)

    assert len(restored) == len(messages)
    for actual, expected in zip(restored, messages, strict=True):
        assert actual.role == expected.role
        assert actual.content == expected.content
