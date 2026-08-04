from __future__ import annotations

from dataclasses import dataclass

from mcp_types import EmbeddedResource, TextResourceContents

from fast_agent.mcp.mcp_content import MCPPrompt


@dataclass
class _ResourceLike:
    type: str
    resource: object


def test_mcp_prompt_accepts_resource_like_object_without_dynamic_probe() -> None:
    resource = TextResourceContents(
        uri="file:///tmp/example.txt",
        text="example",
        mime_type="text/plain",
    )

    messages = MCPPrompt(_ResourceLike(type="resource", resource=resource))

    assert len(messages) == 1
    content = messages[0]["content"]
    assert isinstance(content, EmbeddedResource)
    assert content.resource is resource
