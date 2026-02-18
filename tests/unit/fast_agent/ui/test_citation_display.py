import json

from mcp.types import TextContent

from fast_agent.constants import ANTHROPIC_CITATIONS_CHANNEL, ANTHROPIC_SERVER_TOOLS_CHANNEL
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ui.citation_display import (
    collect_citation_sources,
    render_sources_additional_text,
    render_sources_footer,
    web_tool_badges,
)


def test_collect_citation_sources_dedupes_by_normalized_url() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "title": "Example",
                            "url": "https://Example.com/path/",
                        }
                    ),
                ),
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "title": "Duplicate",
                            "url": "https://example.com/path",
                        }
                    ),
                ),
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "search_result_location",
                            "title": "No URL",
                            "source": "Search Index",
                        }
                    ),
                ),
            ]
        },
    )

    sources = collect_citation_sources(message)
    assert len(sources) == 2
    assert sources[0].index == 1
    assert sources[0].url == "https://example.com/path"
    assert sources[1].index == 2
    assert sources[1].url is None


def test_render_sources_footer_with_markdown_links() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "title": "Fast Agent",
                            "url": "https://fast-agent.ai",
                        }
                    ),
                )
            ]
        },
    )

    footer = render_sources_footer(message)
    assert footer is not None
    assert "Sources" in footer
    assert "- [1] [Fast Agent](https://fast-agent.ai/)" in footer


def test_render_sources_additional_text_multiline() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "title": "Fast Agent",
                            "url": "https://fast-agent.ai",
                        }
                    ),
                )
            ]
        },
    )

    rendered = render_sources_additional_text(message)
    assert rendered is not None
    assert "Sources" in rendered.plain
    assert "[1] Fast Agent" in rendered.plain
    assert "https://fast-agent.ai/" in rendered.plain


def test_web_tool_badges_count_server_tool_use_blocks() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_SERVER_TOOLS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "server_tool_use",
                            "id": "srv_1",
                            "name": "web_search",
                            "input": {"query": "a"},
                        }
                    ),
                ),
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "server_tool_use",
                            "id": "srv_2",
                            "name": "web_fetch",
                            "input": {"url": "https://example.com"},
                        }
                    ),
                ),
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "server_tool_use",
                            "id": "srv_3",
                            "name": "web_search",
                            "input": {"query": "b"},
                        }
                    ),
                ),
            ]
        },
    )

    assert web_tool_badges(message) == ["web_search x2", "web_fetch x1"]
