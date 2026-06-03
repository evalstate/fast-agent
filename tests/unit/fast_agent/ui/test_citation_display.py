import json
from dataclasses import dataclass

from mcp.types import TextContent

from fast_agent.constants import ANTHROPIC_CITATIONS_CHANNEL, ANTHROPIC_SERVER_TOOLS_CHANNEL
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ui.citation_display import (
    collect_citation_sources,
    render_sources_additional_text,
    render_sources_footer,
    web_tool_badges,
)


@dataclass
class _NonStringTextBlock:
    text: object


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
                            "url": "HTTPS://EXAMPLE.COM:443/path",
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


def test_collect_citation_sources_dedupes_metadata_with_trimmed_casefolded_keys() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "search_result_location",
                            "title": " Search Result ",
                            "source": " Index ",
                        }
                    ),
                ),
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "search_result_location",
                            "title": "search result",
                            "source": "index",
                        }
                    ),
                ),
            ]
        },
    )

    sources = collect_citation_sources(message)

    assert len(sources) == 1
    assert sources[0].title == " Search Result "
    assert sources[0].source == " Index "


def test_collect_citation_sources_skips_malformed_payload_blocks() -> None:
    message = PromptMessageExtended.model_construct(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(type="text", text="{"),
                _NonStringTextBlock({"not": "json text"}),
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "title": "Valid",
                            "url": "https://example.com",
                        }
                    ),
                ),
            ]
        },
    )

    sources = collect_citation_sources(message)

    assert len(sources) == 1
    assert sources[0].display_title == "Valid"


def test_collect_citation_sources_keeps_invalid_urls_unmodified() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "title": "Bad port",
                            "url": "https://example.com:bad/path",
                        }
                    ),
                )
            ]
        },
    )

    sources = collect_citation_sources(message)

    assert len(sources) == 1
    assert sources[0].url == "https://example.com:bad/path"


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


def test_render_sources_footer_uses_source_as_title_fallback() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "source": "Search Index",
                            "url": "https://example.com/item",
                        }
                    ),
                )
            ]
        },
    )

    footer = render_sources_footer(message)

    assert footer is not None
    assert "- [1] [Search Index](https://example.com/item)" in footer


def test_render_sources_footer_escapes_markdown_link_labels() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "title": r"A [B] \ C",
                            "url": "https://example.com/item",
                        }
                    ),
                )
            ]
        },
    )

    footer = render_sources_footer(message)

    assert footer is not None
    assert r"- [1] [A \[B\] \\ C](https://example.com/item)" in footer


def test_render_sources_footer_escapes_markdown_link_destinations() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "web_search_result_location",
                            "title": "Escaped URL",
                            "url": r"https://example.com/a(b)\c",
                        }
                    ),
                )
            ]
        },
    )

    footer = render_sources_footer(message)

    assert footer is not None
    assert r"- [1] [Escaped URL](https://example.com/a\(b\)\\c)" in footer


def test_render_sources_footer_escapes_markdown_plain_titles() -> None:
    message = PromptMessageExtended(
        role="assistant",
        channels={
            ANTHROPIC_CITATIONS_CHANNEL: [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "type": "search_result_location",
                            "title": "A [B]",
                            "source": "Search Index",
                        }
                    ),
                )
            ]
        },
    )

    footer = render_sources_footer(message)

    assert footer is not None
    assert r"- [1] A \[B\]" in footer


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
