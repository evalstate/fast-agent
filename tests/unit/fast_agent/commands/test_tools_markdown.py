from __future__ import annotations

from fast_agent.commands.renderers.tools_markdown import render_tools_markdown
from fast_agent.commands.tool_summaries import ProviderToolSummary


def test_render_tools_markdown_includes_provider_hosted_tools() -> None:
    rendered = render_tools_markdown(
        [],
        heading="tools",
        provider_summaries=[
            ProviderToolSummary(
                name="web_search",
                enabled=True,
                description="Provider-hosted web search tool.",
            )
        ],
    )

    assert "## Provider-managed / hosted tools" in rendered
    assert "**web_search** _(provider-hosted, enabled)_" in rendered


def test_render_tools_markdown_uses_provider_summary_suffix() -> None:
    rendered = render_tools_markdown(
        [],
        heading="tools",
        provider_summaries=[
            ProviderToolSummary(
                name="gmail/search_gmail",
                enabled=True,
                description="Gmail connector",
                suffix="provider-managed connector",
            )
        ],
    )

    assert "**gmail/search_gmail** _(provider-managed connector, enabled)_" in rendered


def test_render_tools_markdown_marks_unknown_provider_tool_state() -> None:
    rendered = render_tools_markdown(
        [],
        heading="tools",
        provider_summaries=[
            ProviderToolSummary(
                name="provider_managed_mcp",
                enabled=None,
                description="Provider-managed MCP state is unavailable.",
                suffix="provider-managed MCP",
            )
        ],
    )

    assert "**provider_managed_mcp** _(provider-managed MCP, Unknown)_" in rendered
