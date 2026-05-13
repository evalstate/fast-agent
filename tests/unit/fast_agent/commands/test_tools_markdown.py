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

    assert "## Provider-hosted tools" in rendered
    assert "**web_search** _(provider-hosted, enabled)_" in rendered
