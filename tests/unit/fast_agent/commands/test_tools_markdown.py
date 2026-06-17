from __future__ import annotations

from fast_agent.commands.renderers.tools_markdown import render_tools_markdown
from fast_agent.commands.tool_summaries import (
    PROVIDER_HOSTED_SUFFIX,
    PROVIDER_MANAGED_CONNECTOR_SUFFIX,
    PROVIDER_MANAGED_MCP_SUFFIX,
    ProviderToolSummary,
    ToolSummary,
)


def test_render_tools_markdown_includes_local_tool_details() -> None:
    rendered = render_tools_markdown(
        [
            ToolSummary(
                name="read_text_file",
                title="Read text",
                description="Read a UTF-8 file from disk.",
                args=["path", "limit"],
                suffix="(Shell)",
                template="ui://file-reader",
            )
        ],
        heading="tools",
    )

    assert "1. **read\\_text\\_file** _(Shell)_ — Read text" in rendered
    assert "    > **Args:** `path`, `limit`" in rendered
    assert "    > **Template:** `ui://file-reader`" in rendered
    assert "    > \n    > **Args:**" not in rendered


def test_render_tools_markdown_handles_backticks_in_code_span_values() -> None:
    rendered = render_tools_markdown(
        [
            ToolSummary(
                name="dynamic_tool",
                title=None,
                description="Tool with unusual schema values.",
                args=["path`name"],
                suffix=None,
                template="ui://file`reader",
            )
        ],
        heading="tools",
    )

    assert "    > **Args:** `` path`name ``" in rendered
    assert "    > **Template:** `` ui://file`reader ``" in rendered


def test_render_tools_markdown_trims_args_and_omits_blank_args() -> None:
    rendered = render_tools_markdown(
        [
            ToolSummary(
                name="dynamic_tool",
                title=None,
                description="Tool with padded schema values.",
                args=[" path ", "   ", "\tlimit\n"],
                suffix=None,
                template=None,
            )
        ],
        heading="tools",
    )

    assert "    > **Args:** `path`, `limit`" in rendered
    assert "`   `" not in rendered


def test_render_tools_markdown_trims_template_and_omits_blank_template() -> None:
    rendered = render_tools_markdown(
        [
            ToolSummary(
                name="with_template",
                title=None,
                description="Tool with padded template.",
                args=None,
                suffix=None,
                template=" ui://file-reader ",
            ),
            ToolSummary(
                name="blank_template",
                title=None,
                description="Tool with blank template.",
                args=None,
                suffix=None,
                template="   ",
            ),
        ],
        heading="tools",
    )

    assert "    > **Template:** `ui://file-reader`" in rendered
    assert "`   `" not in rendered


def test_render_tools_markdown_escapes_external_tool_metadata() -> None:
    rendered = render_tools_markdown(
        [
            ToolSummary(
                name="read_[file]",
                title="Use *carefully*",
                description="See [docs](bad) and *bold*.",
                args=None,
                suffix="(MCP)_x",
                template=None,
            )
        ],
        heading="tools",
    )

    assert "1. **read\\_\\[file\\]** _(MCP)\\_x_ — Use \\*carefully\\*" in rendered
    assert "See \\[docs\\](bad) and \\*bold\\*." in rendered
    assert "See [docs](bad)" not in rendered


def test_render_tools_markdown_trims_header_title_and_omits_blank_suffix() -> None:
    rendered = render_tools_markdown(
        [
            ToolSummary(
                name="read_text_file",
                title=" Read text ",
                description="Read a UTF-8 file from disk.",
                args=None,
                suffix="   ",
                template=None,
            )
        ],
        heading="tools",
    )

    assert "1. **read\\_text\\_file** — Read text" in rendered
    assert "   _" not in rendered


def test_render_tools_markdown_normalizes_markdown_heading() -> None:
    rendered = render_tools_markdown([], heading="# tools")

    assert rendered.startswith("# tools")
    assert not rendered.startswith("# #")


def test_render_tools_markdown_escapes_heading() -> None:
    rendered = render_tools_markdown([], heading="tools_[draft]*")

    assert rendered == "# tools\\_\\[draft\\]\\*"


def test_render_tools_markdown_truncates_long_description_blockquote() -> None:
    rendered = render_tools_markdown(
        [
            ToolSummary(
                name="verbose_tool",
                title=None,
                description=" ".join(["description"] * 80),
                args=None,
                suffix=None,
                template=None,
            )
        ],
        heading="tools",
    )

    assert rendered.count("    > ") == 5
    assert "    > …" in rendered


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
    assert f"**web\\_search** _({PROVIDER_HOSTED_SUFFIX}, enabled)_" in rendered


def test_render_tools_markdown_uses_provider_summary_suffix() -> None:
    rendered = render_tools_markdown(
        [],
        heading="tools",
        provider_summaries=[
            ProviderToolSummary(
                name="gmail/search_gmail",
                enabled=True,
                description="Gmail connector",
                suffix=PROVIDER_MANAGED_CONNECTOR_SUFFIX,
            )
        ],
    )

    assert f"**gmail/search\\_gmail** _({PROVIDER_MANAGED_CONNECTOR_SUFFIX}, enabled)_" in rendered


def test_render_tools_markdown_marks_disabled_provider_tool_state() -> None:
    rendered = render_tools_markdown(
        [],
        heading="tools",
        provider_summaries=[
            ProviderToolSummary(
                name="web_fetch",
                enabled=False,
                description="Provider-hosted web fetch tool.",
            )
        ],
    )

    assert f"**web\\_fetch** _({PROVIDER_HOSTED_SUFFIX}, disabled)_" in rendered


def test_render_tools_markdown_marks_unknown_provider_tool_state() -> None:
    rendered = render_tools_markdown(
        [],
        heading="tools",
        provider_summaries=[
            ProviderToolSummary(
                name="provider_managed_mcp",
                enabled=None,
                description="Provider-managed MCP state is unavailable.",
                suffix=PROVIDER_MANAGED_MCP_SUFFIX,
            )
        ],
    )

    assert f"**provider\\_managed\\_mcp** _({PROVIDER_MANAGED_MCP_SUFFIX}, Unknown)_" in rendered


def test_render_tools_markdown_escapes_provider_tool_metadata() -> None:
    rendered = render_tools_markdown(
        [],
        heading="tools",
        provider_summaries=[
            ProviderToolSummary(
                name="gmail/search_[mail]",
                enabled=True,
                description="Connector *beta*",
                suffix=PROVIDER_MANAGED_CONNECTOR_SUFFIX,
            )
        ],
    )

    assert "**gmail/search\\_\\[mail\\]**" in rendered
    assert "Connector \\*beta\\*" in rendered
