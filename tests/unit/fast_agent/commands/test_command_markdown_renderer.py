from __future__ import annotations

from fast_agent.command_actions import MarkdownTextStyle
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.results import CommandMessage, CommandOutcome


def test_render_command_outcome_markdown_adds_heading_and_formats_channels() -> None:
    outcome = CommandOutcome()
    outcome.add_message("all good")
    outcome.add_message("watch this", channel="warning")
    outcome.add_message("failed", channel="error")

    rendered = render_command_outcome_markdown(outcome, heading="skills list")

    assert rendered.startswith("# skills list")
    assert "all good" in rendered
    assert "**Warning:** watch this" in rendered
    assert "**Error:** failed" in rendered


def test_render_command_outcome_markdown_includes_extra_messages() -> None:
    outcome = CommandOutcome()
    outcome.add_message("primary")

    rendered = render_command_outcome_markdown(
        outcome,
        heading="cards list",
        extra_messages=[CommandMessage(text="extra")],
    )

    assert "primary" in rendered
    assert "extra" in rendered


def test_render_command_outcome_markdown_fences_verbatim_source() -> None:
    source = "## Heading\n\n- **bold**\n\n```python\nprint('nested fence')\n```"
    outcome = CommandOutcome()
    outcome.add_message(source, title="Last Assistant Response", verbatim=True)

    rendered = render_command_outcome_markdown(outcome, heading="markdown")

    assert rendered == (f"# markdown\n\n## Last Assistant Response\n\n````\n{source}\n````")


def test_render_command_outcome_markdown_ignores_rich_presentation_styles() -> None:
    source = "| Cached |\n| ---: |\n| 400,000 (40%) |"
    outcome = CommandOutcome()
    outcome.add_message(
        source,
        render_markdown=True,
        markdown_styles=(MarkdownTextStyle(text="40%", style="red"),),
    )

    rendered = render_command_outcome_markdown(outcome, heading="cost")

    assert "400,000 (40%)" in rendered
    assert "[red]" not in rendered
    assert "\x1b" not in rendered
    assert "🔴" not in rendered
