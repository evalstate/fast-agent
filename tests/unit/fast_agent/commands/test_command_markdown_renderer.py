from __future__ import annotations

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
