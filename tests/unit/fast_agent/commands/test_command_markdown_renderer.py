from __future__ import annotations

from rich.text import Text

from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.results import CommandMessage, CommandOutcome, is_command_channel


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


def test_render_command_outcome_markdown_escapes_plain_message_text() -> None:
    outcome = CommandOutcome()
    outcome.add_message("Use [docs](bad) and *care*", channel="warning")

    rendered = render_command_outcome_markdown(outcome, heading="cards add")

    assert "**Warning:** Use \\[docs\\](bad) and \\*care\\*" in rendered
    assert "[docs](bad)" not in rendered


def test_render_command_outcome_markdown_preserves_explicit_markdown_messages() -> None:
    outcome = CommandOutcome()
    outcome.add_message("[docs](https://example.com)", render_markdown=True)

    rendered = render_command_outcome_markdown(outcome, heading="cards add")

    assert "[docs](https://example.com)" in rendered


def test_render_command_outcome_markdown_includes_message_title() -> None:
    outcome = CommandOutcome()
    outcome.add_message("details", title="Result")

    rendered = render_command_outcome_markdown(outcome, heading="cards add")

    assert "## Result\n\ndetails" in rendered


def test_render_command_outcome_markdown_skips_blank_message_text() -> None:
    outcome = CommandOutcome()
    outcome.add_message("   ")
    outcome.add_message("visible")

    rendered = render_command_outcome_markdown(outcome, heading="cards add")

    assert rendered == "# cards add\n\nvisible"


def test_render_command_outcome_markdown_keeps_title_only_message() -> None:
    outcome = CommandOutcome()
    outcome.add_message("   ", title="Result")

    rendered = render_command_outcome_markdown(outcome, heading="cards add")

    assert rendered == "# cards add\n\n## Result"


def test_render_command_outcome_markdown_normalizes_message_title() -> None:
    outcome = CommandOutcome()
    outcome.add_message("details", title="## Result\nInjected")

    rendered = render_command_outcome_markdown(outcome, heading="cards add")

    assert "## Result Injected\n\ndetails" in rendered
    assert "\nInjected\n" not in rendered


def test_render_command_outcome_markdown_escapes_message_title() -> None:
    outcome = CommandOutcome()
    outcome.add_message("details", title="Result_[draft]*")

    rendered = render_command_outcome_markdown(outcome, heading="cards add")

    assert "## Result\\_\\[draft\\]\\*\n\ndetails" in rendered
    assert "## Result_[draft]*" not in rendered


def test_render_command_outcome_markdown_normalizes_heading() -> None:
    outcome = CommandOutcome()
    outcome.add_message("details")

    rendered = render_command_outcome_markdown(outcome, heading="## cards\nadd")

    assert rendered.startswith("# cards add\n")
    assert not rendered.startswith("# ##")


def test_render_command_outcome_markdown_escapes_heading() -> None:
    outcome = CommandOutcome()
    outcome.add_message("details")

    rendered = render_command_outcome_markdown(outcome, heading="cards_[draft]*")

    assert rendered.startswith("# cards\\_\\[draft\\]\\*\n")
    assert not rendered.startswith("# cards_[draft]*")


def test_command_message_plain_text_reads_rich_text_content() -> None:
    message = CommandMessage(Text("plain", style="bold"))

    assert message.plain_text() == "plain"


def test_command_outcome_add_message_copies_metadata() -> None:
    metadata: dict[str, object] = {"status": "connected"}
    outcome = CommandOutcome()

    outcome.add_message("connected", metadata=metadata)
    metadata["status"] = "mutated"

    assert outcome.messages[0].metadata == {"status": "connected"}


def test_is_command_channel_narrows_known_channels() -> None:
    class WarningLike:
        def __eq__(self, other: object) -> bool:
            return other == "warning"

    assert is_command_channel("warning")
    assert not is_command_channel("debug")
    assert not is_command_channel(WarningLike())
