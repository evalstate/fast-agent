from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from prompt_toolkit.document import Document
from rich.console import Console

from fast_agent.ui.command_payloads import CommandError
from fast_agent.ui.prompt import special_commands
from fast_agent.ui.prompt.completer import AgentCompleter
from fast_agent.ui.prompt.parser import parse_special_input
from fast_agent.ui.prompt.special_commands import handle_special_commands

if TYPE_CHECKING:
    import pytest


def test_parse_help_status_topic() -> None:
    assert parse_special_input("/help") == "HELP"
    assert parse_special_input("/HELP STATUS") == "HELP:STATUS"

    invalid = parse_special_input("/help unknown")
    assert invalid == CommandError(message="Unexpected arguments for /help: unknown")


def test_help_status_renders_tree_legend(monkeypatch: pytest.MonkeyPatch) -> None:
    output_buffer = StringIO()
    output_console = Console(file=output_buffer, color_system=None)
    monkeypatch.setattr(special_commands, "rich_print", output_console.print)

    result = handle_special_commands(parse_special_input("/help status"))

    assert result is True
    output = output_buffer.getvalue()
    assert "Interactive Status Bar (left → right):" in output
    assert "├─ Activity" in output
    assert "├─ Model" in output
    assert "└─ Right side" in output
    for glyph in ("↻", "↳", "⌘", "T V D", "▲", "⣀…⣿", "∞", "▼", "»", "⊕", "⇣", "◀"):
        assert glyph in output
    normalized = " ".join(output.split())
    assert "∞<model> plan (OAuth login/monthly token plan)" in normalized
    assert "▼<model> overlay" in normalized


def test_help_sentinel_does_not_consume_user_text() -> None:
    assert handle_special_commands("HELP:anything") is False
    assert handle_special_commands("HELP:STATUS report") is False


def test_help_status_topic_is_completed() -> None:
    completer = AgentCompleter(agents=[])

    completions = list(
        completer.get_completions(
            Document("/help sta"),
            complete_event=None,
        )
    )

    assert [(completion.text, completion.display_meta_text) for completion in completions] == [
        ("status", "Explain the interactive status bar")
    ]
