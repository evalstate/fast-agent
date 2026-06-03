from __future__ import annotations

from fast_agent.commands.session_export_help import SESSION_EXPORT_USAGE
from fast_agent.ui.prompt.command_help import render_help_lines


def test_render_help_lines_uses_catalogued_command_actions() -> None:
    rendered = "\n".join(render_help_lines(show_webclear_help=False))

    assert "  /skills search <query>" in rendered
    assert "  /cards readme [<number|name>]" in rendered
    assert "  /plugins available" in rendered
    assert "  /model catalog <provider> [--all]" in rendered
    assert "  /models catalog <provider> [--all]" in rendered
    assert "  /check [args]" in rendered


def test_render_help_lines_uses_shared_session_export_usage() -> None:
    rendered = "\n".join(render_help_lines(show_webclear_help=False))

    assert f"  {SESSION_EXPORT_USAGE} - Export a session trace" in rendered
    assert "--privacy-filter-path" in rendered
    assert "--download-privacy-filter" in rendered
    assert "--privacy-filter-device" in rendered
    assert "--show-redactions" in rendered
