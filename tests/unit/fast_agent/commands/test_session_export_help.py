from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands import session_export_help
from fast_agent.commands.session_export_help import render_session_export_help_markdown

if TYPE_CHECKING:
    import pytest


def test_render_session_export_help_preserves_standard_labels() -> None:
    rendered = render_session_export_help_markdown()

    assert "Usage: `/session export" in rendered
    assert "- `target` (`latest|id|path`) — Session target" in rendered
    assert "- `--output path`, `-o` — Write trace" in rendered
    assert "- `/session export latest --output trace.jsonl`" in rendered


def test_session_export_help_labels_escape_backticks_in_code_spans() -> None:
    argument_label = session_export_help._argument_label(
        {
            "name": "target`name",
            "required": False,
            "value_name": "latest`path",
            "summary": "",
        }
    )
    option_label = session_export_help._option_label(
        {
            "name": "--output`path",
            "aliases": ["-o`"],
            "value_name": "file`path",
            "summary": "",
        }
    )

    assert argument_label == "`` target`name `` (`` latest`path ``)"
    assert option_label == "`` --output`path file`path ``, `` -o` ``"


def test_session_export_help_labeled_line_helper_skips_and_escapes() -> None:
    lines: list[str] = []

    session_export_help._append_labeled_help_line(
        lines,
        label=None,
        summary="ignored",
    )
    session_export_help._append_labeled_help_line(
        lines,
        label="`target`",
        summary=" Use [docs](bad) and *care* ",
    )
    session_export_help._append_labeled_help_line(
        lines,
        label="`blank`",
        summary="   ",
    )

    assert lines == [
        "- `target` — Use \\[docs\\](bad) and \\*care\\*",
        "- `blank`",
    ]


def test_render_session_export_help_escapes_structured_prose(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    monkeypatch.setattr(
        session_export_help,
        "build_session_export_action_detail",
        lambda: {
            "name": "export",
            "summary": "",
            "usage": "/session export",
            "examples": [],
            "arguments": [
                {
                    "name": "target",
                    "required": False,
                    "value_name": None,
                    "summary": "Use [docs](bad) and *care*",
                }
            ],
            "options": [
                {
                    "name": "--output",
                    "aliases": [],
                    "value_name": "path",
                    "summary": "Write [docs](bad) with *care*",
                }
            ],
            "notes": ["Default [docs](bad) and *care*."],
        },
    )

    rendered = render_session_export_help_markdown()

    assert "- `target` — Use \\[docs\\](bad) and \\*care\\*" in rendered
    assert "- `--output path` — Write \\[docs\\](bad) with \\*care\\*" in rendered
    assert "- Default \\[docs\\](bad) and \\*care\\*." in rendered
    assert "[docs](bad)" not in rendered


def test_render_session_export_help_strips_and_skips_blank_notes_and_examples(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    monkeypatch.setattr(
        session_export_help,
        "build_session_export_action_detail",
        lambda: {
            "name": "export",
            "summary": "",
            "usage": "/session export",
            "examples": [" /session export latest ", "   "],
            "arguments": [],
            "options": [],
            "notes": [" Note text ", "\t"],
        },
    )

    rendered = render_session_export_help_markdown()

    assert "- Note text" in rendered
    assert "- `/session export latest`" in rendered
    assert "- \t" not in rendered
    assert "`   `" not in rendered
