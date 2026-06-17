from __future__ import annotations

from pathlib import Path

from rich.console import Console

from fast_agent.cli.display import (
    UpdateDisplayRow,
    indexed_table,
    print_hint,
    print_update_table,
    print_warning,
)


def test_print_warning_uses_standard_warning_style() -> None:
    console = Console(record=True, width=80)

    print_warning(console, "No managed skills found.")

    output = console.export_text(styles=True)
    assert "No managed skills found." in output
    assert "\x1b[33m" in output


def test_print_hint_treats_markup_as_literal_text() -> None:
    console = Console(record=True, width=80)

    print_hint(console, "Use [draft] registry")

    output = console.export_text(styles=True)
    assert "Use [draft] registry" in output
    assert "\x1b[2m" in output


def test_print_warning_treats_markup_as_literal_text() -> None:
    console = Console(record=True, width=80)

    print_warning(console, "Missing [draft] registry")

    output = console.export_text(styles=True)
    assert "Missing [draft] registry" in output
    assert "\x1b[33m" in output


def test_indexed_table_adds_standard_index_column() -> None:
    console = Console(record=True, width=80)
    table = indexed_table(("Name", "cyan"), ("Description", "dim"))
    table.add_row("1", "alpha", "first")

    console.print(table)

    output = console.export_text()
    assert "#  Name" in output
    assert "1  alpha" in output


def test_print_update_table_uses_shared_status_labels() -> None:
    console = Console(record=True, width=120)

    print_update_table(
        console,
        [
            UpdateDisplayRow(
                index=1,
                name="alpha",
                source_path=Path("/tmp/alpha"),
                current_revision=None,
                available_revision=None,
                status="up_to_date",
            ),
            UpdateDisplayRow(
                index=2,
                name="beta",
                source_path=Path("/tmp/beta"),
                current_revision=None,
                available_revision=None,
                status="source_unreachable",
                detail="git failed",
            ),
            UpdateDisplayRow(
                index=3,
                name="gamma",
                source_path=Path("/tmp/gamma"),
                current_revision=None,
                available_revision=None,
                status="invalid_local_skill",
                detail="missing SKILL.md",
            ),
        ],
        format_revision_short=lambda value: value or "-",
    )

    output = console.export_text()

    assert "already up to date" in output
    assert "source unreachable: git failed" in output
    assert "invalid local skill: missing SKILL.md" in output
