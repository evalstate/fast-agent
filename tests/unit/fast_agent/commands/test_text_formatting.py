from rich.text import Text

from fast_agent.commands.handlers._text_formatting import (
    _console_width,
    append_heading,
    append_indexed_current_line,
    append_indexed_name_line,
    append_revision_line,
    append_warning_line,
    append_wrapped_text,
    indexed_row,
    resolve_terminal_width,
    update_status_text,
)
from fast_agent.marketplace.update_status import ALL_MARKETPLACE_UPDATE_STATUSES


def test_append_heading_treats_markup_as_literal_text() -> None:
    content = Text()

    append_heading(content, "Bundle [draft] <name>")

    assert content.plain == "Bundle [draft] <name>\n\n"
    assert content.spans[0].style == "bold"


def test_append_heading_separates_from_existing_content() -> None:
    content = Text("existing\n")

    append_heading(content, "Next")

    assert content.plain == "existing\n\nNext\n\n"


def test_append_wrapped_text_accounts_for_indent_width() -> None:
    content = Text()

    append_wrapped_text(content, " ".join(["word"] * 18), indent="     ")

    lines = content.plain.splitlines()
    assert len(lines) == 2
    assert all(line.startswith("     ") for line in lines)
    assert all(len(line) <= 72 for line in lines)


def test_append_wrapped_text_omits_blank_text() -> None:
    content = Text()

    append_wrapped_text(content, "   ", indent="  ")

    assert content.plain == ""


def test_append_wrapped_text_keeps_minimum_width_for_large_indent() -> None:
    content = Text()

    append_wrapped_text(content, " ".join(["word"] * 10), indent=" " * 80)

    lines = content.plain.splitlines()
    assert len(lines) > 1
    assert all(line.startswith(" " * 80) for line in lines)
    assert all(len(line.strip()) <= 20 for line in lines)


def test_append_warning_line_uses_yellow_style() -> None:
    content = Text()

    append_warning_line(content, "No managed skills found.")

    assert content.plain == "No managed skills found."
    assert content.spans[0].style == "yellow"


def test_append_indexed_name_line_formats_shared_prefix() -> None:
    content = Text()

    append_indexed_name_line(content, 3, "alpha")

    assert content.plain == "[ 3] alpha\n"


def test_indexed_row_returns_shared_prefix_for_custom_rows() -> None:
    row = indexed_row(4)
    row.append("custom", style="green")

    assert row.plain == "[ 4] custom"


def test_append_indexed_current_line_adds_current_marker() -> None:
    content = Text()

    append_indexed_current_line(content, 2, "registry", is_current=True)

    assert content.plain == "[ 2] registry • current\n"


def test_append_revision_line_omits_missing_revisions() -> None:
    content = Text()

    append_revision_line(content, None, None, format_revision=lambda value: value or "?")

    assert content.plain == ""


def test_append_revision_line_omits_blank_revisions() -> None:
    content = Text()

    append_revision_line(content, "  ", "\t", format_revision=lambda value: value or "?")

    assert content.plain == ""


def test_append_revision_line_formats_revision_pair() -> None:
    content = Text()

    append_revision_line(
        content,
        "0123456789",
        "abcdef1234",
        format_revision=lambda value: (value or "?")[:7],
    )

    assert "revision: 0123456 -> abcdef1" in content.plain


def test_update_status_text_appends_detail_for_common_detail_status() -> None:
    status = update_status_text("source_unreachable", detail=" git\nls-remote  failed ")

    assert status.text == "source unreachable: git ls-remote failed"
    assert status.style == "yellow"


def test_update_status_text_omits_detail_for_non_detail_status() -> None:
    status = update_status_text("updated", detail="revision changed")

    assert status.text == "updated"
    assert status.style == "green"


def test_update_status_text_styles_domain_specific_failures() -> None:
    status = update_status_text("invalid_local_pack", detail="missing manifest")

    assert status.text == "invalid local pack: missing manifest"
    assert status.style == "yellow"


def test_update_status_text_styles_known_statuses_except_unmanaged() -> None:
    for status in ALL_MARKETPLACE_UPDATE_STATUSES:
        status_text = update_status_text(status)
        if status == "unmanaged":
            assert status_text.style is None
        else:
            assert status_text.style is not None


def test_update_status_text_accepts_extra_detail_statuses() -> None:
    status = update_status_text(
        "updated",
        detail="bad response",
        labels={"updated": "custom updated"},
        detail_statuses={"updated"},
    )

    assert status.text == "custom updated: bad response"


def test_console_width_returns_positive_console_width(monkeypatch) -> None:
    class _Size:
        width = 132

    class _Console:
        size = _Size()

    monkeypatch.setattr("fast_agent.ui.console.console", _Console())

    assert _console_width() == 132


def test_console_width_returns_zero_for_invalid_console_width(monkeypatch) -> None:
    class _Size:
        width = "wide"

    class _Console:
        size = _Size()

    monkeypatch.setattr("fast_agent.ui.console.console", _Console())

    assert _console_width() == 0


def test_resolve_terminal_width_uses_terminal_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.commands.handlers._text_formatting._console_width",
        lambda: 0,
    )

    class _TerminalSize:
        columns = 88

    monkeypatch.setattr(
        "fast_agent.commands.handlers._text_formatting.get_terminal_size",
        lambda fallback: _TerminalSize(),
    )

    assert resolve_terminal_width() == 88
