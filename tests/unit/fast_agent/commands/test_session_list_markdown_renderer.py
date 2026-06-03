from fast_agent.commands.renderers.session_markdown import render_session_list_markdown
from fast_agent.commands.session_summaries import SessionListSummary
from fast_agent.session import SessionEntrySummary


def _entry_summary(
    *,
    display_name: str,
    is_pinned: bool = False,
    index: int = 1,
) -> SessionEntrySummary:
    return SessionEntrySummary(
        index=index,
        display_name=display_name,
        is_current=False,
        is_pinned=is_pinned,
        timestamp="Jan 01 00:00",
    )


def test_render_session_list_markdown_highlights_pinned_session_name_once() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=["1. alpha - alpha"],
            entry_summaries=[_entry_summary(display_name="alpha", is_pinned=True)],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions",
    )

    assert "1. **alpha** - alpha" in rendered


def test_render_session_list_markdown_uses_summary_index_for_pinned_highlight() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=["2. alpha - Jan 01 00:00"],
            entry_summaries=[_entry_summary(index=2, display_name="alpha", is_pinned=True)],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions",
    )

    assert "2. **alpha** - Jan" in rendered


def test_render_session_list_markdown_strips_pinned_session_name_before_highlight() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=["1. alpha - Jan 01 00:00"],
            entry_summaries=[_entry_summary(display_name=" alpha ", is_pinned=True)],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions",
    )

    assert "1. **alpha** - Jan" in rendered


def test_render_session_list_markdown_does_not_bold_substring_before_session_name() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=["1. alphabetical - alpha"],
            entry_summaries=[_entry_summary(display_name="alpha", is_pinned=True)],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions",
    )

    assert "1. alphabetical - alpha" in rendered
    assert "**alpha**betical" not in rendered


def test_render_session_list_markdown_preserves_unpinned_entries() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=["1. alpha"],
            entry_summaries=[_entry_summary(display_name="alpha")],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions",
    )

    assert "1. alpha" in rendered


def test_render_session_list_markdown_escapes_markdown_in_session_names() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=["1. alpha_name - alpha_name"],
            entry_summaries=[_entry_summary(display_name="alpha_name", is_pinned=True)],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions",
    )

    assert "1. **alpha\\_name** - alpha\\_name" in rendered


def test_render_session_list_markdown_skips_empty_pinned_session_name() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=["1. Jan 01 00:00 - pin"],
            entry_summaries=[_entry_summary(display_name="", is_pinned=True)],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions",
    )

    assert rendered.startswith("# sessions\n\n1. Jan")
    assert "**" not in rendered


def test_render_session_list_markdown_skips_missing_pinned_session_name() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=["1. beta - Jan 01 00:00 - pin"],
            entry_summaries=[_entry_summary(display_name="alpha", is_pinned=True)],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions",
    )

    assert "1. beta - Jan 01 00:00 - pin" in rendered
    assert "**alpha**" not in rendered


def test_render_session_list_markdown_normalizes_heading() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=[],
            entry_summaries=[],
            usage="Usage: /session resume <id|number>",
        ),
        heading="# sessions",
    )

    assert rendered.startswith("# sessions")
    assert not rendered.startswith("# #")


def test_render_session_list_markdown_escapes_heading() -> None:
    rendered = render_session_list_markdown(
        SessionListSummary(
            entries=[],
            entry_summaries=[],
            usage="Usage: /session resume <id|number>",
        ),
        heading="sessions_[draft]*",
    )

    assert rendered.startswith("# sessions\\_\\[draft\\]\\*\n")
