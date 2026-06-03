from fast_agent.commands.history_summaries import (
    HistoryMessageSnippet,
    HistoryOverview,
    HistoryTurnReport,
    HistoryTurnSummary,
)
from fast_agent.commands.renderers.history_markdown import (
    _format_positive_duration_ms,
    render_history_overview_markdown,
    render_history_turn_report_markdown,
)


def test_render_history_overview_markdown_escapes_recent_message_snippets() -> None:
    overview = HistoryOverview(
        message_count=1,
        user_message_count=1,
        assistant_message_count=0,
        tool_calls=0,
        tool_successes=0,
        tool_errors=0,
        recent_messages=[
            HistoryMessageSnippet(
                role="user_role",
                snippet="See [docs](bad) and *bold*",
            )
        ],
    )

    rendered = render_history_overview_markdown(overview, heading="history")

    assert "- user\\_role: See \\[docs\\](bad) and \\*bold\\*" in rendered
    assert "[docs](bad)" not in rendered


def test_render_history_overview_markdown_pluralizes_recent_message_heading() -> None:
    overview = HistoryOverview(
        message_count=1,
        user_message_count=1,
        assistant_message_count=0,
        tool_calls=0,
        tool_successes=0,
        tool_errors=0,
        recent_messages=[
            HistoryMessageSnippet(role="user", snippet="hello"),
        ],
    )

    rendered = render_history_overview_markdown(overview, heading="history")

    assert "Recent 1 message:" in rendered
    assert "Recent 1 messages:" not in rendered


def test_render_history_overview_markdown_normalizes_blank_recent_message_fields() -> None:
    overview = HistoryOverview(
        message_count=1,
        user_message_count=1,
        assistant_message_count=0,
        tool_calls=0,
        tool_successes=0,
        tool_errors=0,
        recent_messages=[
            HistoryMessageSnippet(role="   ", snippet="  "),
            HistoryMessageSnippet(role=" assistant ", snippet=" done "),
        ],
    )

    rendered = render_history_overview_markdown(overview, heading="history")

    assert "- unknown:" in rendered
    assert "- assistant: done" in rendered
    assert "- unknown: " not in rendered


def test_render_history_turn_report_markdown_escapes_table_cell_pipes() -> None:
    report = HistoryTurnReport(
        turn_count=1,
        total_tool_calls=0,
        total_tool_errors=0,
        total_llm_time_ms=0,
        total_tool_time_ms=0,
        total_turn_time_ms=0,
        average_turn_time_ms=None,
        average_tool_time_ms=None,
        average_ttft_ms=None,
        average_response_ms=None,
        average_tps=None,
        turns=[
            HistoryTurnSummary(
                turn_index=1,
                user_snippet="left | right",
                assistant_snippet="done | ok",
                tool_calls=0,
                tool_errors=0,
                llm_time_ms=None,
                tool_time_ms=None,
                turn_time_ms=120,
                ttft_ms=None,
                response_ms=None,
                output_tokens=None,
                tps=None,
            )
        ],
    )

    rendered = render_history_turn_report_markdown(report, heading="history")

    assert "left \\| right → done \\| ok" in rendered


def test_render_history_turn_report_markdown_formats_summary_lines() -> None:
    report = HistoryTurnReport(
        turn_count=2,
        total_tool_calls=3,
        total_tool_errors=1,
        total_llm_time_ms=1500,
        total_tool_time_ms=0,
        total_turn_time_ms=2300,
        average_turn_time_ms=1150,
        average_tool_time_ms=None,
        average_ttft_ms=120,
        average_response_ms=900,
        average_tps=12.34,
        turns=[],
    )

    rendered = render_history_turn_report_markdown(report, heading="history")

    assert "Tools: 3 (errors: 1)" in rendered
    assert "Totals: turn 2.3s, llm 1.5s, tool -" in rendered
    assert "Averages: turn 1.1s, tool -, ttft 120ms, resp 900ms, tps 12.3" in rendered


def test_format_positive_duration_ms_hides_non_positive_values() -> None:
    assert _format_positive_duration_ms(1500) == "1.5s"
    assert _format_positive_duration_ms(0) == "-"
    assert _format_positive_duration_ms(-1) == "-"


def test_render_history_turn_report_markdown_normalizes_heading() -> None:
    report = HistoryTurnReport(
        turn_count=0,
        total_tool_calls=0,
        total_tool_errors=0,
        total_llm_time_ms=0,
        total_tool_time_ms=0,
        total_turn_time_ms=0,
        average_turn_time_ms=None,
        average_tool_time_ms=None,
        average_ttft_ms=None,
        average_response_ms=None,
        average_tps=None,
        turns=[],
    )

    rendered = render_history_turn_report_markdown(report, heading="# history")

    assert rendered.startswith("# history")
    assert not rendered.startswith("# #")


def test_history_markdown_renderers_escape_headings() -> None:
    overview = HistoryOverview(
        message_count=0,
        user_message_count=0,
        assistant_message_count=0,
        tool_calls=0,
        tool_successes=0,
        tool_errors=0,
        recent_messages=[],
    )
    report = HistoryTurnReport(
        turn_count=0,
        total_tool_calls=0,
        total_tool_errors=0,
        total_llm_time_ms=0,
        total_tool_time_ms=0,
        total_turn_time_ms=0,
        average_turn_time_ms=None,
        average_tool_time_ms=None,
        average_ttft_ms=None,
        average_response_ms=None,
        average_tps=None,
        turns=[],
    )

    overview_rendered = render_history_overview_markdown(
        overview,
        heading="history_[draft]*",
    )
    report_rendered = render_history_turn_report_markdown(
        report,
        heading="turns_[draft]*",
    )

    assert overview_rendered.startswith("# history\\_\\[draft\\]\\*\n")
    assert report_rendered.startswith("# turns\\_\\[draft\\]\\*\n")
