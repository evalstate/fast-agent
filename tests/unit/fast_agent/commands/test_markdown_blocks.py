from fast_agent.commands.renderers.markdown_blocks import markdown_heading, wrapped_quote_lines


def test_markdown_heading_normalizes_and_escapes_heading_text() -> None:
    assert markdown_heading("## Commands [active]", level=2) == "## Commands \\[active\\]"


def test_markdown_heading_returns_empty_for_empty_heading() -> None:
    assert markdown_heading(" #  ") == ""


def test_wrapped_quote_lines_treats_non_positive_max_lines_as_no_output() -> None:
    assert wrapped_quote_lines("content", max_lines=0) == []
    assert wrapped_quote_lines("content", max_lines=-1) == []


def test_wrapped_quote_lines_omits_blank_text() -> None:
    assert wrapped_quote_lines("   ") == []
    assert wrapped_quote_lines(None) == []


def test_wrapped_quote_lines_strips_outer_whitespace() -> None:
    assert wrapped_quote_lines("  content  ") == ["> content"]


def test_wrapped_quote_lines_clamps_non_positive_width() -> None:
    assert wrapped_quote_lines("abc", width=0, max_lines=2) == ["> a", "> b", "> …"]
