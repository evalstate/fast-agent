from markdown_it import MarkdownIt

from fast_agent.utils.markdown import escape_markdown_table_cell, markdown_code_span


def _inline_code_content(markdown: str) -> str:
    tokens = MarkdownIt().parse(markdown)
    inline = tokens[1]
    assert inline.children is not None
    code_token = inline.children[0]
    assert code_token.type == "code_inline"
    return code_token.content


def test_markdown_code_span_preserves_surrounding_spaces() -> None:
    value = " token "

    assert _inline_code_content(markdown_code_span(value)) == value


def test_markdown_code_span_preserves_backticks() -> None:
    value = "use `code` here"

    assert _inline_code_content(markdown_code_span(value)) == value


def test_markdown_code_span_preserves_consecutive_backticks() -> None:
    value = "use ``code`` here"

    assert _inline_code_content(markdown_code_span(value)) == value
    assert markdown_code_span(value).startswith("``` ")


def test_markdown_code_span_uses_fence_longer_than_longest_backtick_run() -> None:
    value = "use ```code``` here"

    assert _inline_code_content(markdown_code_span(value)) == value
    assert markdown_code_span(value).startswith("```` ")


def test_markdown_code_span_keeps_simple_values_compact() -> None:
    assert markdown_code_span("token") == "`token`"


def test_markdown_code_span_renders_empty_value_as_inline_code() -> None:
    assert _inline_code_content(markdown_code_span("")) == " "


def test_markdown_code_span_preserves_all_space_value_without_extra_padding() -> None:
    value = "   "

    assert _inline_code_content(markdown_code_span(value)) == value


def test_escape_markdown_table_cell_normalizes_line_breaks() -> None:
    value = "one\r\ntwo\rthree\nfour"

    assert escape_markdown_table_cell(value) == "one two three four"


def test_escape_markdown_table_cell_escapes_pipes_and_backslashes() -> None:
    value = r"path\to|tool"

    assert escape_markdown_table_cell(value) == r"path\\to\|tool"


def test_escape_markdown_table_cell_escapes_inline_markdown() -> None:
    value = r"*agent* `run` [docs]"

    assert escape_markdown_table_cell(value) == r"\*agent\* \`run\` \[docs\]"


def test_escape_markdown_table_cell_preserves_underscores() -> None:
    assert escape_markdown_table_cell("my_server") == "my_server"
