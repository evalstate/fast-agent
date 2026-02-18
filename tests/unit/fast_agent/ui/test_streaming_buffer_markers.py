from fast_agent.ui.streaming_buffer import StreamBuffer


def test_contains_context_markers_returns_false_for_plain_text() -> None:
    buffer = StreamBuffer()

    text = "This is plain content with punctuation, numbers 123, and markdown-ish words."

    assert buffer._contains_context_markers(text) is False


def test_contains_context_markers_detects_tables() -> None:
    buffer = StreamBuffer()

    table_text = "| Name | Score |\n| --- | --- |\n| Ada | 10 |\n"

    assert buffer._contains_context_markers(table_text) is True


def test_contains_context_markers_detects_tables_without_leading_pipes() -> None:
    buffer = StreamBuffer()

    table_text = "Name | Score\n--- | ---\nAda | 10\n"

    assert buffer._contains_context_markers(table_text) is True


def test_contains_context_markers_detects_indented_code() -> None:
    buffer = StreamBuffer()

    indented = "Intro\n    code line\n    code line 2\n"

    assert buffer._contains_context_markers(indented) is True
