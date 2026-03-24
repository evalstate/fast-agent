from __future__ import annotations

from fast_agent.ui.prompt.attachment_tokens import strip_local_attachment_tokens


def test_strip_local_attachment_tokens_preserves_multiline_whitespace() -> None:
    text = "line  one\n  code block\n^file:/tmp/a.png\nline  two"

    stripped = strip_local_attachment_tokens(text)

    assert stripped == "line  one\n  code block\nline  two"


def test_strip_local_attachment_tokens_collapses_only_attachment_gap_between_words() -> None:
    text = "compare ^file:/tmp/a.png with this"

    stripped = strip_local_attachment_tokens(text)

    assert stripped == "compare with this"
