from __future__ import annotations

import pytest

from fast_agent.ui.prompt.attachment_tokens import (
    build_remote_attachment_token,
    normalize_remote_attachment_reference,
    strip_local_attachment_tokens,
)


def test_strip_local_attachment_tokens_preserves_multiline_whitespace() -> None:
    text = "line  one\n  code block\n^file:/tmp/a.png\nline  two"

    stripped = strip_local_attachment_tokens(text)

    assert stripped == "line  one\n  code block\nline  two"


def test_strip_local_attachment_tokens_collapses_only_attachment_gap_between_words() -> None:
    text = "compare ^file:/tmp/a.png with this"

    stripped = strip_local_attachment_tokens(text)

    assert stripped == "compare with this"


def test_strip_local_attachment_tokens_removes_remote_url_tokens() -> None:
    text = "compare ^url:https://example.com/cat.png with this"

    stripped = strip_local_attachment_tokens(text)

    assert stripped == "compare with this"


def test_build_remote_attachment_token_preserves_query_delimiters() -> None:
    token = build_remote_attachment_token("https://example.com/cat.png?size=full&v=1")

    assert token == "^url:https://example.com/cat.png?size=full&v=1"


def test_normalize_remote_attachment_reference_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="Unsupported attachment URI scheme"):
        normalize_remote_attachment_reference("ftp://example.com/cat.png")
