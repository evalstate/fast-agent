from __future__ import annotations

import pytest

from fast_agent.ui.prompt.attachment_tokens import (
    append_attachment_tokens,
    build_remote_attachment_token,
    is_remote_attachment_reference,
    normalize_local_attachment_reference,
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


def test_normalize_local_attachment_reference_accepts_file_scheme_case_insensitively() -> None:
    assert normalize_local_attachment_reference("FILE:///tmp/cat.png").as_posix() == "/tmp/cat.png"


def test_is_remote_attachment_reference_accepts_http_schemes_case_insensitively() -> None:
    assert is_remote_attachment_reference(" HTTPS://example.com/cat.png ")
    assert not is_remote_attachment_reference("ftp://example.com/cat.png")
    assert not is_remote_attachment_reference("cat.png")


def test_append_attachment_tokens_preserves_existing_trailing_space() -> None:
    assert append_attachment_tokens("describe this ", ["^file:/tmp/a.png"]) == (
        "describe this ^file:/tmp/a.png"
    )


def test_append_attachment_tokens_inserts_single_separator_when_needed() -> None:
    assert append_attachment_tokens(
        "describe this", ["^file:/tmp/a.png", "^url:https://x.test"]
    ) == ("describe this ^file:/tmp/a.png ^url:https://x.test")
