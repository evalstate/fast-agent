"""Check the optional DuckDB CLI's UTF-8 pipe contract under a non-UTF-8 locale."""

import locale
import shutil
import sys

import pytest

from fast_agent.batch.input import _run_duckdb_cli_json
from fast_agent.utils import huggingface_hub


@pytest.mark.skipif(shutil.which("duckdb") is None, reason="DuckDB CLI is not installed")
@pytest.mark.skipif(bool(sys.flags.utf8_mode), reason="UTF-8 mode ignores the locale encoding")
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("SELECT chr(128512) AS message", [{"message": "😀"}]),
        ("SELECT '你好😀' AS message", [{"message": "你好😀"}]),
    ],
    ids=["unicode-output", "unicode-query"],
)
def test_duckdb_cli_preserves_unicode_with_non_utf8_locale(
    monkeypatch: pytest.MonkeyPatch, query: str, expected: list[dict[str, str]]
) -> None:
    # cp936 cannot encode the emoji, so locale-encoded pipes fail in both directions.
    monkeypatch.setattr(locale, "getencoding", lambda: "cp936")
    monkeypatch.setattr(huggingface_hub, "get_huggingface_hub_token", lambda: None)

    assert _run_duckdb_cli_json(query) == expected
