from __future__ import annotations

from typing import Any

import pytest

from fast_agent.utils import commandline
from fast_agent.utils.commandline import (
    join_commandline,
    quote_commandline_token,
    resolve_commandline_syntax,
    split_commandline,
    split_posix_like_preserving_backslashes,
)


def test_resolve_commandline_syntax_auto_defaults_to_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commandline.os, "name", "posix")
    assert resolve_commandline_syntax("auto") == "posix"


def test_resolve_commandline_syntax_auto_uses_windows_on_nt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(commandline.os, "name", "nt")
    assert resolve_commandline_syntax("auto") == "windows"


def test_resolve_commandline_syntax_rejects_unknown_syntax() -> None:
    unsupported_syntax: Any = "fish"
    with pytest.raises(ValueError, match="Unsupported command-line syntax: fish"):
        resolve_commandline_syntax(unsupported_syntax)


def test_resolve_commandline_syntax_rejects_non_string_syntax() -> None:
    unsupported_syntax: Any = 123
    with pytest.raises(ValueError, match="Unsupported command-line syntax: 123"):
        resolve_commandline_syntax(unsupported_syntax)


def test_resolve_commandline_syntax_normalizes_user_input() -> None:
    syntax: Any = " POSIX "

    assert resolve_commandline_syntax(syntax) == "posix"


def test_split_commandline_posix_preserves_spaces() -> None:
    assert split_commandline('demo --root "My Folder"', syntax="posix") == [
        "demo",
        "--root",
        "My Folder",
    ]


def test_join_commandline_posix_quotes_spaces() -> None:
    rendered = join_commandline(["demo", "--root", "My Folder"], syntax="posix")
    assert split_commandline(rendered, syntax="posix") == ["demo", "--root", "My Folder"]


def test_quote_commandline_token_posix_quotes_spaces() -> None:
    rendered = quote_commandline_token("My Folder", syntax="posix")
    assert split_commandline(rendered, syntax="posix") == ["My Folder"]


def test_split_commandline_windows_preserves_quoted_path() -> None:
    text = '"C:\\Program Files\\Tool\\tool.exe" --flag'
    assert split_commandline(text, syntax="windows") == [
        "C:\\Program Files\\Tool\\tool.exe",
        "--flag",
    ]


def test_join_commandline_windows_round_trips_unc_path_and_empty_arg() -> None:
    argv = [r"\\server\share\tool.exe", "", "--flag"]
    rendered = join_commandline(argv, syntax="windows")
    assert split_commandline(rendered, syntax="windows") == argv


def test_quote_commandline_token_windows_round_trips_path() -> None:
    token = r"C:\Program Files\Tool\tool.exe"
    rendered = quote_commandline_token(token, syntax="windows")
    assert split_commandline(rendered, syntax="windows") == [token]


def test_split_commandline_windows_handles_backslashes() -> None:
    text = 'tool.exe "C:\\tmp\\path with spaces\\\\"'
    assert split_commandline(text, syntax="windows") == [
        "tool.exe",
        "C:\\tmp\\path with spaces\\",
    ]


def test_split_posix_like_preserving_backslashes_supports_windows_paths() -> None:
    text = r'C:\tmp\session.json --output "C:\tmp\trace file.jsonl"'

    assert split_posix_like_preserving_backslashes(text) == [
        r"C:\tmp\session.json",
        "--output",
        r"C:\tmp\trace file.jsonl",
    ]


def test_split_posix_like_preserving_backslashes_unescapes_spaces() -> None:
    text = r"latest --agent dev\ agent --output trace\ file.jsonl"

    assert split_posix_like_preserving_backslashes(text) == [
        "latest",
        "--agent",
        "dev agent",
        "--output",
        "trace file.jsonl",
    ]


def test_split_commandline_normalizes_parser_value_errors() -> None:
    with pytest.raises(ValueError, match="No closing quotation"):
        split_commandline('demo "unterminated', syntax="posix")


def test_split_commandline_does_not_hide_unexpected_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken_splitter(_text: str) -> list[str]:
        raise RuntimeError("splitter failed")

    monkeypatch.setitem(commandline._SPLITTERS, "posix", _broken_splitter)

    with pytest.raises(RuntimeError, match="splitter failed"):
        split_commandline("demo", syntax="posix")
