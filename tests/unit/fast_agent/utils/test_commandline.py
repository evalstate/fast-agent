from __future__ import annotations

from fast_agent.utils.commandline import join_commandline, split_commandline


def test_split_commandline_posix_preserves_spaces() -> None:
    assert split_commandline('demo --root "My Folder"', syntax="posix") == [
        "demo",
        "--root",
        "My Folder",
    ]


def test_join_commandline_posix_quotes_spaces() -> None:
    rendered = join_commandline(["demo", "--root", "My Folder"], syntax="posix")
    assert split_commandline(rendered, syntax="posix") == ["demo", "--root", "My Folder"]


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


def test_split_commandline_windows_handles_backslashes() -> None:
    text = 'tool.exe "C:\\tmp\\path with spaces\\\\"'
    assert split_commandline(text, syntax="windows") == [
        "tool.exe",
        "C:\\tmp\\path with spaces\\",
    ]
