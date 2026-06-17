from __future__ import annotations

from fast_agent.utils.slash_commands import parse_slash_command_line, split_subcommand_and_remainder


def test_parse_slash_command_line_returns_command_and_arguments() -> None:
    assert parse_slash_command_line("  /model   switch gpt-5-mini  ") == (
        "model",
        "switch gpt-5-mini",
    )
    assert parse_slash_command_line("/clear") == ("clear", "")


def test_parse_slash_command_line_handles_empty_and_non_slash_input() -> None:
    assert parse_slash_command_line("/") == ("", "")
    assert parse_slash_command_line("  /   ") == ("", "")
    assert parse_slash_command_line("   ") is None
    assert parse_slash_command_line("hello /model") is None


def test_split_subcommand_and_remainder_returns_command_and_remainder() -> None:
    assert split_subcommand_and_remainder("connect demo --name docs") == (
        "connect",
        "demo --name docs",
    )
    assert split_subcommand_and_remainder("connect\tdemo --name docs") == (
        "connect",
        "demo --name docs",
    )


def test_split_subcommand_and_remainder_returns_empty_strings_for_blank_input() -> None:
    assert split_subcommand_and_remainder("") == ("", "")
    assert split_subcommand_and_remainder("   ") == ("", "")
