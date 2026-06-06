from __future__ import annotations

from fast_agent.acp.slash.handlers import commands as commands_handler


def test_resolve_available_command_name_matches_case_insensitively() -> None:
    assert (
        commands_handler._resolve_available_command_name(
            " STATUS ",
            {"status", "history"},
        )
        == "status"
    )


def test_resolve_available_command_name_rejects_ambiguous_casefold_matches() -> None:
    assert commands_handler._resolve_available_command_name("STATUS", {"Status", "status"}) is None


def test_unknown_command_family_escapes_backticks_in_name() -> None:
    rendered = commands_handler._render_unknown_command_family("skill`s")

    assert "Unknown command family: `` skill`s ``." in rendered


def test_missing_command_metadata_escapes_backticks_in_command() -> None:
    rendered = commands_handler._render_missing_metadata("jump`now", "run`fast")

    assert "No discovery metadata for `` /jump`now run`fast `` yet." in rendered
