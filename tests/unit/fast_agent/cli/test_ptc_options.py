from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
import typer
from typer.testing import CliRunner

from fast_agent.cli import __main__ as cli_main
from fast_agent.cli.commands import go as go_command
from fast_agent.config import ProgrammaticToolCallingSettings

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest


@pytest.mark.parametrize("isolated", [False, True])
def test_go_ptc_all_enables_ptc_without_search(
    monkeypatch: pytest.MonkeyPatch, isolated: bool
) -> None:
    captured: list[AgentRunRequest] = []
    monkeypatch.setattr(go_command, "run_request", captured.append)
    args = ["--message", "hi", "--ptc-all"]
    if isolated:
        args.append("--isolated")

    result = CliRunner().invoke(go_command.app, args)

    assert result.exit_code == 0, result.output
    (request,) = captured
    assert request.isolated is isolated
    assert request.ptc is True
    assert request.ptc_search is False
    assert request.to_agent_setup_kwargs()["ptc_search"] is False


@pytest.mark.parametrize(
    ("args", "ptc", "ptc_search"),
    [
        (["--ptc-all", "-m", "hi"], True, False),
        (["--ptc-all", "--message", "hi"], True, False),
        (["--message", "hi", "--ptc-all"], True, False),
        (["--ptc", "-m", "hi"], True, None),
        (["--no-ptc", "-m", "hi"], False, None),
    ],
)
def test_root_ptc_flags_route_to_go(
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    ptc: bool,
    ptc_search: bool | None,
) -> None:
    captured: list[AgentRunRequest] = []
    argv = ["fast-agent", *args, "--no-home"]
    monkeypatch.setattr(go_command, "run_request", captured.append)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        cli_main.main()

    assert exit_info.value.code == 0
    assert argv[1] == "go"
    (request,) = captured
    assert request.ptc is ptc
    assert request.ptc_search is ptc_search


def test_go_ptc_all_conflicts_with_no_ptc() -> None:
    result = CliRunner().invoke(
        go_command.app,
        ["--message", "hi", "--ptc-all", "--no-ptc"],
        standalone_mode=False,
    )

    assert isinstance(result.exception, typer.BadParameter)
    assert result.exception.param_hint == "--ptc-all"
    assert result.exception.message == "Cannot combine --ptc-all with --no-ptc."


def test_ptc_settings_enable_search_by_default() -> None:
    assert ProgrammaticToolCallingSettings().search is True
