import sys

import pytest
import typer
from typer.core import TyperCommand

from fast_agent.cli.commands import acp as acp_command


def test_acp_command_builds_request_with_watch() -> None:
    ctx = typer.Context(TyperCommand("acp"))
    request = acp_command._build_run_request(
        ctx=ctx,
        name="fast-agent-acp",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=["./agents"],
        card_tools=["./tool-cards"],
        urls=None,
        auth=None,
        client_metadata_url=None,
        model=None,
        home=None,
        no_home=False,
        skills_dir=None,
        npx=None,
        uvx=None,
        stdio=None,
        description="Chat with {agent}",
        host="127.0.0.1",
        port=8010,
        shell=False,
        no_permissions=False,
        resume=None,
        reload=True,
        watch=True,
    )

    assert request.mode == "serve"
    assert request.transport == "acp"
    assert request.host == "127.0.0.1"
    assert request.port == 8010
    assert request.agent_cards == ["./agents"]
    assert request.card_tools == ["./tool-cards"]
    assert request.instance_scope == "connection"
    assert request.reload is True
    assert request.watch is True


def test_acp_command_no_home_forces_permissions_disabled() -> None:
    ctx = typer.Context(TyperCommand("acp"))
    request = acp_command._build_run_request(
        ctx=ctx,
        name="fast-agent-acp",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=None,
        card_tools=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        model=None,
        home=None,
        no_home=True,
        skills_dir=None,
        npx=None,
        uvx=None,
        stdio=None,
        description=None,
        host="127.0.0.1",
        port=8010,
        shell=False,
        no_permissions=False,
        resume=None,
        reload=False,
        watch=False,
    )

    assert request.no_home is True
    assert request.permissions_enabled is False


def test_acp_command_builds_request_with_missing_shell_cwd_override() -> None:
    ctx = typer.Context(TyperCommand("acp"))
    request = acp_command._build_run_request(
        ctx=ctx,
        name="fast-agent-acp",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=None,
        card_tools=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        model=None,
        home=None,
        no_home=False,
        skills_dir=None,
        npx=None,
        uvx=None,
        stdio=None,
        description=None,
        host="127.0.0.1",
        port=8010,
        shell=False,
        no_permissions=False,
        resume=None,
        reload=False,
        watch=False,
        missing_shell_cwd=acp_command.serve.MissingShellCwdPolicy.CREATE,
    )

    assert request.missing_shell_cwd_policy == "create"


def test_acp_command_builds_request_with_prefer_local_shell() -> None:
    ctx = typer.Context(TyperCommand("acp"))
    request = acp_command._build_run_request(
        ctx=ctx,
        name="fast-agent-acp",
        instruction=None,
        config_path=None,
        servers=None,
        agent_cards=None,
        card_tools=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        model=None,
        home=None,
        no_home=False,
        skills_dir=None,
        npx=None,
        uvx=None,
        stdio=None,
        description=None,
        host="127.0.0.1",
        port=8010,
        shell=True,
        prefer_local_shell=True,
        no_permissions=False,
        resume=None,
        reload=False,
        watch=False,
    )

    assert request.shell_runtime is True
    assert request.prefer_local_shell is True


def test_acp_entrypoint_formats_usage_errors_without_changing_root_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from typer.testing import CliRunner

    from fast_agent.cli.main import app as root_app

    monkeypatch.setattr(sys, "argv", ["fast-agent-acp", "--unknown-option"])
    with pytest.raises(SystemExit) as exc_info:
        acp_command.main()

    assert exc_info.value.code == 1
    assert "No such command" in capsys.readouterr().err
    result = CliRunner().invoke(root_app, ["--unknown-option"])
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_acp_entrypoint_falls_back_when_rich_error_rendering_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from typer import rich_utils
    from typer._click.exceptions import ClickException

    def fail_render(exc: ClickException) -> None:
        raise RuntimeError("rendering failed")

    monkeypatch.setattr(rich_utils, "rich_format_error", fail_render)
    monkeypatch.setattr(sys, "argv", ["fast-agent-acp", "--unknown-option"])
    with pytest.raises(SystemExit) as exc_info:
        acp_command.main()

    assert exc_info.value.code == 1
    assert "Error: No such command" in capsys.readouterr().err
