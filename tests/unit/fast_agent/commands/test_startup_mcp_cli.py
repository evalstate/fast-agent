import pytest
from typer.testing import CliRunner

from fast_agent.cli.commands import acp, go, serve


@pytest.mark.parametrize("command_module", [go, serve, acp])
def test_startup_surfaces_accept_repeated_urls_and_protocol(
    command_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []
    monkeypatch.setattr(command_module, "run_request", requests.append)

    result = CliRunner().invoke(
        command_module.app,
        [
            "--no-home",
            "--url",
            "https://one.example/mcp",
            "--url",
            "https://two.example/mcp",
            "--mcp-protocol",
            "legacy",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    servers = requests[0].startup_mcp_servers
    assert servers is not None
    assert list(servers) == ["one_example", "two_example"]
    assert {config.protocol_mode for config in servers.values()} == {"legacy"}
