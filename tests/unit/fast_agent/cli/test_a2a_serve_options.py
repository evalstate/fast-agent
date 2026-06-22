from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fast_agent.cli.commands import serve as serve_command

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest


def test_serve_a2a_command_builds_a2a_run_request(monkeypatch) -> None:
    captured: list[AgentRunRequest] = []

    def fake_run_request(request: AgentRunRequest) -> None:
        captured.append(request)

    monkeypatch.setattr(serve_command, "run_request", fake_run_request)

    result = CliRunner().invoke(
        serve_command.app,
        [
            "a2a",
            "--name",
            "research-a2a",
            "--host",
            "127.0.0.1",
            "--port",
            "41241",
            "--instance-scope",
            "connection",
            "--agent-cards",
            "./agents",
            "--model",
            "passthrough",
            "--noenv",
        ],
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.name == "research-a2a"
    assert request.mode == "serve"
    assert request.transport == "a2a"
    assert request.host == "127.0.0.1"
    assert request.port == 41241
    assert request.instance_scope == "connection"
    assert request.agent_cards == ["./agents"]
    assert request.model == "passthrough"
    assert request.noenv is True


def test_serve_transport_a2a_callback_path_builds_a2a_run_request(monkeypatch) -> None:
    captured: list[AgentRunRequest] = []

    def fake_run_request(request: AgentRunRequest) -> None:
        captured.append(request)

    monkeypatch.setattr(serve_command, "run_request", fake_run_request)

    result = CliRunner().invoke(
        serve_command.app,
        [
            "--transport",
            "a2a",
            "--name",
            "generic-a2a",
            "--host",
            "127.0.0.1",
            "--port",
            "41242",
            "--instance-scope",
            "request",
            "--noenv",
        ],
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    request = captured[0]
    assert request.name == "generic-a2a"
    assert request.transport == "a2a"
    assert request.instance_scope == "request"


def test_serve_a2a_warns_for_remote_shell_server(monkeypatch) -> None:
    captured: list[AgentRunRequest] = []

    def fake_run_request(request: AgentRunRequest) -> None:
        captured.append(request)

    monkeypatch.setattr(serve_command, "run_request", fake_run_request)

    result = CliRunner().invoke(
        serve_command.app,
        [
            "a2a",
            "--host",
            "0.0.0.0",
            "--shell",
            "--model",
            "passthrough",
            "--noenv",
        ],
    )

    assert result.exit_code == 0
    assert len(captured) == 1
    stderr = " ".join(result.stderr.split())
    assert "exposes fast-agent to remote network clients" in stderr
    assert "shell execution tool is available to remote callers" in stderr
