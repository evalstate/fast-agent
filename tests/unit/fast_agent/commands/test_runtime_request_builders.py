from pathlib import Path

import pytest
import typer

from fast_agent.cli.runtime.request_builders import (
    build_agent_run_request,
    build_command_run_request,
    resolve_default_instruction,
    resolve_smart_agent_enabled,
)
from fast_agent.constants import SMART_AGENT_INSTRUCTION


def test_build_agent_run_request_merges_url_servers_after_explicit_servers() -> None:
    request = build_agent_run_request(
        name="test-agent",
        instruction="instruction",
        config_path=None,
        servers="alpha,beta",
        urls="http://localhost:9000/mcp",
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    assert request.server_list is not None
    assert request.server_list[:2] == ["alpha", "beta"]
    assert request.url_servers is not None
    assert request.server_list[2:] == list(request.url_servers.keys())


def test_build_agent_run_request_includes_client_metadata_url_in_url_server_auth() -> None:
    request = build_agent_run_request(
        name="test-agent",
        instruction="instruction",
        config_path=None,
        servers=None,
        urls="https://example.com/mcp",
        auth=None,
        client_metadata_url="https://example.com/oauth/client-metadata.json",
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    assert request.url_servers is not None
    server_config = next(iter(request.url_servers.values()))
    assert server_config["auth"] == {
        "oauth": True,
        "client_metadata_url": "https://example.com/oauth/client-metadata.json",
    }


def test_build_agent_run_request_skips_invalid_stdio_commands(capsys) -> None:
    request = build_agent_run_request(
        name="test-agent",
        instruction="instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=["python good.py", "python \"unterminated", ""],
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    captured = capsys.readouterr()
    assert "Error parsing stdio command" in captured.err
    assert "Error: Empty stdio command" in captured.err
    assert request.stdio_servers is not None
    assert len(request.stdio_servers) == 1
    only_config = next(iter(request.stdio_servers.values()))
    assert only_config["command"] == "python"
    assert only_config["args"] == ["good.py"]


def test_build_command_run_request_resolves_defaults() -> None:
    request = build_command_run_request(
        name="cli",
        instruction_option=None,
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file="out.json",
        resume=None,
        npx=None,
        uvx=None,
        stdio=None,
        target_agent_name=None,
        skills_directory=None,
        environment_dir=Path("."),
        shell_enabled=False,
        mode="serve",
    )

    assert request.instruction == resolve_default_instruction(None, "serve")
    assert request.agent_name == "agent"
    assert request.result_file == "out.json"


def test_build_command_run_request_smart_flag_uses_smart_instruction() -> None:
    request = build_command_run_request(
        name="cli",
        instruction_option=None,
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        npx=None,
        uvx=None,
        stdio=None,
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        force_smart=True,
        shell_enabled=False,
        mode="interactive",
    )

    assert request.force_smart is True
    assert request.instruction == SMART_AGENT_INSTRUCTION


def test_build_command_run_request_accepts_missing_shell_cwd_override() -> None:
    request = build_command_run_request(
        name="cli",
        instruction_option=None,
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        npx=None,
        uvx=None,
        stdio=None,
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        shell_enabled=False,
        mode="serve",
        missing_shell_cwd_policy="error",
    )

    assert request.missing_shell_cwd_policy == "error"


def test_resolve_smart_agent_enabled_disables_smart_for_multi_model_even_when_forced() -> None:
    assert resolve_smart_agent_enabled(
        "gpt-4.1,claude-sonnet-4-5",
        "interactive",
        force_smart=True,
    ) is False


def test_build_agent_run_request_noenv_keeps_explicit_cards_only() -> None:
    request = build_agent_run_request(
        name="test-agent",
        instruction="instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=["./cards", "./cards", "./extra"],
        card_tools=["./tools", "./tools"],
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
        noenv=True,
    )

    assert request.agent_cards == ["./cards", "./extra"]
    assert request.card_tools == ["./tools"]
    assert request.environment_dir is None
    assert request.allow_implicit_cards is False


def test_build_agent_run_request_noenv_forces_serve_permissions_off() -> None:
    request = build_agent_run_request(
        name="test-agent",
        instruction="instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        shell_enabled=False,
        mode="serve",
        transport="acp",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
        noenv=True,
    )

    assert request.permissions_enabled is False


def test_build_command_run_request_rejects_noenv_with_env() -> None:
    with pytest.raises(typer.BadParameter, match="Cannot combine --noenv with --env"):
        build_command_run_request(
            name="cli",
            instruction_option=None,
            config_path=None,
            servers=None,
            urls=None,
            auth=None,
            client_metadata_url=None,
            agent_cards=None,
            card_tools=None,
            model=None,
            message=None,
            prompt_file=None,
            result_file=None,
            resume=None,
            npx=None,
            uvx=None,
            stdio=None,
            target_agent_name=None,
            skills_directory=None,
            environment_dir=Path("."),
            shell_enabled=False,
            mode="interactive",
            noenv=True,
        )


def test_build_command_run_request_rejects_noenv_with_resume() -> None:
    with pytest.raises(typer.BadParameter, match="Cannot combine --noenv with --resume"):
        build_command_run_request(
            name="cli",
            instruction_option=None,
            config_path=None,
            servers=None,
            urls=None,
            auth=None,
            client_metadata_url=None,
            agent_cards=None,
            card_tools=None,
            model=None,
            message=None,
            prompt_file=None,
            result_file=None,
            resume="latest",
            npx=None,
            uvx=None,
            stdio=None,
            target_agent_name=None,
            skills_directory=None,
            environment_dir=None,
            shell_enabled=False,
            mode="interactive",
            noenv=True,
        )


def test_build_command_run_request_rejects_malformed_url() -> None:
    with pytest.raises(typer.BadParameter, match="URL must have http or https scheme"):
        build_command_run_request(
            name="cli",
            instruction_option=None,
            config_path=None,
            servers=None,
            urls="not-a-url",
            auth=None,
            client_metadata_url=None,
            agent_cards=None,
            card_tools=None,
            model=None,
            message=None,
            prompt_file=None,
            result_file=None,
            resume=None,
            npx=None,
            uvx=None,
            stdio=None,
            target_agent_name=None,
            skills_directory=None,
            environment_dir=None,
            shell_enabled=False,
            mode="interactive",
        )
