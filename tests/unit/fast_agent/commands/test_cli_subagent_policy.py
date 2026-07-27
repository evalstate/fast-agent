from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

from fast_agent import FastAgent
from fast_agent.agents.subagent_tool import SUBAGENT_TOOL_NAME, install_subagent_tool
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.cli.commands.go import app as go_app
from fast_agent.cli.runtime.agent_setup import _apply_cli_subagent_overrides
from fast_agent.cli.runtime.request_builders import build_command_run_request

if TYPE_CHECKING:
    from pathlib import Path


def _request(
    tmp_path: Path,
    *,
    subagents: bool | None = None,
    subagent_model: str | None = None,
):
    return build_command_run_request(
        name="test",
        instruction_option=None,
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        client_metadata_url=None,
        agent_cards=None,
        card_tools=None,
        model="passthrough",
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        npx=None,
        uvx=None,
        stdio=None,
        target_agent_name=None,
        skills_directory=None,
        home=tmp_path,
        shell_enabled=False,
        mode="interactive",
        subagents=subagents,
        subagent_model=subagent_model,
    )


@pytest.mark.unit
def test_cli_subagents_enable_a_generated_no_card_agent(tmp_path) -> None:
    fast = FastAgent("test", parse_cli_args=False)

    @fast.agent(name="agent", model="passthrough", default=True)
    async def generated_agent() -> None:
        pass

    _apply_cli_subagent_overrides(fast, _request(tmp_path, subagents=True))
    config = fast.agents["agent"]["config"]
    agent = ToolAgent(config)

    assert install_subagent_tool(agent)
    assert SUBAGENT_TOOL_NAME in agent._execution_tools


@pytest.mark.unit
def test_cli_no_subagents_overrides_enabled_card(tmp_path) -> None:
    card = tmp_path / "card.yaml"
    card.write_text(
        "type: agent\nname: card\nmodel: passthrough\nsubagents: true\n",
        encoding="utf-8",
    )
    fast = FastAgent("test", parse_cli_args=False)
    fast.load_agents(str(card))

    config = fast.agents["card"]["config"]
    assert install_subagent_tool(ToolAgent(config)) is True

    _apply_cli_subagent_overrides(fast, _request(tmp_path, subagents=False))
    assert config.subagents is False
    assert config.subagent_activation_source == "cli"
    assert install_subagent_tool(ToolAgent(config)) is False


@pytest.mark.unit
def test_card_disable_overrides_positive_cli_activation(tmp_path) -> None:
    card = tmp_path / "card.yaml"
    card.write_text(
        "type: agent\nname: card\nmodel: passthrough\nsubagents: false\n",
        encoding="utf-8",
    )
    fast = FastAgent("test", parse_cli_args=False)
    fast.load_agents(str(card))

    config = fast.agents["card"]["config"]
    _apply_cli_subagent_overrides(
        fast,
        _request(tmp_path, subagent_model="child-model"),
    )

    assert config.subagents is False
    assert config.subagent_activation_source == "configuration"
    assert config.subagent_model is None
    assert install_subagent_tool(ToolAgent(config)) is False


@pytest.mark.unit
def test_cli_rejects_no_subagents_with_subagent_model() -> None:
    result = CliRunner().invoke(
        go_app,
        ["--no-subagents", "--subagent-model", "child-model"],
    )

    assert result.exit_code == 2
    output = strip_ansi(result.output)
    assert "Cannot combine --subagent-model with" in output
    assert "--no-subagents." in output
