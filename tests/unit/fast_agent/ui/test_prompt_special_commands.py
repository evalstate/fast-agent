from __future__ import annotations

import pytest

from fast_agent.core.exceptions import PromptExitError
from fast_agent.ui.command_payloads import (
    SelectPromptCommand,
    ShowUsageCommand,
    SwitchAgentCommand,
)
from fast_agent.ui.prompt.special_commands import handle_special_commands


def test_handle_special_commands_returns_payload_for_simple_command() -> None:
    assert handle_special_commands("SHOW_USAGE") == ShowUsageCommand()


def test_handle_special_commands_selects_named_prompt() -> None:
    assert handle_special_commands("SELECT_PROMPT: docs ", agent_app=True) == (
        SelectPromptCommand(prompt_index=None, prompt_name="docs")
    )


def test_handle_special_commands_accepts_lowercase_exit() -> None:
    with pytest.raises(PromptExitError):
        handle_special_commands("exit")


def test_handle_special_commands_switches_available_agent() -> None:
    assert handle_special_commands(
        "SWITCH:writer",
        agent_app=True,
        available_agents={"writer"},
    ) == SwitchAgentCommand(agent_name="writer")


def test_handle_special_commands_prints_bracketed_unknown_agent_literally(capsys) -> None:
    result = handle_special_commands(
        "SWITCH:[draft]",
        agent_app=True,
        available_agents={"writer"},
    )

    assert result is True
    assert "Unknown agent: [draft]" in capsys.readouterr().out
