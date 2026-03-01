from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.core.fastagent import AgentInstance

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    acp_commands = {}


class _App:
    def _agent(self, _name: str):
        return _Agent()

    def agent_names(self):
        return ["main"]

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


@pytest.mark.asyncio
async def test_slash_command_models_catalog() -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )

    output = await handler.execute_command("models", "catalog anthropic")

    assert "# models catalog" in output
    assert "Provider: Anthropic" in output


@pytest.mark.asyncio
async def test_slash_command_models_registered_in_available_commands(tmp_path: Path) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )

    previous_cwd = Path.cwd()
    try:
        # Ensure deterministic working directory while command list is produced.
        os.chdir(tmp_path)
        handler = SlashCommandHandler(
            session_id="s1",
            instance=instance,
            primary_agent_name="main",
        )
        command_names = {command.name for command in handler.get_available_commands()}
    finally:
        os.chdir(previous_cwd)

    assert "models" in command_names
    assert "commands" in command_names


@pytest.mark.asyncio
async def test_slash_command_commands_index() -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )

    output = await handler.execute_command("commands", "")

    assert "# commands" in output
    assert "`/skills`" in output


@pytest.mark.asyncio
async def test_slash_command_hints_use_catalog_actions() -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )

    commands = {command.name: command for command in handler.get_available_commands()}
    skills_hint = commands["skills"].input
    cards_hint = commands["cards"].input

    assert skills_hint is not None
    assert cards_hint is not None
    skills_hint_text = skills_hint.root.hint
    cards_hint_text = cards_hint.root.hint
    assert skills_hint_text is not None
    assert cards_hint_text is not None
    assert "available|search <query>" in skills_hint_text
    assert "add <name|number>" in skills_hint_text
    assert "add|remove|update|publish|registry" in cards_hint_text


@pytest.mark.asyncio
async def test_slash_command_models_aliases_set_dry_run(tmp_path: Path) -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        output = await handler.execute_command(
            "models",
            "aliases set $system.fast claude-haiku-4-5 --dry-run",
        )
    finally:
        os.chdir(previous_cwd)

    assert "# models aliases" in output
    assert "Mode: dry-run" in output
    assert "model_aliases.system.fast" in output
