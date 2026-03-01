from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from fast_agent.agents.smart_agent import _run_slash_command_call
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.skills import SKILLS_DEFAULT


@dataclass
class _AgentConfig:
    model: str | None = None
    tool_only: bool = False
    skills: object = SKILLS_DEFAULT


class _SmartAgentStub:
    def __init__(self, *, settings: Settings) -> None:
        self.name = "main"
        self.config = _AgentConfig()
        self.context = Context(config=settings)

    async def attach_mcp_server(self, **_kwargs):
        return object()

    async def detach_mcp_server(self, _server_name: str):
        return object()

    def list_attached_mcp_servers(self) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_run_slash_command_models_doctor_returns_markdown(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = await _run_slash_command_call(agent, "/models doctor")
    finally:
        os.chdir(previous_cwd)

    assert "# models.doctor" in result
    assert "models doctor" in result


@pytest.mark.asyncio
async def test_run_slash_command_check_rejects_invalid_argument_syntax(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    with pytest.raises(AgentConfigError, match="Invalid check arguments"):
        await _run_slash_command_call(agent, '/check "')


@pytest.mark.asyncio
async def test_run_slash_command_check_returns_markdown_heading(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = await _run_slash_command_call(agent, "/check")
    finally:
        os.chdir(previous_cwd)

    assert "# check" in result


@pytest.mark.asyncio
async def test_run_slash_command_skills_help_returns_usage(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/skills --help")

    assert "Usage: /skills [list|available|search|add|remove|update|registry|help]" in result


@pytest.mark.asyncio
async def test_run_slash_command_skills_search_without_query_shows_usage(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/skills search")

    assert "Usage: /skills search <query>" in result


@pytest.mark.asyncio
async def test_run_slash_command_unknown_returns_usage(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/doesnotexist")

    assert "Unknown slash command '/doesnotexist'" in result
    assert "Command map" in result


@pytest.mark.asyncio
async def test_run_slash_command_commands_index(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/commands")

    assert "# commands" in result
    assert "`/skills`" in result


@pytest.mark.asyncio
async def test_run_slash_command_commands_json(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/commands --json")

    assert '"kind": "command_index"' in result
    assert '"schema_version": "1"' in result


@pytest.mark.asyncio
async def test_run_slash_command_cards_help_returns_usage(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/cards --help")

    assert "Usage: /cards [list|add|remove|update|publish|registry|help] [args]" in result
