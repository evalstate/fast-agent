from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from fast_agent.agents.smart_agent import _run_slash_command_call
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
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
        self.llm = None
        self._llm = None

    async def attach_mcp_server(self, **_kwargs):
        return object()

    async def detach_mcp_server(self, _server_name: str):
        return object()

    def list_attached_mcp_servers(self) -> list[str]:
        return []


class _TaskBudgetLlm:
    task_budget_supported = True
    task_budget_tokens = None
    service_tier_supported = False
    web_search_supported = False
    web_fetch_supported = False
    reasoning_effort_spec = None
    text_verbosity_spec = None
    text_verbosity = None
    resolved_model = None
    provider = Provider.ANTHROPIC
    model_name = "claude-opus-4-7"
    default_request_params = RequestParams()
    configured_transport = None
    active_transport = None

    def set_task_budget_tokens(self, value: int | None) -> None:
        self.task_budget_tokens = value


@pytest.mark.asyncio
async def test_run_slash_command_model_doctor_returns_markdown(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = await _run_slash_command_call(agent, "/model doctor")
    finally:
        os.chdir(previous_cwd)

    assert "# model.doctor" in result
    assert "model doctor" in result


@pytest.mark.asyncio
async def test_run_slash_command_model_task_budget_routes_to_task_budget_handler(
    tmp_path: Path,
) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)
    agent.llm = _TaskBudgetLlm()
    agent._llm = agent.llm

    result = await _run_slash_command_call(agent, "/model task_budget 64k")

    assert "# model.task_budget" in result
    assert "Task budget: set to 64k." in result
    assert agent.llm.task_budget_tokens == 64_000


@pytest.mark.asyncio
async def test_run_slash_command_check_rejects_invalid_argument_syntax(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    with pytest.raises(AgentConfigError, match="Invalid check arguments"):
        await _run_slash_command_call(agent, '/check "')


@pytest.mark.asyncio
async def test_run_slash_command_mcp_connect_wraps_parse_errors(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    with pytest.raises(AgentConfigError, match="Invalid /mcp connect arguments"):
        await _run_slash_command_call(agent, "/mcp connect npx demo-server --timeout 0")


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

    assert "Usage: /cards [list|add|remove|readme|update|publish|registry|help] [args]" in result
