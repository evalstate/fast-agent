from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.agents.smart_agent import (
    _build_command_context,
    _parse_family_command_action,
    _parse_slash_command_text,
    _resolve_command_agent_map,
    _run_slash_command_call,
)
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.skills import SKILLS_DEFAULT

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fast_agent.interfaces import AgentProtocol


@dataclass
class _AgentConfig:
    model: str | None = None
    tool_only: bool = False
    skills: object = SKILLS_DEFAULT


class _SmartAgentStub:
    def __init__(self, *, settings: Settings) -> None:
        self.name = "main"
        self.config = _AgentConfig()
        self.context: Context | None = Context(config=settings)
        self.llm = None
        self._llm = None
        self._agent_registry: Mapping[str, AgentProtocol] | None = None

    @property
    def agent_registry(self) -> Mapping[str, AgentProtocol] | None:
        return self._agent_registry

    def set_agent_registry(self, registry: Mapping[str, AgentProtocol] | None) -> None:
        self._agent_registry = registry

    def get_agent(self, name: str) -> AgentProtocol | None:
        if self._agent_registry is None:
            return None
        return self._agent_registry.get(name)

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


def test_parse_family_command_action_uses_catalog_aliases() -> None:
    parsed = _parse_family_command_action("cards", "show docs")

    assert parsed.action == "readme"
    assert parsed.argument == "docs"


def test_parse_family_command_action_normalizes_empty_argument() -> None:
    parsed = _parse_family_command_action("cards", "show    ")

    assert parsed.action == "readme"
    assert parsed.argument is None


def test_parse_family_command_action_reports_split_errors() -> None:
    with pytest.raises(AgentConfigError, match="Invalid command arguments"):
        _parse_family_command_action("cards", 'show "unterminated')


def test_parse_slash_command_text_accepts_slash_and_bare_commands() -> None:
    slash = _parse_slash_command_text("  /MCP   list servers  ")
    bare = _parse_slash_command_text("MCP\tlist servers")

    assert slash.command_name == "mcp"
    assert slash.arguments == "list servers"
    assert bare.command_name == "mcp"
    assert bare.arguments == "list servers"


def test_parse_slash_command_text_rejects_blank_commands() -> None:
    with pytest.raises(AgentConfigError, match="Slash command is empty"):
        _parse_slash_command_text("/")


def test_command_agent_map_includes_registry_and_current_agent() -> None:
    settings = Settings()
    main = _SmartAgentStub(settings=settings)
    peer = _SmartAgentStub(settings=settings)
    peer.name = "peer"
    main.set_agent_registry({"peer": cast("AgentProtocol", peer)})

    agents = _resolve_command_agent_map(main)
    command_context = _build_command_context(main)

    assert agents == {"peer": peer, "main": main}
    assert command_context.context.agent_provider._agent("peer") is peer


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
    assert "model doctor" in result


@pytest.mark.asyncio
async def test_run_slash_command_models_uses_manager_catalog(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/models catalog anthropic")

    assert "# models.catalog" in result
    assert "Provider: Anthropic" in result


@pytest.mark.asyncio
async def test_run_slash_command_model_task_budget_routes_to_task_budget_handler(
    tmp_path: Path,
) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)
    agent.llm = _TaskBudgetLlm()
    agent._llm = agent.llm

    result = await _run_slash_command_call(agent, "/model task_budget 64k")

    assert "# model.task\\_budget" in result
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
        await _run_slash_command_call(agent, "/mcp connect --timeout 0 npx demo-server")


@pytest.mark.parametrize("subcommand", ["disconnect", "reconnect"])
@pytest.mark.asyncio
async def test_run_slash_command_mcp_server_commands_require_one_name(
    tmp_path: Path,
    subcommand: str,
) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    with pytest.raises(AgentConfigError, match=f"Invalid /mcp {subcommand} arguments"):
        await _run_slash_command_call(agent, f"/mcp {subcommand} one two")


@pytest.mark.asyncio
async def test_run_slash_command_mcp_help_returns_usage(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/mcp --help")

    assert "Usage: /mcp" in result
    assert "- /mcp connect <target> [--name <server>]" in result


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

    assert "# commands skills" in result
    assert (
        "Usage: `/skills [list|available|search|add|remove|update|registry|help] [args]`" in result
    )


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
    assert "`/session`" in result


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

    assert "# commands cards" in result
    assert "Usage: `/cards [list|add|remove|readme|update|publish|registry|help] [args]`" in result


@pytest.mark.asyncio
async def test_run_slash_command_cards_publish_help(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/cards publish --help")

    assert "# commands cards publish" in result
    assert "`--no-push`" in result


@pytest.mark.asyncio
async def test_run_slash_command_skills_add_help(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/skills add --help")

    assert "# commands skills add" in result
    assert (
        "`/skills add [<number|name|github-url|path>] [--registry url] [--skills-dir path]`"
        in result
    )
    assert "`--skills-dir path`" in result


@pytest.mark.asyncio
async def test_run_slash_command_session_export_supports_hf_options(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(
        agent,
        "/session export latest --hf-dataset-path exports/",
    )

    assert "# session.export" in result
    assert "--hf-dataset-path requires --hf-dataset." in result


@pytest.mark.asyncio
async def test_run_slash_command_session_export_help(tmp_path: Path) -> None:
    settings = Settings(environment_dir=str(tmp_path / ".fast-agent"))
    agent = _SmartAgentStub(settings=settings)

    result = await _run_slash_command_call(agent, "/session export --help")

    assert "# session export" in result
    assert "file path, not a directory path" in result
    assert "`--hf-dataset-path path`" in result
