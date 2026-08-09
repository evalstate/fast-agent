from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import Settings, SkillsSettings
from fast_agent.context import Context
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.harness_tools import (
    GET_RESOURCE_TOOL_NAME,
    HARNESS_TOOL_NAMES,
    SLASH_COMMAND_TOOL_NAME,
    harness_tools_enabled,
    set_harness_tools,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult
from fast_agent.skills.mcp_registry import McpRegistrySkill, McpSkillRegistry
from fast_agent.tools.function_tool_loader import build_default_function_tool

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions


def _tool_names(agent: ToolAgent) -> set[str]:
    return set(agent._execution_tools)


class _SimulatedMcpAgent(McpAgent):
    def __init__(self, *, shell: bool = True) -> None:
        super().__init__(
            config=AgentConfig(
                "dev",
                instruction="System details",
                harness_tools=True,
                shell=shell,
            ),
            context=Context(config=Settings()),
        )
        self.attached_servers: list[str] = []
        self.last_server_config: MCPServerSettings | None = None
        self.last_options: MCPAttachOptions | None = None

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        if server_name == "broken":
            raise RuntimeError("simulated connection failure")
        self.last_server_config = server_config
        self.last_options = options
        already_attached = server_name in self.attached_servers
        force_reconnect = options is not None and options.force_reconnect
        if not already_attached:
            self.attached_servers.append(server_name)
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=already_attached and not force_reconnect,
            tools_added=[] if already_attached else [f"{server_name}.echo"],
            prompts_added=[],
            warnings=[],
        )

    async def detach_mcp_server(self, server_name: str) -> MCPDetachResult:
        detached = server_name in self.attached_servers
        if detached:
            self.attached_servers.remove(server_name)
        return MCPDetachResult(
            server_name=server_name,
            detached=detached,
            tools_removed=[f"{server_name}.echo"] if detached else [],
            prompts_removed=[],
        )

    def list_attached_mcp_servers(self) -> list[str]:
        return list(self.attached_servers)


class _SkillsCatalogAggregator:
    async def list_mcp_skill_registries(self) -> list[McpSkillRegistry]:
        return [
            McpSkillRegistry(
                server_name="hf",
                server_version="1.0",
                skills=[
                    McpRegistrySkill(
                        name=f"skill-{index:02}",
                        description=f"Skill {index} description. " * 30,
                        uri=f"skill://skill-{index:02}/SKILL.md",
                        server_name="hf",
                    )
                    for index in range(1, 26)
                ],
            )
        ]


class _SkillsCatalogAgent(ToolAgent):
    @property
    def aggregator(self) -> _SkillsCatalogAggregator:
        return _SkillsCatalogAggregator()


@pytest.mark.asyncio
async def test_harness_tools_install_execute_and_disable() -> None:
    agent = ToolAgent(AgentConfig("dev", instruction="System details", harness_tools=True))

    assert not harness_tools_enabled(agent)
    assert set_harness_tools(agent, True)
    assert harness_tools_enabled(agent)
    assert HARNESS_TOOL_NAMES <= _tool_names(agent)
    assert agent._execution_tools[SLASH_COMMAND_TOOL_NAME].parameters["properties"]["command"] == {
        "type": "string"
    }
    assert agent._execution_tools[GET_RESOURCE_TOOL_NAME].parameters["properties"]["uri"] == {
        "type": "string"
    }

    commands = await agent.call_tool(SLASH_COMMAND_TOOL_NAME, {"command": "/commands"})
    resource = await agent.call_tool(
        GET_RESOURCE_TOOL_NAME,
        {"uri": "internal://fast-agent/agent-cards"},
    )

    commands_text = get_text(commands.content[0])
    resource_text = get_text(resource.content[0])
    assert commands_text is not None
    assert resource_text is not None
    assert "/usage" in commands_text
    assert "/mcp" in commands_text
    assert "/skills" in commands_text
    assert "/status" in commands_text
    assert "AgentCard" in resource_text

    for command, expected in (
        ("/commands mcp", "Usage: `/mcp"),
        ("/commands mcp attach", "Usage: `/mcp attach <server-name>`"),
        ("/commands --json", '"kind": "command_index"'),
        ("/commands mcp attach --json", '"kind": "command_action_detail"'),
    ):
        discovered = await agent.call_tool(SLASH_COMMAND_TOOL_NAME, {"command": command})
        discovered_text = get_text(discovered.content[0])
        assert not discovered.is_error
        assert discovered_text is not None
        assert expected in discovered_text

    async def echo(value: str) -> str:
        return value

    agent.add_tool(build_default_function_tool(echo))
    tool_detail = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/tools echo"},
    )
    tool_detail_text = get_text(tool_detail.content[0])
    assert not tool_detail.is_error
    assert tool_detail_text is not None
    assert "# Tool schema: echo" in tool_detail_text
    assert '"value"' in tool_detail_text

    for command, expected in (
        ("/usage", "No usage data available."),
        ("/system", "System details"),
        ("/status", "No MCP status is available for this agent."),
    ):
        result = await agent.call_tool(SLASH_COMMAND_TOOL_NAME, {"command": command})
        result_text = get_text(result.content[0])
        assert not result.is_error
        assert result_text is not None
        assert expected in result_text

    removed_status_alias = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcpstatus"},
    )
    removed_status_alias_text = get_text(removed_status_alias.content[0])
    assert removed_status_alias.is_error
    assert removed_status_alias_text is not None
    assert "Unsupported harness command" in removed_status_alias_text

    unavailable = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp connect https://example.com/mcp"},
    )
    unavailable_text = get_text(unavailable.content[0])
    assert unavailable.is_error
    assert unavailable_text is not None
    assert "does not support runtime MCP server management" in unavailable_text

    invalid_skills = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": '/skills "unterminated'},
    )
    invalid_skills_text = get_text(invalid_skills.content[0])
    assert invalid_skills.is_error
    assert invalid_skills_text is not None
    assert "Invalid /skills arguments" in invalid_skills_text

    skills_help = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/skills --help"},
    )
    skills_help_text = get_text(skills_help.content[0])
    assert not skills_help.is_error
    assert skills_help_text is not None
    assert "/skills \\[list|available|search|add|remove|update|registry|help\\]" in skills_help_text

    assert not set_harness_tools(agent, False)
    assert not harness_tools_enabled(agent)
    assert HARNESS_TOOL_NAMES.isdisjoint(_tool_names(agent))


@pytest.mark.asyncio
async def test_harness_tools_manage_mcp_servers() -> None:
    agent = _SimulatedMcpAgent()
    set_harness_tools(agent, True)

    connected = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp connect --name demo npx demo-server"},
    )
    connected_text = get_text(connected.content[0])
    assert not connected.is_error
    assert connected_text is not None
    assert "Connected MCP server 'demo'" in connected_text
    assert "(npx)" in connected_text
    assert agent.attached_servers == ["demo"]
    assert agent.last_server_config is not None
    assert agent.last_options is not None
    assert agent.last_options.trigger_oauth is False
    assert agent.last_options.allow_oauth_paste_fallback is False

    listed = await agent.call_tool(SLASH_COMMAND_TOOL_NAME, {"command": '/mcp "list"'})
    listed_text = get_text(listed.content[0])
    assert not listed.is_error
    assert listed_text is not None
    assert "Attached MCP servers: demo" in listed_text

    oauth = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp connect --oauth https://example.com/mcp"},
    )
    oauth_text = get_text(oauth.content[0])
    assert oauth.is_error
    assert oauth_text is not None
    assert "Interactive MCP OAuth is unavailable" in oauth_text

    failed = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp connect --name broken npx broken-server"},
    )
    failed_text = get_text(failed.content[0])
    assert failed.is_error
    assert failed_text is not None
    assert "simulated connection failure" in failed_text

    reconnected = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp reconnect demo"},
    )
    reconnected_text = get_text(reconnected.content[0])
    assert not reconnected.is_error
    assert reconnected_text is not None
    assert "Reconnected MCP server 'demo'" in reconnected_text
    assert "via stdio" in reconnected_text

    disconnected = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp disconnect demo"},
    )
    disconnected_text = get_text(disconnected.content[0])
    assert not disconnected.is_error
    assert disconnected_text is not None
    assert "Disconnected MCP server 'demo'" in disconnected_text
    assert agent.attached_servers == []


@pytest.mark.asyncio
async def test_harness_command_discovery_drilldowns_are_available() -> None:
    agent = ToolAgent(AgentConfig("dev", harness_tools=True))
    set_harness_tools(agent, True)

    index = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/commands --json"},
    )
    index_text = get_text(index.content[0])
    assert not index.is_error
    assert index_text is not None
    payload = json.loads(index_text)
    assert isinstance(payload, dict)
    assert '"name": "--oauth"' not in index_text
    assert '"name": "--no-oauth"' in index_text
    commands = payload["commands"]
    assert isinstance(commands, list)

    for command_entry in commands:
        assert isinstance(command_entry, dict)
        command_name = command_entry["name"]
        assert isinstance(command_name, str)
        detail = await agent.call_tool(
            SLASH_COMMAND_TOOL_NAME,
            {"command": f"/commands {command_name}"},
        )
        assert not detail.is_error, get_text(detail.content[0])

        actions = command_entry["actions"]
        assert isinstance(actions, list)
        for action_entry in actions:
            assert isinstance(action_entry, dict)
            action_name = action_entry["name"]
            assert isinstance(action_name, str)
            action_detail = await agent.call_tool(
                SLASH_COMMAND_TOOL_NAME,
                {"command": f"/commands {command_name} {action_name}"},
            )
            assert not action_detail.is_error, get_text(action_detail.content[0])


@pytest.mark.asyncio
async def test_harness_skills_catalog_supports_one_shot_mcp_json() -> None:
    agent = _SkillsCatalogAgent(AgentConfig("dev", harness_tools=True))
    set_harness_tools(agent, True)

    result = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": ("/skills available --registry hf --limit 10 --page 2 --json")},
    )
    result_text = get_text(result.content[0])

    assert not result.is_error
    assert result_text is not None
    payload = json.loads(result_text)
    assert payload["source"]["server_name"] == "hf"
    assert payload["page"] == 2
    assert payload["total"] == 25
    assert payload["skills"][0]["name"] == "skill-11"
    assert not result_text.startswith("# skills")


@pytest.mark.asyncio
async def test_harness_mcp_connect_protects_host_execution_and_credentials() -> None:
    agent = _SimulatedMcpAgent(shell=False)
    set_harness_tools(agent, True)

    stdio = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp connect --name run sh -c 'echo unsafe'"},
    )
    stdio_text = get_text(stdio.content[0])
    assert stdio.is_error
    assert stdio_text is not None
    assert "Shell access is required" in stdio_text
    assert agent.attached_servers == []

    auth = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/mcp connect --auth '$OPENAI_API_KEY' https://example.com/mcp"},
    )
    auth_text = get_text(auth.content[0])
    assert auth.is_error
    assert auth_text is not None
    assert "Environment-backed MCP auth is unavailable" in auth_text
    assert agent.attached_servers == []


@pytest.mark.asyncio
async def test_harness_tools_manage_installed_skills(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "alpha"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: alpha\ndescription: Test skill\n---\n\nUse alpha.\n",
        encoding="utf-8",
    )
    settings = Settings(skills=SkillsSettings(directories=[str(skills_dir)]))
    agent = McpAgent(
        config=AgentConfig("dev", instruction="{{agentSkills}}", harness_tools=True),
        context=Context(config=settings),
    )
    set_harness_tools(agent, True)

    listed = await agent.call_tool(SLASH_COMMAND_TOOL_NAME, {"command": "/skills list"})
    listed_text = get_text(listed.content[0])
    assert not listed.is_error
    assert listed_text is not None
    assert "alpha" in listed_text

    selection = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/skills remove"},
    )
    selection_text = get_text(selection.content[0])
    assert not selection.is_error
    assert selection_text is not None
    assert "Remove with \\`/skills remove <number|name>\\`." in selection_text
    assert skill_dir.exists()

    removed = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/skills remove alpha"},
    )
    removed_text = get_text(removed.content[0])
    assert not removed.is_error
    assert removed_text is not None
    assert "Removed skill: alpha" in removed_text
    assert not skill_dir.exists()


@pytest.mark.asyncio
async def test_harness_skill_management_respects_explicit_empty_skills(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "alpha"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: alpha\ndescription: Test skill\n---\n",
        encoding="utf-8",
    )
    agent = McpAgent(
        config=AgentConfig(
            "dev",
            instruction="{{agentSkills}}",
            harness_tools=True,
            skills=[],
        ),
        context=Context(config=Settings(skills=SkillsSettings(directories=[str(skills_dir)]))),
    )
    set_harness_tools(agent, True)

    removed = await agent.call_tool(
        SLASH_COMMAND_TOOL_NAME,
        {"command": "/skills remove alpha"},
    )

    assert not removed.is_error
    assert not skill_dir.exists()
    assert agent.skill_manifests == []
    assert not agent.shell_runtime_enabled


def test_harness_tools_do_not_replace_user_tools() -> None:
    def slash_command(command: str) -> str:
        return command

    agent = ToolAgent(
        AgentConfig("dev"),
        tools=[build_default_function_tool(slash_command)],
    )

    with pytest.raises(AgentConfigError, match="reserved"):
        set_harness_tools(agent, True)

    assert agent.config.harness_tools is False
    assert SLASH_COMMAND_TOOL_NAME in _tool_names(agent)


@pytest.mark.asyncio
async def test_isolated_clone_does_not_inherit_harness_tools() -> None:
    parent = ToolAgent(AgentConfig("dev", harness_tools=True))
    set_harness_tools(parent, True)

    clone = await parent.spawn_isolated_instance()
    try:
        assert clone.config.harness_tools is False
        assert HARNESS_TOOL_NAMES.isdisjoint(_tool_names(clone))
        assert HARNESS_TOOL_NAMES <= _tool_names(parent)
    finally:
        await clone.shutdown()
