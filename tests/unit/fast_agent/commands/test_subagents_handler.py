import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.subagent_tool import install_subagent_tool, subagent_tool_enabled
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.subagents import handle_subagents_command
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.results import CommandMessage
from fast_agent.session import SessionChildLinkSnapshot, SessionManager


class _IO(NonInteractiveCommandIOBase):
    async def emit(self, message: CommandMessage) -> None:
        del message


def _context(tmp_path, agent: ToolAgent) -> tuple[CommandContext, SessionManager]:
    manager = SessionManager(
        cwd=tmp_path,
        home_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    manager.create_session()
    return (
        CommandContext(
            agent_provider=StaticAgentProvider({"main": agent}),
            current_agent_name="main",
            io=_IO(),
            session_manager=manager,
        ),
        manager,
    )


@pytest.mark.asyncio
async def test_subagents_list_renders_persisted_aliases(tmp_path) -> None:
    agent = ToolAgent(AgentConfig("main", subagents=True))
    context, manager = _context(tmp_path, agent)
    parent = manager.current_session
    assert parent is not None
    manager.create_child_session(
        parent,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
        ),
        alias_slug="investigate_item",
        label="Investigate item",
        task_preview="Investigate item",
    ).set_execution_status("completed")

    outcome = await handle_subagents_command(context, agent_name="main", action="list")

    rendered = outcome.messages[0].plain_text()
    assert "01_investigate_item" in rendered
    assert "completed" in rendered
    assert "Investigate item" in rendered
    markdown = render_command_outcome_markdown(outcome, heading="subagents")
    assert "| Alias | Status | Agent | Task |" in markdown
    assert "| 01_investigate_item | completed | main | Investigate item |" in markdown


@pytest.mark.asyncio
async def test_subagents_runtime_toggle_adds_and_removes_tool(tmp_path) -> None:
    agent = ToolAgent(AgentConfig("main", subagents=True))
    assert install_subagent_tool(agent)
    context, _manager = _context(tmp_path, agent)

    off = await handle_subagents_command(context, agent_name="main", action="off")
    assert not subagent_tool_enabled(agent)
    assert "disabled" in off.messages[0].plain_text()

    on = await handle_subagents_command(context, agent_name="main", action="on")
    assert subagent_tool_enabled(agent)
    assert agent.config.subagent_activation_source == "runtime"
    assert "enabled" in on.messages[0].plain_text()


@pytest.mark.asyncio
async def test_subagents_runtime_enable_respects_explicit_configuration_disable(tmp_path) -> None:
    agent = ToolAgent(AgentConfig("main", subagents=False))
    context, _manager = _context(tmp_path, agent)

    outcome = await handle_subagents_command(context, agent_name="main", action="on")

    assert not subagent_tool_enabled(agent)
    assert "disabled by configuration" in outcome.messages[0].plain_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["configuration", "cli"])
async def test_subagents_off_does_not_erase_explicit_disable(tmp_path, source) -> None:
    config = AgentConfig("main", subagents=False)
    config.subagent_activation_source = source
    agent = ToolAgent(config)
    context, _manager = _context(tmp_path, agent)

    await handle_subagents_command(context, agent_name="main", action="off")
    outcome = await handle_subagents_command(context, agent_name="main", action="on")

    assert not subagent_tool_enabled(agent)
    assert agent.config.subagent_activation_source == source
    assert f"disabled by {source}" in outcome.messages[0].plain_text()


@pytest.mark.asyncio
async def test_subagents_runtime_enable_preserves_user_tool_name_collision(tmp_path) -> None:
    def subagent() -> str:
        return "custom"

    agent = ToolAgent(AgentConfig("main"), [subagent])
    context, _manager = _context(tmp_path, agent)

    outcome = await handle_subagents_command(context, agent_name="main", action="on")

    assert not subagent_tool_enabled(agent)
    assert agent.config.subagents is None
    assert agent.config.subagent_activation_source is None
    assert "Failed to enable" in outcome.messages[0].plain_text()


@pytest.mark.asyncio
async def test_subagents_list_includes_legacy_child_sessions(tmp_path) -> None:
    agent = ToolAgent(AgentConfig("main", subagents=True))
    context, manager = _context(tmp_path, agent)
    parent = manager.current_session
    assert parent is not None
    child = manager.create_child_session(
        parent,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
        ),
    )

    outcome = await handle_subagents_command(context, agent_name="main", action="list")

    assert f"legacy_{child.info.name}" in outcome.messages[0].plain_text()
