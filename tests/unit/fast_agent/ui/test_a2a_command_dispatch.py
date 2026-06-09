from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from a2a.types import AgentCard, AgentInterface, AgentProvider

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.core.agent_app import AgentApp
from fast_agent.ui.command_payloads import A2ACommand
from fast_agent.ui.interactive import command_dispatch
from fast_agent.ui.interactive.command_dispatch import dispatch_command_payload
from fast_agent.ui.interactive_prompt import InteractivePrompt

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _SelectedTransport:
    pass


def _remote_agent(*, name: str = "remote") -> A2ARemoteAgent:
    agent = A2ARemoteAgent(
        config=AgentConfig(name=name, agent_type=AgentType.A2A, use_history=True),
        a2a_config=A2AAgentConfig(url="http://127.0.0.1:41242", transport="JSONRPC"),
    )
    agent.context_id = "ctx-current"
    agent.current_task_id = "task-current"
    agent.last_task_state = "TASK_STATE_INPUT_REQUIRED"
    agent.task_states = {
        "task-done": "TASK_STATE_COMPLETED",
        "task-current": "TASK_STATE_INPUT_REQUIRED",
    }
    agent.remote_card = AgentCard(
        name="Remote A2A",
        description="Deterministic remote A2A agent.",
        provider=AgentProvider(organization="tests", url="https://example.com"),
        version="1.0",
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url="http://127.0.0.1:41242/a2a/jsonrpc",
            )
        ],
    )
    agent._client = _SelectedTransport()
    return agent


async def _dispatch(
    owner: InteractivePrompt,
    app: AgentApp,
    payload: A2ACommand,
) -> command_dispatch.DispatchResult:
    return await dispatch_command_payload(
        owner,
        payload,
        prompt_provider=app,
        agent="remote",
        available_agents=list(app.registered_agents()),
        available_agents_set=set(app.registered_agents()),
        merge_pinned_agents=lambda names: names,
    )


def _app(agents: dict[str, object]) -> AgentApp:
    return AgentApp(cast("dict[str, AgentProtocol]", agents))


@pytest.mark.asyncio
async def test_a2a_tui_dispatch_reports_status_transport_and_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        command_dispatch,
        "rich_print",
        lambda value="", *args, **kwargs: printed.append(str(value)),
    )
    remote = _remote_agent()
    app = _app({"remote": remote})
    owner = InteractivePrompt(agent_types={"remote": AgentType.A2A})

    status = await _dispatch(owner, app, A2ACommand(action="status", argument=None))
    transport = await _dispatch(owner, app, A2ACommand(action="transport", argument="remote"))
    card = await _dispatch(owner, app, A2ACommand(action="card", argument="remote"))

    assert status.handled
    assert transport.handled
    assert card.handled
    output = "\n".join(printed)
    assert "A2A status: remote" in output
    assert "Context: ctx-current" in output
    assert "Task: task-current" in output
    assert "Last state: TASK_STATE_INPUT_REQUIRED" in output
    assert "Tasks: 1 finished, 1 pending" in output
    assert "A2A transport: remote" in output
    assert "Requested: JSONRPC" in output
    assert "Selected client: _SelectedTransport" in output
    assert "A2A card: Remote A2A" in output
    assert "JSONRPC 1.0: http://127.0.0.1:41242/a2a/jsonrpc" in output


@pytest.mark.asyncio
async def test_a2a_tui_dispatch_reports_tasks_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        command_dispatch,
        "rich_print",
        lambda value="", *args, **kwargs: printed.append(str(value)),
    )
    remote = _remote_agent()
    app = _app({"remote": remote})
    owner = InteractivePrompt(agent_types={"remote": AgentType.A2A})

    result = await _dispatch(owner, app, A2ACommand(action="tasks", argument="remote"))

    assert result.handled
    output = "\n".join(printed)
    assert "A2A tasks: remote" in output
    assert "Tasks: 1 finished, 1 pending" in output


@pytest.mark.asyncio
async def test_a2a_tui_dispatch_lists_and_resets_remote_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        command_dispatch,
        "rich_print",
        lambda value="", *args, **kwargs: printed.append(str(value)),
    )
    remote = _remote_agent()
    local = object()
    app = _app({"remote": remote, "local": local})
    owner = InteractivePrompt(agent_types={"remote": AgentType.A2A, "local": AgentType.BASIC})

    listed = await _dispatch(owner, app, A2ACommand(action="list", argument=None))
    reset = await _dispatch(owner, app, A2ACommand(action="reset", argument="remote"))

    assert listed.handled
    assert reset.handled
    assert "  • remote" in printed
    assert all("local" not in line for line in printed)
    assert remote.context_id is None
    assert remote.current_task_id is None
    assert remote.last_task_state is None
    assert remote.task_states == {}


@pytest.mark.asyncio
async def test_a2a_tui_dispatch_rejects_a2a_commands_for_local_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        command_dispatch,
        "rich_print",
        lambda value="", *args, **kwargs: printed.append(str(value)),
    )
    app = _app({"local": object()})
    owner = InteractivePrompt(agent_types={"local": AgentType.BASIC})

    result = await dispatch_command_payload(
        owner,
        A2ACommand(action="status", argument="local"),
        prompt_provider=app,
        agent="local",
        available_agents=["local"],
        available_agents_set={"local"},
        merge_pinned_agents=lambda names: names,
    )

    assert result.handled
    assert "Agent 'local' is not an A2A agent." in "\n".join(printed)
