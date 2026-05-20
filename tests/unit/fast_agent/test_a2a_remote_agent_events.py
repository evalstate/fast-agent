from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from a2a.types import Artifact, Part, StreamResponse, Task, TaskState, TaskStatus

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


async def _events(*events: StreamResponse) -> AsyncIterator[StreamResponse]:
    for event in events:
        yield event


def _remote_agent() -> A2ARemoteAgent:
    return A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="http://127.0.0.1:41242"),
    )


@pytest.mark.asyncio
async def test_a2a_remote_agent_clears_task_id_for_terminal_full_task_event() -> None:
    agent = _remote_agent()
    agent.current_task_id = "previous-task"

    result = await agent._consume_events(
        _events(
            StreamResponse(
                task=Task(
                    id="terminal-task",
                    context_id="ctx-1",
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                    artifacts=[Artifact(name="response", parts=[Part(text="done")])],
                )
            )
        )
    )

    assert result.text == "done"
    assert result.state == "TASK_STATE_COMPLETED"
    assert agent.context_id == "ctx-1"
    assert agent.last_task_state == "TASK_STATE_COMPLETED"
    assert agent.current_task_id is None


@pytest.mark.asyncio
async def test_a2a_remote_agent_keeps_task_id_for_input_required_full_task_event() -> None:
    agent = _remote_agent()

    result = await agent._consume_events(
        _events(
            StreamResponse(
                task=Task(
                    id="input-task",
                    context_id="ctx-2",
                    status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
                )
            )
        )
    )

    assert result.state == "TASK_STATE_INPUT_REQUIRED"
    assert agent.context_id == "ctx-2"
    assert agent.last_task_state == "TASK_STATE_INPUT_REQUIRED"
    assert agent.current_task_id == "input-task"
