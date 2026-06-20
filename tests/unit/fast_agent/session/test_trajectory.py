from datetime import UTC, datetime

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.prompt import Prompt
from fast_agent.session.session_manager import Session, SessionInfo
from fast_agent.session.trajectory import TrajectoryRecord, save_trajectory_record


def test_save_trajectory_requires_stateless_agent() -> None:
    with pytest.raises(AgentConfigError, match="save_trajectory requires use_history=False"):
        AgentConfig(name="agent", use_history=True, save_trajectory=True)


@pytest.mark.asyncio
async def test_save_trajectory_record_writes_replay_fields(tmp_path) -> None:
    now = datetime.now(UTC)
    session = Session(
        SessionInfo(name="session-1", created_at=now, last_activity=now),
        tmp_path,
    )
    record = TrajectoryRecord(
        trajectory_id="traj_1",
        session_id=session.info.name,
        parent_agent_name="parent",
        agent_name="child[1]",
        template_agent_name="child",
        tool_name="agent__child",
        parent_tool_call_id="call_1",
        use_history=False,
        started_at="2026-06-20T12:00:00Z",
        completed_at="2026-06-20T12:00:01Z",
        tool_input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        tool_arguments={"query": "hello", "response_mode": "passthrough"},
        effective_tool_arguments={"query": "hello"},
        rendered_child_input='{"query": "hello"}',
        messages=[Prompt.user('{"query": "hello"}'), Prompt.assistant("done")],
    )

    path = await save_trajectory_record(session, record)

    assert path.parent == tmp_path / "trajectories"
    text = path.read_text(encoding="utf-8")
    assert '"tool_input_schema"' in text
    assert '"tool_arguments"' in text
    assert '"effective_tool_arguments"' in text
    assert '"messages"' in text
