from __future__ import annotations

from fast_agent.a2a.config import A2AAgentConfig
from fast_agent.a2a.remote_agent import A2ARemoteAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.types import RequestParams


def test_a2a_use_history_falls_back_to_agent_config_when_request_defaulted() -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="http://example.test"),
    )

    assert agent._resolve_turn_use_history(RequestParams(maxTokens=100)) is False


def test_a2a_use_history_respects_explicit_request_override() -> None:
    agent = A2ARemoteAgent(
        config=AgentConfig(name="remote", agent_type=AgentType.A2A, use_history=False),
        a2a_config=A2AAgentConfig(url="http://example.test"),
    )

    assert agent._resolve_turn_use_history(RequestParams(use_history=True)) is True
