"""
Unit tests for agent types and their interactions with the interactive prompt.
"""

from fast_agent.agents.agent_types import AgentConfig, AgentType
from mcp_agent.agents.agent import Agent


def test_agent_type_default():
    """Test that agent_type defaults to AgentType.BASIC.value"""
    agent = Agent(config=AgentConfig(name="test_agent"))
    assert agent.agent_type == AgentType.BASIC
