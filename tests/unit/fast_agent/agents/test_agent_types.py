"""
Unit tests for agent types and their interactions with the interactive prompt.
"""

from fast_agent.agents import McpAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.types import RequestParams


def test_agent_type_default():
    """Test that agent_type defaults to AgentType.BASIC.value"""
    agent = McpAgent(config=AgentConfig(name="test_agent"))
    assert agent.agent_type == AgentType.BASIC


def test_instruction_propagates_to_default_request_params():
    """
    Test that AgentConfig.instruction is propagated to
    default_request_params.systemPrompt when both are provided.

    This reproduces the bug where the instruction is lost when
    a user provides their own default_request_params.
    """
    # Create RequestParams with custom settings but no systemPrompt
    request_params = RequestParams(
        model="sonnet",
        temperature=0.7,
        maxTokens=32768
    )

    # Verify systemPrompt is not set initially
    assert not hasattr(request_params, 'systemPrompt') or request_params.systemPrompt is None

    # Create AgentConfig with both instruction and default_request_params
    instruction = "You are a helpful assistant specialized in testing."
    config = AgentConfig(
        name="my_agent",
        instruction=instruction,
        default_request_params=request_params,
        model="sonnet"
    )

    # The instruction should be propagated to default_request_params.systemPrompt
    assert config.default_request_params.systemPrompt == instruction, (
        f"Expected systemPrompt to be '{instruction}', "
        f"but got {config.default_request_params.systemPrompt}"
    )
