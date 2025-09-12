from mcp.types import CreateMessageRequestParams, SamplingMessage, TextContent

from fast_agent.mcp.sampling import sampling_agent_config


def test_build_sampling_agent_config_with_system_prompt():
    """Test building AgentConfig with system prompt from params"""
    # Create params with system prompt
    params = CreateMessageRequestParams(
        maxTokens=1024,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
        systemPrompt="Custom system instruction",
    )

    # Build config
    config = sampling_agent_config(params)

    # Verify instruction is set from systemPrompt
    assert config.name == "sampling_agent"
    assert config.instruction == "Custom system instruction"
    assert config.servers == []


def test_build_sampling_agent_config_default():
    """Test building AgentConfig with default values"""
    # Build config with no params
    config = sampling_agent_config(None)

    # Verify default instruction
    assert config.name == "sampling_agent"
    assert config.instruction == "You are a helpful AI Agent."
    assert config.servers == []


def test_build_sampling_agent_config_empty_system_prompt():
    """Test building AgentConfig with empty system prompt"""
    # Create params with empty system prompt
    params = CreateMessageRequestParams(
        maxTokens=512,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
        systemPrompt="",
    )

    # Build config
    config = sampling_agent_config(params)

    # Verify instruction is the empty string as received in params.systemPrompt
    assert config.instruction == ""
