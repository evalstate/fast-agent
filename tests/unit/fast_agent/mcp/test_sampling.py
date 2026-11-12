import pytest
from mcp_types import CreateMessageRequestParams, CreateMessageResult, SamplingMessage, TextContent

from fast_agent.config import MCPSamplingSettings, MCPServerSettings
from fast_agent.mcp import sampling
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.sampling import sampling_agent_config
from fast_agent.types.llm_stop_reason import LlmStopReason


def test_build_sampling_agent_config_with_system_prompt():
    """Test building AgentConfig with system prompt from params"""
    # Create params with system prompt
    params = CreateMessageRequestParams(
        max_tokens=1024,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
        system_prompt="Custom system instruction",
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
        max_tokens=512,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
        system_prompt="",
    )

    # Build config
    config = sampling_agent_config(params)

    # Verify instruction is the empty string as received in params.systemPrompt
    assert config.instruction == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("llm_stop_reason", "expected"),
    [
        (LlmStopReason.MAX_TOKENS, "maxTokens"),
        (LlmStopReason.STOP_SEQUENCE, "stopSequence"),
        (LlmStopReason.END_TURN, "endTurn"),
    ],
)
async def test_sample_reports_why_generation_stopped(
    monkeypatch: pytest.MonkeyPatch, llm_stop_reason: LlmStopReason, expected: str
) -> None:
    class _StoppedLLM:
        async def generate(self, *_args: object, **_kwargs: object):
            return Prompt.assistant("partial", stop_reason=llm_stop_reason)

    monkeypatch.setattr(sampling, "create_sampling_llm", lambda *_args: _StoppedLLM())
    params = CreateMessageRequestParams(
        max_tokens=5,
        messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello"))],
    )

    result = await sampling.sample(
        params,
        server_name="test",
        server_config=MCPServerSettings(sampling=MCPSamplingSettings(model="passthrough")),
        agent_model=None,
        api_key=None,
        app_context=None,
    )

    assert isinstance(result, CreateMessageResult)
    assert result.stop_reason == expected
