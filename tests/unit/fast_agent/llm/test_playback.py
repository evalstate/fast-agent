import pytest
from pydantic import BaseModel

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.llm.internal.playback import PlaybackLLM
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.prompt import Prompt


class FormattedResponse(BaseModel):
    thinking: str
    message: str


sample_json = '{"thinking":"The user wants to have a conversation about guitars, which are a broad...","message":"Sure! I love talking about guitars."}'


@pytest.fixture
def llm() -> FastAgentLLMProtocol:
    return PlaybackLLM()


@pytest.mark.asyncio
async def test_model_factory_creates_playback():
    """Test that ModelFactory correctly creates a PlaybackLLM instance"""
    # Create a factory for the playback model
    factory = ModelFactory.create_factory("playback")

    # Verify the factory is callable
    assert callable(factory)

    # Create an instance using the factory

    instance = factory(
        LlmAgent(
            AgentConfig(name="playback_agent", instruction="Helpful AI Agent", servers=[]),
            context=None,
        )
    )

    assert isinstance(instance, PlaybackLLM)


@pytest.mark.asyncio
async def test_basic_playback_function(llm):
    """Test that ModelFactory correctly creates a PlaybackLLM instance"""
    result = await llm.generate([Prompt.user("hello, world!")])
    assert "HISTORY LOADED (1) messages" == result.first_text()


@pytest.mark.asyncio
async def test_simple_playback_functionality(llm):
    await llm.generate(
        [
            Prompt.user("message 1"),
            Prompt.assistant("response 1"),
            Prompt.user("message 2"),
            Prompt.assistant("response 2"),
        ],
    )
    response1 = await llm.generate([Prompt.user("evalstate")])
    response2 = await llm.generate([Prompt.user("llmindset")])
    assert "response 1" == response1.first_text()
    assert "response 2" == response2.first_text()


@pytest.mark.asyncio
async def test_exhaustion_behaviour(llm):
    await llm.generate(
        [
            Prompt.user("message 1"),
            Prompt.assistant("response 1"),
        ],
    )
    response1 = await llm.generate([Prompt.user("evalstate")])
    response2 = await llm.generate([Prompt.user("llmindset")])
    assert "response 1" == response1.first_text()
    assert "MESSAGES EXHAUSTED" in response2.first_text()
    assert "(0 overage)" in response2.first_text()

    for _ in range(3):
        overage = await llm.generate([Prompt.user("overage?")])
        assert f"({_ + 1} overage)" in overage.first_text()


@pytest.mark.asyncio
async def test_cannot_load_history_with_structured(llm):
    with pytest.raises(ModelConfigError):
        await llm.structured(
            [Prompt.user("use generate to load messages")], FormattedResponse, None
        )


@pytest.mark.asyncio
async def test_generates_structured(llm):
    await llm.generate([Prompt.user("jlyst guitars"), Prompt.assistant(sample_json)])
    model, response = await llm.structured(
        [Prompt.user("use generate to load messages")], FormattedResponse
    )
    assert (
        model.thinking
        == "The user wants to have a conversation about guitars, which are a broad..."
    )


@pytest.mark.asyncio
async def test_generates_structured_exhaustion_behaves(llm):
    # this is the same as the "bad JSON" scenario
    await llm.generate([Prompt.user("jlyst guitars"), Prompt.assistant(sample_json)])
    await llm.structured([Prompt.user("pop the stack")], FormattedResponse)

    model, response = await llm.structured([Prompt.user("exhausted stack")], FormattedResponse)
    assert model is None
    assert "MESSAGES EXHAUSTED" in response.first_text()


@pytest.mark.asyncio
async def test_playback_does_not_report_character_counts_as_tokens(llm):
    response1 = await llm.generate([Prompt.user("test"), Prompt.assistant("response1")])
    assert "HISTORY LOADED" in response1.first_text()
    await llm.generate([Prompt.user("next message")])

    assert llm.usage_accumulator.turns == []
    assert llm.usage_accumulator.summary.total is None
