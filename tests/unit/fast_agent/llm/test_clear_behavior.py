import pytest
from mcp.types import GetPromptResult, PromptMessage, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.context import Context
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def _make_template_prompt(text: str) -> GetPromptResult:
    return GetPromptResult(
        description="template",
        messages=[PromptMessage(role="assistant", content=TextContent(type="text", text=text))],
    )


def _make_user_message(text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="user", content=[TextContent(type="text", text=text)]
    )


@pytest.mark.asyncio
async def test_llm_clear_retains_templates():
    ctx = Context()
    llm = PassthroughLLM(provider=Provider.FAST_AGENT, context=ctx)

    await llm.apply_prompt_template(_make_template_prompt("template context"), "demo")
    assert [msg.first_text() for msg in llm.message_history] == ["template context"]

    await llm.generate([_make_user_message("hello")])
    assert len(llm.message_history) >= 3  # template + user + assistant

    llm.clear()
    assert [msg.first_text() for msg in llm.message_history] == ["template context"]
    assert len(llm.history.get()) == 1

    llm.clear(clear_prompts=True)
    assert llm.message_history == []
    assert llm.history.get() == []


@pytest.mark.asyncio
async def test_agent_clear_delegates_to_llm():
    ctx = Context()
    agent = LlmAgent(config=AgentConfig(name="agent-under-test"), context=ctx)
    llm = PassthroughLLM(provider=Provider.FAST_AGENT, context=ctx)
    agent._llm = llm

    await agent.apply_prompt_template(_make_template_prompt("agent template"), "tmpl")
    await agent.generate(_make_user_message("hi"))
    assert len(agent.message_history) >= 3

    agent.clear()
    assert [msg.first_text() for msg in agent.message_history] == ["agent template"]

    agent.clear(clear_prompts=True)
    assert agent.message_history == []
    assert llm.message_history == []
