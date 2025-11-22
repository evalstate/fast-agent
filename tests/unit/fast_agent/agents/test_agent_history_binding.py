import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.prompt import Prompt
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.types import PromptMessageExtended


class FakeLLM(FastAgentLLM[PromptMessageExtended, PromptMessageExtended]):
    def __init__(self, **kwargs):
        super().__init__(provider=Provider.FAST_AGENT, name="fake-llm", **kwargs)
        self.last_messages: list[PromptMessageExtended] | None = None

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools=None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        self.last_messages = list(multipart_messages)
        return Prompt.assistant("ok")

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: list[PromptMessageExtended],
        model,
        request_params: RequestParams | None = None,
    ):
        self.last_messages = list(multipart_messages)
        return None, Prompt.assistant("ok")

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[PromptMessageExtended]:
        return messages


@pytest.mark.asyncio
async def test_templates_sent_when_history_disabled():
    agent = LlmAgent(AgentConfig("test-agent"))
    llm = FakeLLM()
    agent._llm = llm

    # Seed a template baseline and make sure history mirrors it
    template_result = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="template baseline")],
    )
    agent._template_messages.append(template_result)
    agent._message_history.extend(agent._template_messages)

    user_msg = PromptMessageExtended(
        role="user", content=[TextContent(type="text", text="hello world")]
    )

    response = await agent.generate_impl([user_msg], RequestParams(use_history=False))

    assert llm.last_messages is not None
    assert llm.last_messages[0].first_text() == template_result.first_text()
    # History not extended when use_history is False (template remains)
    assert len(agent.message_history) == 1
    assert agent.message_history[0].first_text() == template_result.first_text()
    assert response.role == "assistant"
