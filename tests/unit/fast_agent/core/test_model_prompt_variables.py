"""Model identity placeholders follow the attached LLM, not the configured alias."""

from typing import TYPE_CHECKING

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.config import OpenAISettings, Settings
from fast_agent.context import Context
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.core.instruction_utils import (
    apply_instruction_context,
    build_agent_instruction_context,
)
from fast_agent.llm.model_factory import ModelFactory

if TYPE_CHECKING:
    from pathlib import Path

TEMPLATE = "Agent={{agentName}} Model={{modelName}} Provider={{modelProvider}}"


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [LlmAgent, McpAgent])
async def test_model_placeholders_follow_attachment_and_switch(
    agent_type: type[LlmAgent],
) -> None:
    agent = agent_type(
        AgentConfig("parent", instruction=TEMPLATE, model="$models.primary", skills=[]),
        context=Context(
            config=Settings(
                responses=OpenAISettings(api_key="test"),
                model_references={"models": {"primary": "responses.gpt-5-mini?verbosity=low"}},
            )
        ),
    )
    try:
        await agent.attach_llm(ModelFactory.create_factory("passthrough"))
        assert agent.instruction == "Agent=parent Model=passthrough Provider=fast-agent"

        await agent.set_model("$models.primary")
        assert agent.instruction == "Agent=parent Model=gpt-5-mini Provider=responses"
        assert agent.llm is not None
        assert agent.llm.instruction == agent.instruction
        assert agent.llm.default_request_params.system_prompt == agent.instruction

        await apply_instruction_context(
            [agent], {"modelName": "stale-parent-model", "modelProvider": "stale-provider"}
        )
        assert agent.instruction == "Agent=parent Model=gpt-5-mini Provider=responses"

        if isinstance(agent, McpAgent):
            await rebuild_agent_instruction(agent)
            assert agent.instruction == "Agent=parent Model=gpt-5-mini Provider=responses"
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [LlmAgent, McpAgent])
async def test_isolated_child_uses_its_own_model_identity(agent_type: type[LlmAgent]) -> None:
    parent = agent_type(AgentConfig("parent", instruction=TEMPLATE, skills=[]))
    child = None
    try:
        await parent.attach_llm(ModelFactory.create_factory("passthrough"))
        child = await parent.spawn_isolated_instance(name="child", model="playback")
        assert child.instruction == "Agent=child Model=playback Provider=fast-agent"
        assert parent.instruction == "Agent=parent Model=passthrough Provider=fast-agent"
    finally:
        if child is not None:
            await child.shutdown()
        await parent.shutdown()


def test_unattached_agent_has_no_model_identity() -> None:
    agent = LlmAgent(AgentConfig("unattached", instruction=TEMPLATE, model="$models.primary"))
    context = build_agent_instruction_context(agent)
    assert context["modelName"] == ""
    assert context["modelProvider"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [LlmAgent, McpAgent])
async def test_included_model_identity_preserves_shared_context(
    agent_type: type[LlmAgent], tmp_path: "Path"
) -> None:
    (tmp_path / "identity.md").write_text(
        "Model={{modelName}} Provider={{modelProvider}} Root={{workspaceRoot}}",
        encoding="utf-8",
    )
    parent = agent_type(AgentConfig("parent", instruction="{{file:identity.md}}", skills=[]))
    parent.set_instruction_context({"workspaceRoot": str(tmp_path)})
    child = None
    try:
        await parent.attach_llm(ModelFactory.create_factory("passthrough"))
        assert parent.instruction == f"Model=passthrough Provider=fast-agent Root={tmp_path}"

        await parent.set_model("playback")
        assert parent.instruction == f"Model=playback Provider=fast-agent Root={tmp_path}"

        child = await parent.spawn_isolated_instance(name="child", model="passthrough")
        assert child.instruction == f"Model=passthrough Provider=fast-agent Root={tmp_path}"

        if isinstance(parent, McpAgent):
            await rebuild_agent_instruction(parent)
            assert parent.instruction == f"Model=playback Provider=fast-agent Root={tmp_path}"
    finally:
        if child is not None:
            await child.shutdown()
        await parent.shutdown()
