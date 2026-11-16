"""Input delivered by herdr-core is shown apart from what a person typed."""

from __future__ import annotations

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.prompt import Prompt


@pytest.mark.asyncio
async def test_hc_messages_get_their_own_panel_heading(capsys) -> None:
    agent = LlmAgent(AgentConfig(name="lead"))
    await agent.attach_llm(ModelFactory.create_factory("passthrough"))

    await agent.generate(
        [
            Prompt.user("typed by a person"),
            Prompt.user("<hc>t135 probe exited (exit 0)</hc>"),
        ]
    )

    lines = capsys.readouterr().out.splitlines()
    headings = [line for line in lines if line.startswith("▎▶")]
    assert len(headings) == 2
    assert "lead" in headings[0] and "hc" not in headings[0]
    assert "hc · t135 exited" in headings[1]
