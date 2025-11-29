from pathlib import Path

import pytest
import pytest_asyncio

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.constants import REASONING
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types.llm_stop_reason import LlmStopReason

TEST_MODELS = [
    "kimithink",
    "minimax",
]


@pytest_asyncio.fixture
async def reasoning_agent(model_name: str) -> LlmAgent:
    config_path = Path(__file__).parent / "fastagent.config.yaml"
    core = Core(settings=config_path)
    await core.initialize()
    agent = LlmAgent(AgentConfig("test"), core.context)
    await agent.attach_llm(ModelFactory.create_factory(model_name))
    return agent


def _make_stream_tracker():
    state = {"in_think": False, "plain": 0, "reason": 0}

    def on_chunk(chunk: str) -> None:
        if not chunk:
            return
        text = chunk
        idx = 0
        while idx < len(text):
            if state["in_think"]:
                close = text.find("</think>", idx)
                if close == -1:
                    state["reason"] += 1
                    break
                if close > idx:
                    state["reason"] += 1
                state["in_think"] = False
                idx = close + len("</think>")
            else:
                open_idx = text.find("<think>", idx)
                if open_idx == -1:
                    state["plain"] += 1
                    break
                if open_idx > idx:
                    state["plain"] += 1
                state["in_think"] = True
                idx = open_idx + len("<think>")

    return on_chunk, state


async def _run_turn(agent: LlmAgent, prompt: str) -> tuple[dict[str, int], list[str], str | None]:
    listener, state = _make_stream_tracker()
    remove = agent.llm.add_stream_listener(listener)
    try:
        result = await agent.generate(prompt)
    finally:
        remove()

    assert result.stop_reason is LlmStopReason.END_TURN

    channels = result.channels or {}
    reasoning_blocks = channels.get(REASONING) or []
    reasoning_texts = [txt for txt in (get_text(block) for block in reasoning_blocks) if txt]

    return state, reasoning_texts, result.last_text()


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_reasoning_streams(model_name: str, reasoning_agent: LlmAgent):
    agent = reasoning_agent

    turn1_state, turn1_reasoning, turn1_text = await _run_turn(agent, "Good evening")
    turn2_state, turn2_reasoning, turn2_text = await _run_turn(
        agent, "Tell me some facts about the moon"
    )

    # Both reasoning and plain text should stream more than once across the two turns
    total_reason_chunks = turn1_state["reason"] + turn2_state["reason"]
    total_plain_chunks = turn1_state["plain"] + turn2_state["plain"]
    assert total_reason_chunks > 1
    assert total_plain_chunks > 1

    # Reasoning channel should contain content for each turn
    assert turn1_reasoning and "".join(turn1_reasoning).strip()
    assert turn2_reasoning and "".join(turn2_reasoning).strip()

    # Final text should exist for each turn
    assert turn1_text is not None and turn1_text.strip()
    assert turn2_text is not None and turn2_text.strip()
