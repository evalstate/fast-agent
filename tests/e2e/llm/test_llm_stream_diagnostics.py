from __future__ import annotations

from pathlib import Path
from typing import Any

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
    "hf.moonshotai/Kimi-K2-Thinking:nebius",
    "hf.moonshotai/Kimi-K2-Thinking:novita",
]


@pytest_asyncio.fixture
async def diagnostic_agent(model_name: str) -> LlmAgent:
    config_path = Path(__file__).parent / "fastagent.config.yaml"
    core = Core(settings=config_path)
    await core.initialize()
    agent = LlmAgent(AgentConfig("test"), core.context)
    await agent.attach_llm(ModelFactory.create_factory(model_name))
    return agent


def _patch_stream_logging(llm: Any):
    summaries: list[list[dict[str, Any]]] = []

    def _wrap(original):
        async def wrapped(stream, model, *args):
            local: list[dict[str, Any]] = []

            async def logged_stream():
                async for chunk in stream:
                    info: dict[str, Any] = {}
                    if getattr(chunk, "choices", None):
                        choice = chunk.choices[0]
                        delta = getattr(choice, "delta", None)
                        if delta:
                            content = getattr(delta, "content", None)
                            reasoning_content = getattr(delta, "reasoning_content", None)
                            tool_calls = getattr(delta, "tool_calls", None)
                            info["content_len"] = len(content) if content else 0
                            info["reasoning_count"] = (
                                len(reasoning_content) if reasoning_content else 0
                            )
                            info["tool_calls"] = len(tool_calls) if tool_calls else 0
                            info["finish_reason"] = choice.finish_reason
                    if getattr(chunk, "usage", None):
                        info["usage"] = True
                    local.append(info)
                    yield chunk

            result = await original(logged_stream(), model, *args)
            summaries.append(local)
            return result

        return wrapped

    # Patch both streaming paths so we capture whatever the provider uses
    llm._process_stream = _wrap(llm._process_stream)
    llm._process_stream_manual = _wrap(llm._process_stream_manual)

    def consume_summary() -> list[dict[str, Any]]:
        return summaries.pop(0) if summaries else []

    return consume_summary


async def _run_turn(
    agent: LlmAgent, prompt: str, consume_summary
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    result = await agent.generate(prompt)
    summary = consume_summary()

    assert result.stop_reason is LlmStopReason.END_TURN

    channels = result.channels or {}
    reasoning_blocks = channels.get(REASONING) or []
    reasoning_texts = [txt for txt in (get_text(block) for block in reasoning_blocks) if txt]

    return summary, reasoning_texts, result.last_text()


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", TEST_MODELS)
async def test_stream_diagnostics(model_name: str, diagnostic_agent: LlmAgent):
    agent = diagnostic_agent
    consume_summary = _patch_stream_logging(agent.llm)

    turn1_summary, turn1_reasoning, turn1_text = await _run_turn(
        agent, "Hello there", consume_summary
    )
    turn2_summary, turn2_reasoning, turn2_text = await _run_turn(
        agent, "Please share two quick facts about the moon", consume_summary
    )

    def count_chunks(summary: list[dict[str, Any]]) -> tuple[int, int, int]:
        reasoning = sum(1 for item in summary if item.get("reasoning_count", 0) > 0)
        content = sum(1 for item in summary if item.get("content_len", 0) > 0)
        tools = sum(1 for item in summary if item.get("tool_calls", 0) > 0)
        return reasoning, content, tools

    r1, c1, t1 = count_chunks(turn1_summary)
    r2, c2, t2 = count_chunks(turn2_summary)

    print("turn1", turn1_summary)
    print("turn2", turn2_summary)
    # Expect to see both reasoning and content chunks across the two turns
    assert (r1 + r2) > 0
    assert (c1 + c2) > 0

    # Reasoning channel should contain content for each turn
    assert turn1_reasoning and "".join(turn1_reasoning).strip()
    assert turn2_reasoning and "".join(turn2_reasoning).strip()

    # Final text should exist for each turn
    assert turn1_text is not None and turn1_text.strip()
    assert turn2_text is not None and turn2_text.strip()

    # Tool streaming might differ per provider; ensure we at least captured the counts for visibility
    _ = t1 + t2
