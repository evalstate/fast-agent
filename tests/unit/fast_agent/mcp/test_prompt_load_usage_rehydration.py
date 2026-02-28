from __future__ import annotations

import json

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.constants import FAST_AGENT_USAGE
from fast_agent.core.prompt import Prompt
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.openresponses import OpenResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import FastAgentUsage, TurnUsage
from fast_agent.mcp.prompt_serialization import save_messages
from fast_agent.mcp.prompts.prompt_load import load_history_into_agent


def _usage_payload(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "responses",
) -> dict[str, object]:
    total_tokens = input_tokens + output_tokens
    return {
        "turn": {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tool_calls": 1,
        },
        "summary": {
            "model": model,
        },
    }


def _history_with_usage(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "responses",
):
    assistant = Prompt.assistant("done")
    assistant.channels = {
        FAST_AGENT_USAGE: [
            TextContent(
                type="text",
                text=json.dumps(
                    _usage_payload(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        provider=provider,
                    )
                ),
            )
        ]
    }
    return [Prompt.user("hello"), assistant]


@pytest.mark.unit
def test_load_history_rehydrates_responses_usage_when_model_matches(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(model="gpt-5.3-codex", input_tokens=120, output_tokens=30),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("rehydrate-responses"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    agent._llm = llm

    notice = load_history_into_agent(agent, history_path)

    assert notice is None
    assert llm.usage_accumulator.turn_count == 1
    turn = llm.usage_accumulator.turns[0]
    assert turn.provider == Provider.RESPONSES
    assert turn.model == "gpt-5.3-codex"
    assert turn.input_tokens == 120
    assert turn.output_tokens == 30
    assert llm.usage_accumulator.model == "gpt-5.3-codex"


@pytest.mark.unit
def test_load_history_skips_responses_usage_when_model_changes(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(model="gpt-5.2", input_tokens=220, output_tokens=40),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("rehydrate-responses-mismatch"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    llm.usage_accumulator.add_turn(
        TurnUsage.from_fast_agent(
            FastAgentUsage(input_chars=10, output_chars=5, model_type="test"),
            model="gpt-5.3-codex",
        )
    )
    agent._llm = llm

    notice = load_history_into_agent(agent, history_path)

    assert notice == "Model changed from gpt-5.2 to gpt-5.3-codex -- usage info not available"
    assert llm.usage_accumulator.turn_count == 0


@pytest.mark.unit
def test_load_history_rehydrates_when_switching_between_responses_and_codex(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(model="gpt-5.2", input_tokens=95, output_tokens=25, provider="responses"),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("rehydrate-codex-switch"))
    llm = CodexResponsesLLM(provider=Provider.CODEX_RESPONSES, model="gpt-5.3-codex")
    agent._llm = llm

    notice = load_history_into_agent(agent, history_path)

    assert notice is None
    assert llm.usage_accumulator.turn_count == 1


@pytest.mark.unit
def test_load_history_rehydrates_openresponses_usage(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(
            model="openai/gpt-5",
            input_tokens=80,
            output_tokens=20,
            provider="openresponses",
        ),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("rehydrate-openresponses"))
    llm = OpenResponsesLLM(provider=Provider.OPENRESPONSES, model="openai/gpt-5")
    agent._llm = llm

    notice = load_history_into_agent(agent, history_path)

    assert notice is None
    assert llm.usage_accumulator.turn_count == 1
    turn = llm.usage_accumulator.turns[0]
    assert turn.provider == Provider.OPENRESPONSES
    assert turn.model == "openai/gpt-5"
    assert turn.input_tokens == 80
    assert turn.output_tokens == 20
