from __future__ import annotations

import json

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.constants import FAST_AGENT_USAGE
from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.openresponses import OpenResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageSchema,
)
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_serialization import save_messages
from fast_agent.mcp.prompts.prompt_load import (
    load_prompt,
    load_transcript_into_agent,
    rehydrate_usage_from_history,
)


def _usage_payload(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "responses",
    raw_usage: object | None = None,
) -> dict[str, object]:
    return {
        "schema": "fast-agent.usage/v2",
        "provider_attempts": [
            {
                "provider": provider,
                "usage_schema": "openai-responses",
                "model": model,
                "prompt": {"total": input_tokens},
                "completion": {"total": output_tokens},
                "tool_calls": 1,
                "raw_usage": raw_usage,
            }
        ],
    }


def _legacy_usage_payload(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "responses",
) -> dict[str, object]:
    return {
        "turn": {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "timestamp": 1_700_000_000.0,
            "cache_usage": {
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cache_hit_tokens": 7,
            },
            "tool_use_tokens": 0,
            "reasoning_tokens": 5,
            "tool_calls": 1,
        },
        "raw_usage": {"legacy": True},
        "summary": {"model": model},
    }


def _history_with_usage(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "responses",
    raw_usage: object | None = None,
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
                        raw_usage=raw_usage,
                    )
                ),
            )
        ]
    }
    return [Prompt.user("hello"), assistant]


def _seed_turn(model: str = "gpt-5.3-codex") -> TurnUsage:
    return TurnUsage(
        provider=Provider.RESPONSES,
        usage_schema=UsageSchema.OPENAI_RESPONSES,
        model=model,
        prompt=PromptTokenUsage(total=10),
        completion=CompletionTokenUsage(total=5),
    )


def _load_history_for_rehydration_test(agent: LlmAgent, history_path) -> str | None:
    messages = load_prompt(history_path)
    usage_accumulator = agent.usage_accumulator
    if usage_accumulator is not None:
        usage_accumulator.reset()
    load_transcript_into_agent(agent, messages)
    return rehydrate_usage_from_history(agent, messages)


@pytest.mark.unit
def test_load_transcript_into_agent_restores_messages_without_rehydrating_usage(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(model="gpt-5.3-codex", input_tokens=120, output_tokens=30),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("transcript-only"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    llm.usage_accumulator.add_turn(_seed_turn())
    agent._llm = llm

    summary_before = llm.usage_accumulator.get_summary()

    load_transcript_into_agent(agent, history_path)

    assert [message.role for message in agent.message_history] == ["user", "assistant"]
    assert agent.message_history[0].first_text() == "hello"
    assert agent.message_history[1].first_text() == "done"
    assert llm.usage_accumulator.get_summary() == summary_before


@pytest.mark.unit
def test_rehydrate_usage_from_history_does_not_load_transcript(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(model="gpt-5.3-codex", input_tokens=120, output_tokens=30),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("usage-only"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    agent._llm = llm
    agent.message_history.append(Prompt.user("keep existing transcript"))

    notice = rehydrate_usage_from_history(agent, history_path)

    assert notice is None
    assert len(agent.message_history) == 1
    assert agent.message_history[0].first_text() == "keep existing transcript"
    assert len(llm.usage_accumulator.turns) == 1
    assert llm.usage_accumulator.summary.total == 150


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

    notice = _load_history_for_rehydration_test(agent, history_path)

    assert notice is None
    assert [message.role for message in agent.message_history] == ["user", "assistant"]
    assert agent.message_history[0].first_text() == "hello"
    assert agent.message_history[1].first_text() == "done"
    assert len(llm.usage_accumulator.turns) == 1
    turn = llm.usage_accumulator.turns[0]
    assert turn.provider == Provider.RESPONSES
    assert turn.model == "gpt-5.3-codex"
    assert turn.prompt.total == 120
    assert turn.completion.total == 30
    assert llm.usage_accumulator.model == "gpt-5.3-codex"


@pytest.mark.unit
def test_rehydrated_retry_usage_keeps_final_request_context() -> None:
    assistant = Prompt.assistant("done")
    payload = _usage_payload(
        model="gpt-5.3-codex",
        input_tokens=25,
        output_tokens=5,
    )
    attempts = payload["provider_attempts"]
    assert isinstance(attempts, list)
    payload["provider_attempts"] = [
        {
            "provider": "responses",
            "usage_schema": "openai-responses",
            "model": "gpt-5.3-codex",
            "prompt": {"total": 20},
            "completion": {"total": 0},
            "tool_calls": 0,
            "raw_usage": None,
        },
        *attempts,
    ]
    assistant.channels = {
        FAST_AGENT_USAGE: [TextContent(type="text", text=json.dumps(payload))]
    }

    agent = LlmAgent(AgentConfig("rehydrate-retry"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    agent._llm = llm

    notice = rehydrate_usage_from_history(agent, [Prompt.user("hello"), assistant])

    assert notice is None
    assert llm.usage_accumulator.current_context_tokens == 30
    assert llm.usage_accumulator.summary.total == 50
    assert llm.usage_accumulator.turns[-1].prompt.total == 25


@pytest.mark.unit
def test_load_history_maps_legacy_responses_usage_only_during_rehydration() -> None:
    assistant = Prompt.assistant("done")
    legacy_payload = _legacy_usage_payload(
        model="gpt-5.3-codex",
        input_tokens=120,
        output_tokens=30,
    )
    assistant.channels = {
        FAST_AGENT_USAGE: [TextContent(type="text", text=json.dumps(legacy_payload))]
    }
    messages = [Prompt.user("hello"), assistant]

    agent = LlmAgent(AgentConfig("rehydrate-legacy-responses"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    agent._llm = llm

    notice = rehydrate_usage_from_history(agent, messages)

    assert notice is None
    assert len(llm.usage_accumulator.turns) == 1
    turn = llm.usage_accumulator.turns[0]
    assert turn.prompt.total == 120
    assert turn.prompt.cache_read == 7
    assert turn.completion.total == 30
    assert turn.completion.reasoning == 5
    usage_block = (messages[1].channels or {})[FAST_AGENT_USAGE][0]
    assert isinstance(usage_block, TextContent)
    assert json.loads(usage_block.text) == legacy_payload


@pytest.mark.unit
def test_load_history_does_not_migrate_non_responses_legacy_usage(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    assistant = Prompt.assistant("done")
    legacy_payload = _legacy_usage_payload(
        model="gpt-5.3-codex",
        input_tokens=120,
        output_tokens=30,
        provider="anthropic",
    )
    assistant.channels = {
        FAST_AGENT_USAGE: [
            TextContent(type="text", text=json.dumps(legacy_payload)),
        ]
    }
    save_messages([Prompt.user("hello"), assistant], str(history_path))

    messages = load_prompt(history_path)

    usage_block = (messages[1].channels or {})[FAST_AGENT_USAGE][0]
    assert isinstance(usage_block, TextContent)
    assert json.loads(usage_block.text) == legacy_payload

    agent = LlmAgent(AgentConfig("ignore-legacy-anthropic"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    agent._llm = llm

    notice = rehydrate_usage_from_history(agent, messages)

    assert notice is None
    assert not llm.usage_accumulator.turns


@pytest.mark.unit
def test_load_history_skips_responses_usage_when_model_changes(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(model="gpt-5.2", input_tokens=220, output_tokens=40),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("rehydrate-responses-mismatch"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    llm.usage_accumulator.add_turn(_seed_turn())
    agent._llm = llm

    notice = _load_history_for_rehydration_test(agent, history_path)

    assert notice == "Model changed from gpt-5.2 to gpt-5.3-codex -- usage info not available"
    assert len(llm.usage_accumulator.turns) == 0


@pytest.mark.unit
def test_load_history_rehydrates_when_switching_between_responses_and_codex(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(
            model="gpt-5.2", input_tokens=95, output_tokens=25, provider="responses"
        ),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("rehydrate-codex-switch"))
    llm = CodexResponsesLLM(provider=Provider.CODEX_RESPONSES, model="gpt-5.3-codex")
    agent._llm = llm

    notice = _load_history_for_rehydration_test(agent, history_path)

    assert notice is None
    assert len(llm.usage_accumulator.turns) == 1


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

    notice = _load_history_for_rehydration_test(agent, history_path)

    assert notice is None
    assert len(llm.usage_accumulator.turns) == 1
    turn = llm.usage_accumulator.turns[0]
    assert turn.provider == Provider.OPENRESPONSES
    assert turn.model == "openai/gpt-5"
    assert turn.prompt.total == 80
    assert turn.completion.total == 20


@pytest.mark.unit
def test_load_history_preserves_raw_usage_snapshot_shape(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages(
        _history_with_usage(
            model="gpt-5.3-codex",
            input_tokens=102,
            output_tokens=108,
            raw_usage={
                "input_tokens": 102,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens": 108,
                "output_tokens_details": {"reasoning_tokens": 64},
                "total_tokens": 210,
            },
        ),
        str(history_path),
    )

    agent = LlmAgent(AgentConfig("rehydrate-raw-usage"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    agent._llm = llm

    notice = _load_history_for_rehydration_test(agent, history_path)

    assert notice is None
    assert len(llm.usage_accumulator.turns) == 1
    turn = llm.usage_accumulator.turns[0]
    assert turn.raw_usage == {
        "input_tokens": 102,
        "input_tokens_details": {"cached_tokens": 5},
        "output_tokens": 108,
        "output_tokens_details": {"reasoning_tokens": 64},
        "total_tokens": 210,
    }


@pytest.mark.unit
def test_load_history_clears_stale_usage_when_history_has_no_usage_payload(tmp_path) -> None:
    history_path = tmp_path / "history.json"
    save_messages([Prompt.user("hello"), Prompt.assistant("done")], str(history_path))

    agent = LlmAgent(AgentConfig("rehydrate-no-usage"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    llm.usage_accumulator.add_turn(_seed_turn())
    agent._llm = llm

    notice = _load_history_for_rehydration_test(agent, history_path)

    assert notice is None
    assert len(llm.usage_accumulator.turns) == 0
