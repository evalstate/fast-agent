from __future__ import annotations

import json
from pathlib import Path

import pytest
from anthropic.types.beta import BetaUsage
from google.genai.types import GenerateContentResponseUsageMetadata, ServiceTier, UsageMetadata
from openai.types.completion_usage import CompletionUsage
from openai.types.responses.response_usage import ResponseUsage
from pydantic import ValidationError

from fast_agent.llm.provider_types import Provider
from fast_agent.llm.response_telemetry import build_usage_payload
from fast_agent.llm.usage_tracking import (
    CharacterUsage,
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageReport,
    UsageSchema,
    usage_from_anthropic,
    usage_from_google_generate_content,
    usage_from_google_usage_metadata,
    usage_from_openai_chat,
    usage_from_openai_compatible,
    usage_from_openai_responses,
)


def test_usage_report_requires_provider_attempts_and_v2_schema() -> None:
    with pytest.raises(ValidationError, match="provider_attempts"):
        UsageReport(provider_attempts=[])

    with pytest.raises(ValidationError, match="schema"):
        UsageReport.model_validate(
            {"schema": "fast-agent.usage/v1", "provider_attempts": [{}]}
        )


def test_anthropic_translates_disjoint_prompt_partitions_and_thinking_subset() -> None:
    usage = BetaUsage.model_validate(
        {
            "input_tokens": 114,
            "cache_read_input_tokens": 5_251,
            "cache_creation_input_tokens": 0,
            "output_tokens": 56,
            "output_tokens_details": {"thinking_tokens": 12},
        }
    )

    turn = usage_from_anthropic(
        usage,
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-5",
    )

    assert turn.prompt == PromptTokenUsage(
        total=5_365,
        uncached=114,
        cache_read=5_251,
        cache_write=0,
    )
    assert turn.completion == CompletionTokenUsage(total=56, reasoning=12)
    assert turn.total == 5_421
    assert turn.raw_usage == usage.model_dump(mode="json")


def test_anthropic_preserves_optional_cache_and_thinking_details_as_unknown() -> None:
    usage = BetaUsage.model_validate({"input_tokens": 10, "output_tokens": 3})

    turn = usage_from_anthropic(
        usage,
        provider=Provider.ANTHROPIC_VERTEX,
        model="claude",
    )

    assert turn.provider is Provider.ANTHROPIC_VERTEX
    assert turn.prompt.total is None
    assert turn.prompt.cache_read is None
    assert turn.prompt.cache_write is None
    assert turn.completion.total == 3
    assert turn.completion.reasoning is None
    assert turn.total is None


def test_openai_chat_cache_and_reasoning_are_subsets() -> None:
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 3_000,
            "completion_tokens": 300,
            "total_tokens": 3_300,
            "prompt_tokens_details": {
                "cached_tokens": 2_048,
                "cache_write_tokens": 512,
            },
            "completion_tokens_details": {"reasoning_tokens": 120},
        }
    )

    turn = usage_from_openai_chat(
        usage,
        provider=Provider.HUGGINGFACE,
        upstream_provider="fireworks-ai",
        model="kimi",
    )

    assert turn.provider is Provider.HUGGINGFACE
    assert turn.upstream_provider == "fireworks-ai"
    assert turn.usage_schema is UsageSchema.OPENAI_CHAT
    assert turn.prompt == PromptTokenUsage(
        total=3_000,
        cache_read=2_048,
        cache_write=512,
    )
    assert turn.completion == CompletionTokenUsage(total=300, reasoning=120)
    assert turn.total == 3_300


def test_openai_chat_preserves_absent_details_and_explicit_zero() -> None:
    absent = usage_from_openai_chat(
        CompletionUsage.model_validate(
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        ),
        provider=Provider.OPENAI,
        model="gpt",
    )
    zero = usage_from_openai_chat(
        CompletionUsage.model_validate(
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
        ),
        provider=Provider.OPENAI,
        model="gpt",
    )

    assert absent.prompt.cache_read is None
    assert absent.prompt.cache_write is None
    assert absent.completion.reasoning is None
    assert zero.prompt.cache_read == 0
    assert zero.prompt.cache_write == 0
    assert zero.completion.reasoning == 0


def test_openai_responses_uses_its_required_typed_details() -> None:
    usage = ResponseUsage.model_validate(
        {
            "input_tokens": 3_000,
            "input_tokens_details": {
                "cached_tokens": 2_048,
                "cache_write_tokens": 512,
            },
            "output_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 120},
            "total_tokens": 3_300,
        }
    )

    turn = usage_from_openai_responses(
        usage,
        provider=Provider.RESPONSES,
        model="gpt-5.6",
    )

    assert turn.usage_schema is UsageSchema.OPENAI_RESPONSES
    assert turn.prompt == PromptTokenUsage(
        total=3_000,
        cache_read=2_048,
        cache_write=512,
    )
    assert turn.completion == CompletionTokenUsage(total=300, reasoning=120)


def test_openai_responses_preserves_overlapping_cache_subsets() -> None:
    usage = ResponseUsage.model_validate(
        {
            "input_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 80,
                "cache_write_tokens": 70,
            },
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 105,
        }
    )

    turn = usage_from_openai_responses(
        usage,
        provider=Provider.RESPONSES,
        model="gpt",
    )

    assert turn.prompt.uncached is None
    assert turn.prompt.cache_read == 80
    assert turn.prompt.cache_write == 70


def test_provider_total_mismatch_is_a_translation_error() -> None:
    usage = CompletionUsage.model_validate(
        {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 99}
    )

    with pytest.raises(ValueError, match="reported total"):
        usage_from_openai_chat(usage, provider=Provider.OPENAI, model="gpt")


def test_google_includes_reported_tool_use_in_prompt_total() -> None:
    usage = GenerateContentResponseUsageMetadata.model_validate(
        {
            "prompt_token_count": 3_425,
            "candidates_token_count": 23,
            "thoughts_token_count": 47,
            "cached_content_token_count": 1_024,
            "tool_use_prompt_token_count": 17,
            "total_token_count": 3_512,
        }
    )

    turn = usage_from_google_generate_content(usage, model="gemini")

    assert turn.prompt == PromptTokenUsage(
        total=3_442,
        cache_read=1_024,
        tool_use=17,
    )
    assert turn.completion == CompletionTokenUsage(total=70, reasoning=47)


def test_google_uses_total_minus_prompt_only_as_complete_fallback() -> None:
    fallback = usage_from_google_generate_content(
        GenerateContentResponseUsageMetadata.model_validate(
            {"prompt_token_count": 100, "total_token_count": 130}
        ),
        model="gemini",
    )
    partial = usage_from_google_generate_content(
        GenerateContentResponseUsageMetadata.model_validate(
            {"prompt_token_count": 100, "candidates_token_count": 20}
        ),
        model="gemini",
    )

    assert fallback.completion.total == 30
    assert partial.completion.total is None


def test_google_interactions_usage_preserves_service_tier() -> None:
    usage = UsageMetadata.model_validate(
        {
            "prompt_token_count": 10,
            "response_token_count": 5,
            "thoughts_token_count": 0,
            "total_token_count": 15,
            "service_tier": ServiceTier.PRIORITY,
        }
    )

    turn = usage_from_google_usage_metadata(usage, model="gemini")

    assert turn.service_tier == "priority"
    assert turn.completion == CompletionTokenUsage(total=5, reasoning=0)


def test_dynamic_openai_compatible_decoder_preserves_attribution_and_unknowns() -> None:
    turn = usage_from_openai_compatible(
        {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        provider=Provider.HUGGINGFACE,
        upstream_provider="fireworks-ai",
        model="kimi",
    )

    assert turn.provider is Provider.HUGGINGFACE
    assert turn.upstream_provider == "fireworks-ai"
    assert turn.usage_schema is UsageSchema.OPENAI_CHAT
    assert turn.prompt.cache_read is None
    assert turn.completion.reasoning is None


def test_canonical_subset_invariants_are_validated() -> None:
    with pytest.raises(ValidationError, match="cache_read"):
        PromptTokenUsage(total=10, cache_read=11)
    with pytest.raises(ValidationError, match="reasoning"):
        CompletionTokenUsage(total=10, reasoning=11)


def test_aggregation_uses_complete_data_semantics_per_metric() -> None:
    accumulator = UsageAccumulator(
        turns=[
            TurnUsage(
                provider=Provider.OPENAI,
                usage_schema=UsageSchema.OPENAI_CHAT,
                model="gpt",
                prompt=PromptTokenUsage(total=10, cache_read=0),
                completion=CompletionTokenUsage(total=2, reasoning=0),
            ),
            TurnUsage(
                provider=Provider.OPENAI,
                usage_schema=UsageSchema.OPENAI_CHAT,
                model="gpt",
                prompt=PromptTokenUsage(total=20, cache_read=None),
                completion=CompletionTokenUsage(total=3, reasoning=1),
            ),
        ]
    )

    assert accumulator.summary.prompt.total == 30
    assert accumulator.summary.prompt.cache_read is None
    assert accumulator.summary.completion.total == 5
    assert accumulator.summary.completion.reasoning == 1
    assert accumulator.summary.total == 35


def test_versioned_usage_payload_has_no_legacy_fields() -> None:
    turn = TurnUsage(
        provider=Provider.OPENAI,
        usage_schema=UsageSchema.OPENAI_CHAT,
        model="gpt",
        prompt=PromptTokenUsage(total=10, cache_read=0),
        completion=CompletionTokenUsage(total=2, reasoning=0),
        raw_usage={"prompt_tokens": 10},
    )
    payload = build_usage_payload(UsageAccumulator(turns=[turn]))

    assert payload == {
        "schema": "fast-agent.usage/v2",
        "provider_attempts": [
            {
                "provider": "openai",
                "upstream_provider": None,
                "usage_schema": "openai-chat",
                "model": "gpt",
                "prompt": {
                    "total": 10,
                    "uncached": None,
                    "cache_read": 0,
                    "cache_write": None,
                    "tool_use": None,
                },
                "completion": {"total": 2, "reasoning": 0},
                "tool_calls": 0,
                "reasoning_effort": None,
                "requested_service_tier": None,
                "service_tier": None,
                "cost_usd": None,
                "timestamp": turn.timestamp,
                "raw_usage": {"prompt_tokens": 10},
                "total": 12,
            },
        ],
    }
    assert payload is not None
    turn_payload = payload["provider_attempts"][0]
    assert isinstance(turn_payload, dict)
    assert not {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_usage",
        "display_input_tokens",
        "effective_input_tokens",
    } & turn_payload.keys()


def test_usage_payload_aggregates_provider_retries_for_one_outward_turn() -> None:
    prior = TurnUsage(
        provider=Provider.RESPONSES,
        usage_schema=UsageSchema.OPENAI_RESPONSES,
        model="gpt-5.4",
        prompt=PromptTokenUsage(total=10, cache_read=2),
        completion=CompletionTokenUsage(total=1, reasoning=0),
        raw_usage={"attempt": 0},
    )
    first_attempt = TurnUsage(
        provider=Provider.RESPONSES,
        usage_schema=UsageSchema.OPENAI_RESPONSES,
        model="gpt-5.4",
        prompt=PromptTokenUsage(total=20, cache_read=3),
        completion=CompletionTokenUsage(total=0, reasoning=0),
        raw_usage={"attempt": 1},
    )
    retry = TurnUsage(
        provider=Provider.RESPONSES,
        usage_schema=UsageSchema.OPENAI_RESPONSES,
        model="gpt-5.4",
        prompt=PromptTokenUsage(total=25, cache_read=4),
        completion=CompletionTokenUsage(total=5, reasoning=1),
        raw_usage={"attempt": 2},
    )

    payload = build_usage_payload(
        UsageAccumulator(turns=[prior, first_attempt, retry]),
        start_index=1,
    )

    assert payload is not None
    attempts = payload["provider_attempts"]
    assert [attempt["prompt"]["total"] for attempt in attempts] == [20, 25]
    assert [attempt["completion"]["total"] for attempt in attempts] == [0, 5]
    assert [attempt["total"] for attempt in attempts] == [20, 30]
    assert [attempt["raw_usage"] for attempt in attempts] == [
        {"attempt": 1},
        {"attempt": 2},
    ]


def test_synthetic_usage_is_character_telemetry_not_turn_usage() -> None:
    usage = CharacterUsage(
        input_characters=10,
        output_characters=4,
        model_type="passthrough",
    )

    assert usage.input_characters == 10
    with pytest.raises(ValidationError):
        TurnUsage.model_validate(usage.model_dump())


def test_sanitized_live_usage_replay_matches_provider_contracts() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "fixtures"
        / "llm_traces"
        / "sanitized"
        / "token_usage_live_20260715.json"
    )
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))

    anthropic = usage_from_anthropic(
        BetaUsage.model_validate(fixtures["anthropic"]),
        provider=Provider.ANTHROPIC,
        model="claude-sonnet-5",
    )
    google = usage_from_google_generate_content(
        GenerateContentResponseUsageMetadata.model_validate(fixtures["google"]),
        model="gemini-3.5-flash",
    )
    responses = usage_from_openai_responses(
        ResponseUsage.model_validate(fixtures["openai_responses"]),
        provider=Provider.CODEX_RESPONSES,
        model="gpt-5.6-terra",
    )
    compatible = usage_from_openai_chat(
        CompletionUsage.model_validate(fixtures["openai_chat_compatible"]),
        provider=Provider.HUGGINGFACE,
        model="moonshotai/Kimi-K2.7-Code",
    )

    assert anthropic.prompt.total == 13_223
    assert anthropic.completion.total == 83
    assert google.completion == CompletionTokenUsage(total=81, reasoning=59)
    assert responses.completion == CompletionTokenUsage(total=84, reasoning=37)
    assert compatible.provider is Provider.HUGGINGFACE
    assert compatible.prompt.cache_read == 15
