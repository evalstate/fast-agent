"""Canonical provider-neutral token usage and provider translation."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from fast_agent.core.logging.json_serializer import JsonValue, snapshot_json_value
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from anthropic.types.beta import BetaUsage as AnthropicUsage
    from google.genai.types import GenerateContentResponseUsageMetadata, UsageMetadata
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses.response_usage import ResponseUsage


TokenCount = Annotated[int, Field(ge=0)]
CostUsd = Annotated[float, Field(ge=0)]


class CharacterUsage(BaseModel):
    """Operational character telemetry for synthetic fast-agent providers."""

    input_characters: TokenCount
    output_characters: TokenCount
    model_type: str
    tool_calls: TokenCount = 0
    delay_seconds: Annotated[float, Field(ge=0)] = 0.0


class PromptTokenUsage(BaseModel):
    """Complete prompt usage and provider-observed prompt partitions."""

    total: TokenCount | None = None
    uncached: TokenCount | None = None
    cache_read: TokenCount | None = None
    cache_write: TokenCount | None = None
    tool_use: TokenCount | None = None

    @model_validator(mode="after")
    def validate_subsets(self) -> PromptTokenUsage:
        if self.total is None:
            return self
        for name in ("uncached", "cache_read", "cache_write", "tool_use"):
            value = getattr(self, name)
            if value is not None and value > self.total:
                raise ValueError(f"prompt.{name} exceeds prompt.total")
        return self


class CompletionTokenUsage(BaseModel):
    """Complete generated usage and provider-observed output subsets."""

    total: TokenCount | None = None
    reasoning: TokenCount | None = None

    @model_validator(mode="after")
    def validate_subsets(self) -> CompletionTokenUsage:
        if (
            self.total is not None
            and self.reasoning is not None
            and self.reasoning > self.total
        ):
            raise ValueError("completion.reasoning exceeds completion.total")
        return self


class UsageSchema(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI_CHAT = "openai-chat"
    OPENAI_RESPONSES = "openai-responses"
    OPENAI_RESPONSES_COMPATIBLE = "openai-responses-compatible"
    GOOGLE_GENERATE_CONTENT = "google-generate-content"
    BEDROCK = "bedrock"


class TurnUsage(BaseModel):
    """Canonical usage observation for one completed provider inference."""

    provider: Provider
    upstream_provider: str | None = None
    usage_schema: UsageSchema
    model: str
    prompt: PromptTokenUsage
    completion: CompletionTokenUsage
    tool_calls: TokenCount = 0
    reasoning_effort: str | None = None
    requested_service_tier: str | None = None
    service_tier: str | None = None
    cost_usd: CostUsd | None = None
    timestamp: float = Field(default_factory=time.time)
    raw_usage: JsonValue = None

    @field_validator("raw_usage", mode="before")
    @classmethod
    def snapshot_raw_usage(cls, value: object | None) -> JsonValue:
        return snapshot_json_value(value)

    @model_validator(mode="after")
    def validate_provider_contract(self) -> TurnUsage:
        if self.usage_schema is not UsageSchema.ANTHROPIC:
            return self
        values = (
            self.prompt.uncached,
            self.prompt.cache_read,
            self.prompt.cache_write,
        )
        if self.prompt.total is not None and all(value is not None for value in values):
            known_values = [value for value in values if value is not None]
            if self.prompt.total != sum(known_values):
                raise ValueError("Anthropic prompt total does not equal its partitions")
        return self

    @computed_field
    @property
    def total(self) -> int | None:
        if self.prompt.total is None or self.completion.total is None:
            return None
        return self.prompt.total + self.completion.total


class UsageSummary(BaseModel):
    """Complete-data aggregate of canonical usage observations."""

    prompt: PromptTokenUsage
    completion: CompletionTokenUsage
    provider_attempts: TokenCount
    tool_calls: TokenCount

    @computed_field
    @property
    def total(self) -> int | None:
        if self.prompt.total is None or self.completion.total is None:
            return None
        return self.prompt.total + self.completion.total


class UsageReport(BaseModel):
    """Usage for one outward inference, including every provider attempt."""

    schema_version: Literal["fast-agent.usage/v2"] = Field(
        default="fast-agent.usage/v2",
        alias="schema",
    )
    provider_attempts: list[TurnUsage] = Field(min_length=1)

    def to_payload(self) -> dict[str, object]:
        """Serialize the canonical report using its wire-format field names."""

        return self.model_dump(mode="json", by_alias=True)

    @property
    def final_attempt(self) -> TurnUsage:
        """The provider attempt whose token count represents current context."""

        return self.provider_attempts[-1]

    @property
    def consumed(self) -> UsageSummary:
        """Aggregate provider consumption across all attempts."""

        return summarize_usage(self.provider_attempts)


def _complete_sum(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def summarize_usage(attempts: Sequence[TurnUsage]) -> UsageSummary:
    """Aggregate canonical provider attempts using complete-data semantics."""

    return UsageSummary(
        prompt=PromptTokenUsage(
            total=_complete_sum([attempt.prompt.total for attempt in attempts]),
            uncached=_complete_sum([attempt.prompt.uncached for attempt in attempts]),
            cache_read=_complete_sum([attempt.prompt.cache_read for attempt in attempts]),
            cache_write=_complete_sum([attempt.prompt.cache_write for attempt in attempts]),
            tool_use=_complete_sum([attempt.prompt.tool_use for attempt in attempts]),
        ),
        completion=CompletionTokenUsage(
            total=_complete_sum([attempt.completion.total for attempt in attempts]),
            reasoning=_complete_sum([attempt.completion.reasoning for attempt in attempts]),
        ),
        provider_attempts=len(attempts),
        tool_calls=sum(attempt.tool_calls for attempt in attempts),
    )


def _validated_provider_total(
    reported: int,
    prompt_total: int,
    completion_total: int,
    *,
    schema: UsageSchema,
) -> None:
    if reported != prompt_total + completion_total:
        raise ValueError(f"{schema} reported total does not equal prompt plus completion")


def usage_from_anthropic(
    usage: AnthropicUsage,
    *,
    provider: Provider,
    model: str,
) -> TurnUsage:
    """Translate Anthropic's disjoint prompt partitions."""

    cache_read = usage.cache_read_input_tokens
    cache_write = usage.cache_creation_input_tokens
    prompt_total = (
        usage.input_tokens + cache_read + cache_write
        if cache_read is not None and cache_write is not None
        else None
    )
    output_details = usage.output_tokens_details
    return TurnUsage(
        provider=provider,
        usage_schema=UsageSchema.ANTHROPIC,
        model=model,
        prompt=PromptTokenUsage(
            total=prompt_total,
            uncached=usage.input_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
        ),
        completion=CompletionTokenUsage(
            total=usage.output_tokens,
            reasoning=(
                output_details.thinking_tokens if output_details is not None else None
            ),
        ),
        service_tier=usage.service_tier,
        raw_usage=snapshot_json_value(usage),
    )


def usage_from_openai_chat(
    usage: CompletionUsage,
    *,
    provider: Provider,
    model: str,
    upstream_provider: str | None = None,
) -> TurnUsage:
    """Translate OpenAI Chat Completions subset semantics."""

    _validated_provider_total(
        usage.total_tokens,
        usage.prompt_tokens,
        usage.completion_tokens,
        schema=UsageSchema.OPENAI_CHAT,
    )
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return TurnUsage(
        provider=provider,
        upstream_provider=upstream_provider,
        usage_schema=UsageSchema.OPENAI_CHAT,
        model=model,
        prompt=PromptTokenUsage(
            total=usage.prompt_tokens,
            cache_read=prompt_details.cached_tokens if prompt_details is not None else None,
            cache_write=(
                prompt_details.cache_write_tokens if prompt_details is not None else None
            ),
        ),
        completion=CompletionTokenUsage(
            total=usage.completion_tokens,
            reasoning=(
                completion_details.reasoning_tokens
                if completion_details is not None
                else None
            ),
        ),
        raw_usage=snapshot_json_value(usage),
    )


def usage_from_openai_responses(
    usage: ResponseUsage,
    *,
    provider: Provider,
    model: str,
    upstream_provider: str | None = None,
) -> TurnUsage:
    """Translate the distinct OpenAI Responses usage type."""

    _validated_provider_total(
        usage.total_tokens,
        usage.input_tokens,
        usage.output_tokens,
        schema=UsageSchema.OPENAI_RESPONSES,
    )
    return TurnUsage(
        provider=provider,
        upstream_provider=upstream_provider,
        usage_schema=UsageSchema.OPENAI_RESPONSES,
        model=model,
        prompt=PromptTokenUsage(
            total=usage.input_tokens,
            cache_read=usage.input_tokens_details.cached_tokens,
            cache_write=usage.input_tokens_details.cache_write_tokens,
        ),
        completion=CompletionTokenUsage(
            total=usage.output_tokens,
            reasoning=usage.output_tokens_details.reasoning_tokens,
        ),
        raw_usage=snapshot_json_value(usage),
    )


def usage_from_responses_compatible(
    usage: Mapping[str, object],
    *,
    provider: Provider,
    model: str,
) -> TurnUsage:
    """Translate Responses-compatible usage without inventing omitted details."""

    def optional_count(mapping: Mapping[str, object], key: str) -> TokenCount | None:
        value = mapping.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer or null")
        return value

    def details(key: str) -> Mapping[str, object]:
        value = usage.get(key)
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be an object or null")
        return {str(name): item for name, item in value.items()}

    input_details = details("input_tokens_details")
    output_details = details("output_tokens_details")
    input_total = optional_count(usage, "input_tokens")
    output_total = optional_count(usage, "output_tokens")
    reported_total = optional_count(usage, "total_tokens")
    if None not in (reported_total, input_total, output_total):
        assert reported_total is not None
        assert input_total is not None
        assert output_total is not None
        _validated_provider_total(
            reported_total,
            input_total,
            output_total,
            schema=UsageSchema.OPENAI_RESPONSES_COMPATIBLE,
        )
    return TurnUsage(
        provider=provider,
        usage_schema=UsageSchema.OPENAI_RESPONSES_COMPATIBLE,
        model=model,
        prompt=PromptTokenUsage(
            total=input_total,
            cache_read=optional_count(input_details, "cached_tokens"),
            cache_write=optional_count(input_details, "cache_write_tokens"),
        ),
        completion=CompletionTokenUsage(
            total=output_total,
            reasoning=optional_count(output_details, "reasoning_tokens"),
        ),
        raw_usage=snapshot_json_value(usage),
    )


def _google_completion_total(
    *,
    prompt: int | None,
    visible: int | None,
    thoughts: int | None,
    provider_total: int | None,
) -> int | None:
    if visible is not None and thoughts is not None:
        return visible + thoughts
    if prompt is not None and provider_total is not None and provider_total >= prompt:
        return provider_total - prompt
    return None


def _usage_from_google(
    *,
    prompt: int | None,
    visible: int | None,
    thoughts: int | None,
    cached: int | None,
    tool_use: int | None,
    provider_total: int | None,
    service_tier: str | None,
    model: str,
    raw_usage: object,
) -> TurnUsage:
    # Google reports tool-use prompt tokens separately from prompt_token_count,
    # while total_token_count includes them. We have not observed a nonzero
    # value in captured responses yet; the synthetic contract test covers it.
    prompt_total = prompt + tool_use if prompt is not None and tool_use is not None else prompt
    completion_total = _google_completion_total(
        prompt=prompt_total,
        visible=visible,
        thoughts=thoughts,
        provider_total=provider_total,
    )
    if (
        provider_total is not None
        and prompt_total is not None
        and completion_total is not None
        and provider_total != prompt_total + completion_total
    ):
        raise ValueError("Google reported total does not equal prompt plus completion")
    return TurnUsage(
        provider=Provider.GOOGLE,
        usage_schema=UsageSchema.GOOGLE_GENERATE_CONTENT,
        model=model,
        prompt=PromptTokenUsage(
            total=prompt_total,
            cache_read=cached,
            tool_use=tool_use,
        ),
        completion=CompletionTokenUsage(total=completion_total, reasoning=thoughts),
        service_tier=service_tier,
        raw_usage=snapshot_json_value(raw_usage),
    )


def usage_from_google_generate_content(
    usage: GenerateContentResponseUsageMetadata,
    *,
    model: str,
) -> TurnUsage:
    return _usage_from_google(
        prompt=usage.prompt_token_count,
        visible=usage.candidates_token_count,
        thoughts=usage.thoughts_token_count,
        cached=usage.cached_content_token_count,
        tool_use=usage.tool_use_prompt_token_count,
        provider_total=usage.total_token_count,
        service_tier=None,
        model=model,
        raw_usage=usage,
    )


def usage_from_google_usage_metadata(
    usage: UsageMetadata,
    *,
    model: str,
) -> TurnUsage:
    service_tier = usage.service_tier
    return _usage_from_google(
        prompt=usage.prompt_token_count,
        visible=usage.response_token_count,
        thoughts=usage.thoughts_token_count,
        cached=usage.cached_content_token_count,
        tool_use=usage.tool_use_prompt_token_count,
        provider_total=usage.total_token_count,
        service_tier=service_tier.value if service_tier is not None else None,
        model=model,
        raw_usage=usage,
    )


def usage_from_openai_compatible(
    usage: Mapping[str, object],
    *,
    provider: Provider,
    model: str,
    upstream_provider: str | None = None,
) -> TurnUsage:
    """Decode a genuinely dynamic OpenAI Chat-compatible usage payload."""

    def optional_count(mapping: Mapping[str, object], key: str) -> int | None:
        value = mapping.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer or null")
        return value

    def details(key: str) -> Mapping[str, object]:
        value = usage.get(key)
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be an object or null")
        return {str(item_key): item for item_key, item in value.items()}

    prompt_total = optional_count(usage, "prompt_tokens")
    completion_total = optional_count(usage, "completion_tokens")
    reported_total = optional_count(usage, "total_tokens")
    if None not in (reported_total, prompt_total, completion_total):
        assert reported_total is not None
        assert prompt_total is not None
        assert completion_total is not None
        _validated_provider_total(
            reported_total,
            prompt_total,
            completion_total,
            schema=UsageSchema.OPENAI_CHAT,
        )
    prompt_details = details("prompt_tokens_details")
    completion_details = details("completion_tokens_details")
    return TurnUsage(
        provider=provider,
        upstream_provider=upstream_provider,
        usage_schema=UsageSchema.OPENAI_CHAT,
        model=model,
        prompt=PromptTokenUsage(
            total=prompt_total,
            cache_read=optional_count(prompt_details, "cached_tokens"),
            cache_write=optional_count(prompt_details, "cache_write_tokens"),
        ),
        completion=CompletionTokenUsage(
            total=completion_total,
            reasoning=optional_count(completion_details, "reasoning_tokens"),
        ),
        raw_usage=snapshot_json_value(usage),
    )


def usage_from_bedrock(
    usage: Mapping[str, object],
    *,
    model: str,
) -> TurnUsage:
    """Translate fast-agent's normalized Bedrock response usage."""

    input_value = usage.get("input_tokens")
    output_value = usage.get("output_tokens")
    input_tokens = (
        input_value
        if isinstance(input_value, int) and not isinstance(input_value, bool)
        else None
    )
    output_tokens = (
        output_value
        if isinstance(output_value, int) and not isinstance(output_value, bool)
        else None
    )
    return TurnUsage(
        provider=Provider.BEDROCK,
        usage_schema=UsageSchema.BEDROCK,
        model=model,
        prompt=PromptTokenUsage(total=input_tokens),
        completion=CompletionTokenUsage(total=output_tokens),
        raw_usage=snapshot_json_value(usage),
    )


class UsageAccumulator(BaseModel):
    """Accumulate canonical turns and operational context state."""

    turns: list[TurnUsage] = Field(default_factory=list)
    model: str | None = None
    last_cache_activity_time: float | None = None
    _context_window_size: int | None = PrivateAttr(default=None)
    _context_estimate: int | None = PrivateAttr(default=None)

    def set_context_window_size(self, value: int | None) -> None:
        self._context_window_size = value

    def set_context_estimate(self, value: int | None) -> None:
        self._context_estimate = value

    def reset(self) -> None:
        self.turns = []
        self.model = None
        self.last_cache_activity_time = None
        self._context_estimate = None

    def add_turn(self, turn: TurnUsage) -> None:
        self._context_estimate = None
        self.turns.append(turn)
        if self.model is None:
            self.model = turn.model
        if (turn.prompt.cache_read or 0) > 0 or (turn.prompt.cache_write or 0) > 0:
            self.last_cache_activity_time = turn.timestamp

    def count_tools(self, tool_calls: int) -> None:
        if self.turns:
            self.turns[-1].tool_calls = tool_calls

    @property
    def summary(self) -> UsageSummary:
        return summarize_usage(self.turns)

    @computed_field
    @property
    def current_context_tokens(self) -> int | None:
        if self._context_estimate is not None:
            return self._context_estimate
        if not self.turns:
            return None
        return self.turns[-1].total

    @computed_field
    @property
    def context_window_size(self) -> int | None:
        if self._context_window_size is not None:
            return self._context_window_size
        return ModelDatabase.get_context_window(self.model) if self.model else None

    @computed_field
    @property
    def context_usage_percentage(self) -> float | None:
        current = self.current_context_tokens
        window = self.context_window_size
        if current is None or window is None or window <= 0:
            return None
        return (current / window) * 100

    def get_summary(self) -> dict[str, object]:
        """Return the canonical summary plus operational context metadata."""

        return {
            **self.summary.model_dump(mode="json"),
            "model": self.model,
            "current_context_tokens": self.current_context_tokens,
            "context_window_size": self.context_window_size,
            "context_usage_percentage": self.context_usage_percentage,
        }


def create_character_usage(
    input_content: str,
    output_content: str,
    model_type: str,
    tool_calls: int = 0,
    delay_seconds: float = 0.0,
) -> CharacterUsage:
    return CharacterUsage(
        input_characters=len(input_content),
        output_characters=len(output_content),
        model_type=model_type,
        tool_calls=tool_calls,
        delay_seconds=delay_seconds,
    )


def last_turn_usage(
    usage_accumulator: UsageAccumulator | None,
    start_index: int | None,
) -> dict[str, int] | None:
    if usage_accumulator is None or not usage_accumulator.turns:
        return None
    turns = usage_accumulator.turns
    if start_index is not None and start_index >= len(turns):
        return None
    selected = turns[start_index:] if start_index is not None else [turns[-1]]
    prompt = _complete_sum([turn.prompt.total for turn in selected])
    completion = _complete_sum([turn.completion.total for turn in selected])
    if prompt is None or completion is None:
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "tool_calls": sum(turn.tool_calls for turn in selected),
    }
