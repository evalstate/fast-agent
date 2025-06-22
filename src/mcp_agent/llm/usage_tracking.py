"""
Usage tracking system for LLM providers with comprehensive cache support.

This module provides unified usage tracking across Anthropic, OpenAI, and Google providers,
including detailed cache metrics and context window management.
"""

import time
from typing import List, Optional, Union

# Proper type imports for each provider
from anthropic.types import Usage as AnthropicUsage
from google.genai.types import GenerateContentResponseUsageMetadata as GoogleUsage
from openai.types.completion_usage import CompletionUsage as OpenAIUsage
from pydantic import BaseModel, Field, computed_field

from mcp_agent.llm.model_database import ModelDatabase
from mcp_agent.llm.provider_types import Provider


# Fast-agent specific usage type for synthetic providers
class FastAgentUsage(BaseModel):
    """Usage data for fast-agent providers (passthrough, playback, slow)"""

    input_chars: int = Field(description="Characters in input messages")
    output_chars: int = Field(description="Characters in output messages")
    model_type: str = Field(description="Type of fast-agent model (passthrough/playbook/slow)")
    tool_calls: int = Field(default=0, description="Number of tool calls made")
    delay_seconds: float = Field(default=0.0, description="Artificial delays added")


# Union type for raw usage data from any provider
ProviderUsage = Union[AnthropicUsage, OpenAIUsage, GoogleUsage, FastAgentUsage]


class ModelContextWindows:
    """Context window sizes and cache configurations for various models"""

    @classmethod
    def get_context_window(cls, model: str) -> Optional[int]:
        return ModelDatabase.get_context_window(model)


class CacheUsage(BaseModel):
    """Cache-specific usage metrics"""

    cache_read_tokens: int = Field(default=0, description="Tokens read from cache")
    cache_write_tokens: int = Field(default=0, description="Tokens written to cache")
    cache_hit_tokens: int = Field(default=0, description="Total tokens served from cache")

    @computed_field
    @property
    def total_cache_tokens(self) -> int:
        """Total cache-related tokens"""
        return self.cache_read_tokens + self.cache_write_tokens + self.cache_hit_tokens

    @computed_field
    @property
    def has_cache_activity(self) -> bool:
        """Whether any cache activity occurred"""
        return self.total_cache_tokens > 0


class TurnUsage(BaseModel):
    """Usage data for a single turn/completion with cache support"""

    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: float = Field(default_factory=time.time)

    # Cache-specific metrics
    cache_usage: CacheUsage = Field(default_factory=CacheUsage)

    # Provider-specific token types
    tool_use_tokens: int = Field(default=0, description="Tokens used for tool calling prompts")
    reasoning_tokens: int = Field(default=0, description="Tokens used for reasoning/thinking")

    # Raw usage data from provider (preserves all original data)
    raw_usage: ProviderUsage

    @computed_field
    @property
    def current_context_tokens(self) -> int:
        """Current context size after this turn (input + output)"""
        return self.input_tokens + self.output_tokens

    @computed_field
    @property
    def effective_input_tokens(self) -> int:
        """Input tokens excluding cache reads (tokens actually processed)"""
        return max(
            0,
            self.input_tokens
            - self.cache_usage.cache_read_tokens
            - self.cache_usage.cache_hit_tokens,
        )

    @classmethod
    def from_anthropic(cls, usage: AnthropicUsage, model: str) -> "TurnUsage":
        # Extract cache tokens with proper null handling
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

        cache_usage = CacheUsage(
            cache_read_tokens=cache_read_tokens,  # Tokens read from cache (90% discount)
            cache_write_tokens=cache_creation_tokens,  # Tokens written to cache (25% surcharge)
        )

        return cls(
            provider=Provider.ANTHROPIC,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            cache_usage=cache_usage,
            raw_usage=usage,  # Store the original Anthropic usage object
        )

    @classmethod
    def from_openai(cls, usage: OpenAIUsage, model: str) -> "TurnUsage":
        # Extract cache tokens with proper null handling
        cached_tokens = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        cache_usage = CacheUsage(
            cache_hit_tokens=cached_tokens  # These are tokens served from cache (50% discount)
        )

        return cls(
            provider=Provider.OPENAI,
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cache_usage=cache_usage,
            raw_usage=usage,  # Store the original OpenAI usage object
        )

    @classmethod
    def from_google(cls, usage: GoogleUsage, model: str) -> "TurnUsage":
        # Extract token counts with proper null handling
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0
        cached_content_tokens = getattr(usage, "cached_content_token_count", 0) or 0

        # Extract additional Google-specific token types
        tool_use_tokens = getattr(usage, "tool_use_prompt_token_count", 0) or 0
        thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        # Google cache tokens are read hits (75% discount on Gemini 2.5)
        cache_usage = CacheUsage(cache_hit_tokens=cached_content_tokens)

        return cls(
            provider=Provider.GOOGLE,
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=candidates_tokens,
            total_tokens=total_tokens,
            cache_usage=cache_usage,
            tool_use_tokens=tool_use_tokens,
            reasoning_tokens=thinking_tokens,
            raw_usage=usage,  # Store the original Google usage object
        )

    @classmethod
    def from_fast_agent(cls, usage: FastAgentUsage, model: str) -> "TurnUsage":
        # For fast-agent providers, we use characters as "tokens"
        # This provides a consistent unit of measurement across all providers
        input_tokens = usage.input_chars
        output_tokens = usage.output_chars
        total_tokens = input_tokens + output_tokens

        # Fast-agent providers don't have cache functionality
        cache_usage = CacheUsage()

        return cls(
            provider=Provider.FAST_AGENT,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_usage=cache_usage,
            raw_usage=usage,  # Store the original FastAgentUsage object
        )


class UsageAccumulator(BaseModel):
    """Accumulates usage data across multiple turns with cache analytics"""

    turns: List[TurnUsage] = Field(default_factory=list)
    model: Optional[str] = None

    def add_turn(self, turn: TurnUsage) -> None:
        """Add a new turn to the accumulator"""
        self.turns.append(turn)
        if self.model is None:
            self.model = turn.model

    @computed_field
    @property
    def cumulative_input_tokens(self) -> int:
        """Total input tokens charged across all turns"""
        return sum(turn.input_tokens for turn in self.turns)

    @computed_field
    @property
    def cumulative_output_tokens(self) -> int:
        """Total output tokens charged across all turns"""
        return sum(turn.output_tokens for turn in self.turns)

    @computed_field
    @property
    def cumulative_billing_tokens(self) -> int:
        """Total tokens charged across all turns"""
        return sum(turn.total_tokens for turn in self.turns)

    @computed_field
    @property
    def cumulative_cache_read_tokens(self) -> int:
        """Total tokens read from cache across all turns"""
        return sum(turn.cache_usage.cache_read_tokens for turn in self.turns)

    @computed_field
    @property
    def cumulative_cache_write_tokens(self) -> int:
        """Total tokens written to cache across all turns"""
        return sum(turn.cache_usage.cache_write_tokens for turn in self.turns)

    @computed_field
    @property
    def cumulative_cache_hit_tokens(self) -> int:
        """Total tokens served from cache across all turns"""
        return sum(turn.cache_usage.cache_hit_tokens for turn in self.turns)

    @computed_field
    @property
    def cumulative_effective_input_tokens(self) -> int:
        """Total input tokens excluding cache reads across all turns"""
        return sum(turn.effective_input_tokens for turn in self.turns)

    @computed_field
    @property
    def cumulative_tool_use_tokens(self) -> int:
        """Total tokens used for tool calling prompts across all turns"""
        return sum(turn.tool_use_tokens for turn in self.turns)

    @computed_field
    @property
    def cumulative_reasoning_tokens(self) -> int:
        """Total tokens used for reasoning/thinking across all turns"""
        return sum(turn.reasoning_tokens for turn in self.turns)

    @computed_field
    @property
    def cache_hit_rate(self) -> Optional[float]:
        """Percentage of input tokens served from cache"""
        if self.cumulative_input_tokens == 0:
            return None
        cache_tokens = self.cumulative_cache_read_tokens + self.cumulative_cache_hit_tokens
        return (cache_tokens / self.cumulative_input_tokens) * 100

    @computed_field
    @property
    def current_context_tokens(self) -> int:
        """Current context usage (last turn's context tokens)"""
        if not self.turns:
            return 0
        return self.turns[-1].current_context_tokens

    @computed_field
    @property
    def context_window_size(self) -> Optional[int]:
        """Get context window size for current model"""
        if self.model:
            return ModelContextWindows.get_context_window(self.model)
        return None

    @computed_field
    @property
    def context_usage_percentage(self) -> Optional[float]:
        """Percentage of context window used"""
        window_size = self.context_window_size
        if window_size and window_size > 0:
            return (self.current_context_tokens / window_size) * 100
        return None

    @computed_field
    @property
    def turn_count(self) -> int:
        """Number of turns accumulated"""
        return len(self.turns)

    def get_cache_summary(self) -> dict[str, Union[int, float, None]]:
        """Get cache-specific metrics summary"""
        return {
            "cumulative_cache_read_tokens": self.cumulative_cache_read_tokens,
            "cumulative_cache_write_tokens": self.cumulative_cache_write_tokens,
            "cumulative_cache_hit_tokens": self.cumulative_cache_hit_tokens,
            "cache_hit_rate_percent": self.cache_hit_rate,
            "cumulative_effective_input_tokens": self.cumulative_effective_input_tokens,
        }

    def get_summary(self) -> dict[str, Union[int, float, str, None]]:
        """Get comprehensive usage statistics"""
        cache_summary = self.get_cache_summary()
        return {
            "model": self.model,
            "turn_count": self.turn_count,
            "cumulative_input_tokens": self.cumulative_input_tokens,
            "cumulative_output_tokens": self.cumulative_output_tokens,
            "cumulative_billing_tokens": self.cumulative_billing_tokens,
            "cumulative_tool_use_tokens": self.cumulative_tool_use_tokens,
            "cumulative_reasoning_tokens": self.cumulative_reasoning_tokens,
            "current_context_tokens": self.current_context_tokens,
            "context_window_size": self.context_window_size,
            "context_usage_percentage": self.context_usage_percentage,
            **cache_summary,
        }


# Utility functions for fast-agent integration
def create_fast_agent_usage(
    input_content: str,
    output_content: str,
    model_type: str,
    tool_calls: int = 0,
    delay_seconds: float = 0.0,
) -> FastAgentUsage:
    """
    Create FastAgentUsage from message content.

    Args:
        input_content: Input message content
        output_content: Output message content
        model_type: Type of fast-agent model (passthrough/playback/slow)
        tool_calls: Number of tool calls made
        delay_seconds: Artificial delays added

    Returns:
        FastAgentUsage object with character counts
    """
    return FastAgentUsage(
        input_chars=len(input_content),
        output_chars=len(output_content),
        model_type=model_type,
        tool_calls=tool_calls,
        delay_seconds=delay_seconds,
    )


def create_turn_usage_from_messages(
    input_content: str,
    output_content: str,
    model: str,
    model_type: str,
    tool_calls: int = 0,
    delay_seconds: float = 0.0,
) -> TurnUsage:
    """
    Create TurnUsage directly from message content for fast-agent providers.

    Args:
        input_content: Input message content
        output_content: Output message content
        model: Model name (e.g., "passthrough", "playback", "slow")
        model_type: Type for internal tracking
        tool_calls: Number of tool calls made
        delay_seconds: Artificial delays added

    Returns:
        TurnUsage object ready for accumulation
    """
    usage = create_fast_agent_usage(
        input_content=input_content,
        output_content=output_content,
        model_type=model_type,
        tool_calls=tool_calls,
        delay_seconds=delay_seconds,
    )
    return TurnUsage.from_fast_agent(usage, model)
