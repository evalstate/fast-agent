from anthropic.types.beta import BetaUsage as AnthropicUsage
from google.genai.types import GenerateContentResponseUsageMetadata as GoogleUsage
from openai.types.completion_usage import (
    CompletionTokensDetails,
    PromptTokensDetails,
)
from openai.types.completion_usage import (
    CompletionUsage as OpenAIUsage,
)

from fast_agent.core.logging.json_serializer import snapshot_json_value
from fast_agent.llm.provider.openai.responses_websocket import _AttrObjectView
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.response_telemetry import build_usage_payload
from fast_agent.llm.usage_tracking import (
    CacheUsage,
    TurnUsage,
    UsageAccumulator,
    create_turn_usage_from_messages,
)


def test_anthropic_usage_calculation():
    """Test Anthropic usage calculations with cache tokens"""
    # Create real Anthropic usage object with cache data
    anthropic_usage = AnthropicUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=300,
    )

    turn = TurnUsage.from_anthropic(anthropic_usage, "claude-sonnet-4-0")

    # Basic token counts
    assert turn.input_tokens == 1000
    assert turn.output_tokens == 500
    assert turn.total_tokens == 1500
    # Current context includes cache tokens: input(1000) + cache_read(300) + cache_write(200) + output(500) = 2000
    assert turn.current_context_tokens == 2000

    # Cache calculations
    assert turn.cache_usage.cache_write_tokens == 200  # creation
    assert turn.cache_usage.cache_read_tokens == 300  # read
    assert turn.cache_usage.cache_hit_tokens == 0  # not used for anthropic

    # Effective tokens: For Anthropic, input_tokens already excludes cache
    assert turn.effective_input_tokens == 1000  # input_tokens (cache already excluded by Anthropic)

    # Provider and raw data
    assert turn.provider == Provider.ANTHROPIC
    assert isinstance(turn.raw_usage, dict)
    assert turn.raw_usage["input_tokens"] == 1000
    assert turn.raw_usage["output_tokens"] == 500
    assert turn.raw_usage["cache_creation_input_tokens"] == 200
    assert turn.raw_usage["cache_read_input_tokens"] == 300


def test_openai_usage_calculation():
    """Test OpenAI usage calculations with cache tokens"""
    # Create real OpenAI usage object with cache data
    prompt_details = PromptTokensDetails(cached_tokens=400)
    completion_details = CompletionTokensDetails(reasoning_tokens=100)

    openai_usage = OpenAIUsage(
        prompt_tokens=1200,
        completion_tokens=600,
        total_tokens=1800,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
    )

    turn = TurnUsage.from_openai(openai_usage, "gpt-4o")

    # Basic token counts
    assert turn.input_tokens == 1200
    assert turn.output_tokens == 600
    assert turn.total_tokens == 1800
    assert turn.current_context_tokens == 1800

    # Cache calculations
    assert turn.cache_usage.cache_write_tokens == 0  # not used for openai
    assert turn.cache_usage.cache_read_tokens == 0  # not used for openai
    assert turn.cache_usage.cache_hit_tokens == 400  # cached tokens

    # Effective tokens (input minus cache hits)
    assert turn.effective_input_tokens == 800  # 1200 - 400

    # Provider and raw data
    assert turn.provider == Provider.OPENAI
    assert isinstance(turn.raw_usage, dict)
    assert turn.raw_usage["prompt_tokens"] == 1200
    assert turn.raw_usage["completion_tokens"] == 600
    assert turn.raw_usage["total_tokens"] == 1800
    prompt_details_snapshot = turn.raw_usage["prompt_tokens_details"]
    assert isinstance(prompt_details_snapshot, dict)
    assert prompt_details_snapshot["cached_tokens"] == 400
    completion_details_snapshot = turn.raw_usage["completion_tokens_details"]
    assert isinstance(completion_details_snapshot, dict)
    assert completion_details_snapshot["reasoning_tokens"] == 100


def test_google_usage_calculation():
    """Test Google usage calculations with cache tokens"""
    # Create real Google usage object with cache data
    google_usage = GoogleUsage(
        prompt_token_count=1500,
        candidates_token_count=750,
        total_token_count=2250,
        cached_content_token_count=500,
    )

    turn = TurnUsage.from_google(google_usage, "gemini-2.0-flash")

    # Basic token counts
    assert turn.input_tokens == 1500
    assert turn.output_tokens == 750
    assert turn.total_tokens == 2250
    assert turn.current_context_tokens == 2250

    # Cache calculations
    assert turn.cache_usage.cache_write_tokens == 0  # not used for google
    assert turn.cache_usage.cache_read_tokens == 0  # not used for google
    assert turn.cache_usage.cache_hit_tokens == 500  # cached content

    # Effective tokens (input minus cache hits)
    assert turn.effective_input_tokens == 1000  # 1500 - 500

    # Provider and raw data
    assert turn.provider == Provider.GOOGLE
    assert isinstance(turn.raw_usage, dict)
    assert turn.raw_usage["prompt_token_count"] == 1500
    assert turn.raw_usage["candidates_token_count"] == 750
    assert turn.raw_usage["total_token_count"] == 2250
    assert turn.raw_usage["cached_content_token_count"] == 500


def test_fast_agent_usage_calculation():
    """Test fast-agent usage calculations based on character counts"""
    input_content = "This is a test input message with some content"
    output_content = "This is the response from the fast-agent"

    turn = create_turn_usage_from_messages(
        input_content=input_content,
        output_content=output_content,
        model="passthrough",
        model_type="passthrough",
        tool_calls=2,
        delay_seconds=0.0,
    )

    # Character counts as "tokens"
    assert turn.input_tokens == len(input_content)  # 45 chars
    assert turn.output_tokens == len(output_content)  # 39 chars
    assert turn.total_tokens == len(input_content) + len(output_content)  # 84 chars

    # No cache for fast-agent
    assert turn.cache_usage.cache_write_tokens == 0
    assert turn.cache_usage.cache_read_tokens == 0
    assert turn.cache_usage.cache_hit_tokens == 0

    # Effective tokens equals input tokens (no cache)
    assert turn.effective_input_tokens == len(input_content)

    # Provider and raw data
    assert turn.provider == Provider.FAST_AGENT
    assert turn.raw_usage == {
        "input_chars": len(input_content),
        "output_chars": len(output_content),
        "model_type": "passthrough",
        "tool_calls": 2,
        "delay_seconds": 0.0,
    }


def test_responses_websocket_raw_usage_snapshot_is_preserved_without_coercion():
    """Websocket Responses usage should persist as a JSON-safe snapshot."""
    websocket_usage = _AttrObjectView(
        {
            "input_tokens": 102,
            "input_tokens_details": {"cached_tokens": 5},
            "output_tokens": 108,
            "output_tokens_details": {"reasoning_tokens": 64},
            "total_tokens": 210,
        }
    )

    turn = TurnUsage(
        provider=Provider.CODEX_RESPONSES,
        model="gpt-5.4",
        input_tokens=102,
        output_tokens=108,
        total_tokens=210,
        cache_usage=CacheUsage(cache_hit_tokens=5),
        reasoning_tokens=64,
        raw_usage=snapshot_json_value(websocket_usage),
    )

    assert turn.raw_usage == {
        "input_tokens": 102,
        "input_tokens_details": {"cached_tokens": 5},
        "output_tokens": 108,
        "output_tokens_details": {"reasoning_tokens": 64},
        "total_tokens": 210,
    }

    accumulator = UsageAccumulator(turns=[turn], model="gpt-5.4")
    payload = build_usage_payload(accumulator)

    assert payload is not None
    assert payload["raw_usage"] == turn.raw_usage


def test_snapshot_json_value_falls_back_to_strings_for_unknown_nested_objects():
    class UnknownLeaf:
        def __str__(self) -> str:
            return "unknown-leaf"

    class UnknownUsage:
        def __init__(self) -> None:
            self.input_tokens = 11
            self.details = UnknownLeaf()

    assert snapshot_json_value(UnknownUsage()) == {
        "input_tokens": 11,
        "details": "unknown-leaf",
    }


def test_usage_accumulator():
    """Test accumulation of usage across multiple turns"""
    accumulator = UsageAccumulator()

    # Add Anthropic turn
    anthropic_usage = AnthropicUsage(input_tokens=1000, output_tokens=500)
    anthropic_turn = TurnUsage.from_anthropic(anthropic_usage, "claude-sonnet-4-0")
    accumulator.add_turn(anthropic_turn)

    # Add OpenAI turn
    openai_usage = OpenAIUsage(prompt_tokens=800, completion_tokens=400, total_tokens=1200)
    openai_turn = TurnUsage.from_openai(openai_usage, "gpt-4o")
    accumulator.add_turn(openai_turn)

    # Verify accumulation
    assert accumulator.turn_count == 2
    assert accumulator.cumulative_input_tokens == 1800  # 1000 + 800
    assert accumulator.cumulative_output_tokens == 900  # 500 + 400
    assert accumulator.cumulative_billing_tokens == 2700  # 1500 + 1200

    # Current context is from last turn
    assert accumulator.current_context_tokens == 1200  # openai turn total

    # Model from first turn
    assert accumulator.model == "claude-sonnet-4-0"


def test_cache_usage_properties():
    """Test cache usage computed properties"""
    cache = CacheUsage(cache_read_tokens=100, cache_write_tokens=50, cache_hit_tokens=200)

    assert cache.total_cache_tokens == 350  # 100 + 50 + 200
    assert cache.has_cache_activity is True

    empty_cache = CacheUsage()
    assert empty_cache.total_cache_tokens == 0
    assert empty_cache.has_cache_activity is False


def test_context_window_calculations():
    """Test context window size and percentage calculations"""
    # Anthropic usage with known context window
    anthropic_usage = AnthropicUsage(input_tokens=100000, output_tokens=50000)
    turn = TurnUsage.from_anthropic(anthropic_usage, "claude-sonnet-4-0")

    accumulator = UsageAccumulator()
    accumulator.add_turn(turn)

    # Context window for claude-sonnet-4-0 is 200,000
    assert accumulator.context_window_size == 200000
    assert accumulator.context_usage_percentage == 75.0  # 150000/200000 * 100

    # Test model without context window
    fast_agent_turn = create_turn_usage_from_messages("test", "response", "unknown-model", "test")
    unknown_accumulator = UsageAccumulator()
    unknown_accumulator.add_turn(fast_agent_turn)

    assert unknown_accumulator.context_window_size is None
    assert unknown_accumulator.context_usage_percentage is None


def test_cache_hit_rate_calculation():
    """Test cache hit rate percentage calculation"""
    accumulator = UsageAccumulator()

    # Anthropic turn with cache reads
    anthropic_usage = AnthropicUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=300,
    )
    anthropic_turn = TurnUsage.from_anthropic(anthropic_usage, "claude-sonnet-4-0")
    accumulator.add_turn(anthropic_turn)

    # OpenAI turn with cache hits
    prompt_details = PromptTokensDetails(cached_tokens=200)
    openai_usage = OpenAIUsage(
        prompt_tokens=800,
        completion_tokens=400,
        total_tokens=1200,
        prompt_tokens_details=prompt_details,
    )
    openai_turn = TurnUsage.from_openai(openai_usage, "gpt-4o")
    accumulator.add_turn(openai_turn)

    # With our updated algorithm:
    # Anthropic cumulative_input: 1000 + 300 (cache read) = 1300
    # OpenAI cumulative_input: 800 (already includes cache)
    # Total cumulative_input: 1300 + 800 = 2100
    # Total cache: 300 (anthropic read) + 200 (openai hit) = 500
    # Hit rate: 500 / (2100 + 500) * 100 = 500/2600 = 19.23%
    expected_hit_rate = 500 / (2100 + 500) * 100
    assert accumulator.cache_hit_rate is not None
    assert abs(accumulator.cache_hit_rate - expected_hit_rate) < 0.01

    # Test with no input tokens
    empty_accumulator = UsageAccumulator()
    assert empty_accumulator.cache_hit_rate is None


def test_provider_cache_differences():
    """Test that Anthropic and OpenAI handle cache tokens differently"""
    # Anthropic: separates cache creation (write) and cache read tokens
    anthropic_usage = AnthropicUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_creation_input_tokens=100,  # cache write
        cache_read_input_tokens=200,  # cache read
    )
    anthropic_turn = TurnUsage.from_anthropic(anthropic_usage, "claude-sonnet-4-0")

    # OpenAI: only has cached_tokens (cache hits)
    prompt_details = PromptTokensDetails(cached_tokens=300)
    openai_usage = OpenAIUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=prompt_details,
    )
    openai_turn = TurnUsage.from_openai(openai_usage, "gpt-4o")

    # Anthropic cache structure
    assert anthropic_turn.cache_usage.cache_write_tokens == 100  # creation
    assert anthropic_turn.cache_usage.cache_read_tokens == 200  # read
    assert anthropic_turn.cache_usage.cache_hit_tokens == 0  # not used
    # For Anthropic: input_tokens already excludes cached content, so effective_input = input_tokens
    assert anthropic_turn.effective_input_tokens == 1000  # input_tokens (already excludes cache)

    # OpenAI cache structure
    assert openai_turn.cache_usage.cache_write_tokens == 0  # not used
    assert openai_turn.cache_usage.cache_read_tokens == 0  # not used
    assert openai_turn.cache_usage.cache_hit_tokens == 300  # cached_tokens
    assert openai_turn.effective_input_tokens == 700  # 1000 - 300

    # Both have same total input/output but different cache accounting
    assert anthropic_turn.input_tokens == openai_turn.input_tokens == 1000
    assert anthropic_turn.output_tokens == openai_turn.output_tokens == 500


def test_usage_accumulator_context_window_size_override():
    """UsageAccumulator.context_window_size respects explicit active-model size."""
    acc = UsageAccumulator()
    acc.model = "claude-opus-4-6"

    # Without explicit size, should return ModelDatabase value (1M)
    assert acc.context_window_size == 1_000_000

    # With explicit active-model size, should return that size
    acc.set_context_window_size(1_000_000)
    assert acc.context_window_size == 1_000_000

    # Clear explicit size
    acc.set_context_window_size(None)
    assert acc.context_window_size == 1_000_000
