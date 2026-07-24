from datetime import datetime

from fast_agent.ui.turn_usage_display import (
    CacheTTLExpiry,
    NamedTurnUsageDisplay,
    TurnUsageDisplay,
    format_parallel_turn_usage,
    format_regular_turn_usage,
)


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    tool_calls: int = 0,
    cache_percentage: float | None = None,
    context_percentage: float | None = None,
    cache_ttl: CacheTTLExpiry | None = None,
) -> TurnUsageDisplay:
    return TurnUsageDisplay(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        cache_percentage=cache_percentage,
        context_percentage=context_percentage,
        cache_ttl=cache_ttl,
    )


def test_regular_turn_usage_uses_compact_detail_hierarchy() -> None:
    expiry = datetime(2030, 1, 1, 14, 32).timestamp()

    rendered = format_regular_turn_usage(
        _usage(
            input_tokens=1_234_567,
            output_tokens=12_345,
            tool_calls=3,
            cache_percentage=82,
            context_percentage=14.2,
            cache_ttl=CacheTTLExpiry(expires_at=expiry),
        )
    )

    assert (
        rendered
        == "[dim]Last:[/dim] [blue]▶ 1.23M[/blue] input [dim](cache 82%)[/dim]  "
        "[green]◀ 12,345[/green] output"
        " [dim]· 3 tool calls · context 14.2% · cache TTL 14:32[/dim]"
    )


def test_parallel_turn_usage_weights_cache_percentage_by_input_tokens() -> None:
    lines = format_parallel_turn_usage(
        [
            NamedTurnUsageDisplay(
                name="one",
                usage=_usage(input_tokens=100, output_tokens=10, cache_percentage=50),
            ),
            NamedTurnUsageDisplay(
                name="two",
                usage=_usage(input_tokens=300, output_tokens=20, cache_percentage=25),
            ),
        ]
    )

    assert "▶ 400[/blue] input [dim](cache 31%)[/dim]" in lines[0]
    assert "◀ 30[/green] output" in lines[0]
    assert lines[1].startswith("[dim]  ├─ one:[/dim]")
    assert lines[2].startswith("[dim]  └─ two:[/dim]")
