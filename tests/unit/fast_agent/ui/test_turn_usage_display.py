from datetime import datetime

from fast_agent.ui.turn_usage_display import (
    CacheTTLExpiry,
    NamedTurnUsageDisplay,
    TurnUsageDisplay,
    _render_turn_usage,
    format_parallel_turn_usage,
    format_regular_turn_usage,
    format_regular_turn_usage_with_subagents,
)


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    tool_calls: int = 0,
    cache_percentage: float | None = None,
    cache_write_tokens: int | None = None,
    context_percentage: float | None = None,
    cache_ttl: CacheTTLExpiry | None = None,
) -> TurnUsageDisplay:
    return TurnUsageDisplay(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
        cache_percentage=cache_percentage,
        cache_write_tokens=cache_write_tokens,
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
            cache_write_tokens=1_200,
            context_percentage=14.2,
            cache_ttl=CacheTTLExpiry(expires_at=expiry),
        )
    )

    assert (
        rendered == "[dim]Last:[/dim] [blue]▶ 1.23M[/blue] input "
        "[dim](cache 82%, wrote 1,200)[/dim]  "
        "[green]◀ 12,345[/green] output"
        " [dim]· 3 tool calls · context 14.2% · cache TTL 14:32[/dim]"
    )


def test_turn_usage_rendering_does_not_auto_highlight_numbers() -> None:
    rendered = _render_turn_usage(
        format_regular_turn_usage(_usage(input_tokens=2_200_000, output_tokens=20))
    )

    input_start = rendered.plain.index("▶")
    input_end = rendered.plain.index(" input")
    input_spans = [
        span for span in rendered.spans if span.start <= input_start and span.end >= input_end
    ]

    assert len(input_spans) == 1
    assert input_spans[0].style == "blue"


def test_regular_turn_usage_includes_delegated_breakdown() -> None:
    lines = format_regular_turn_usage_with_subagents(
        _usage(
            input_tokens=400,
            output_tokens=30,
            tool_calls=3,
            cache_percentage=80,
            context_percentage=5.3,
        ),
        _usage(
            input_tokens=300,
            output_tokens=20,
            tool_calls=2,
            cache_percentage=90,
        ),
    )

    assert len(lines) == 2
    assert lines[0].startswith("[dim]Last:[/dim]")
    assert lines[1].startswith("[dim]  └─ subagents:[/dim]")
    assert "context" not in lines[1]


def test_parallel_turn_usage_weights_cache_percentage_by_input_tokens() -> None:
    lines = format_parallel_turn_usage(
        [
            NamedTurnUsageDisplay(
                name="one",
                usage=_usage(
                    input_tokens=100,
                    output_tokens=10,
                    cache_percentage=50,
                    cache_write_tokens=10,
                ),
            ),
            NamedTurnUsageDisplay(
                name="two",
                usage=_usage(
                    input_tokens=300,
                    output_tokens=20,
                    cache_percentage=25,
                    cache_write_tokens=20,
                ),
            ),
        ]
    )

    assert "▶ 400[/blue] input [dim](cache 31%, wrote 30)[/dim]" in lines[0]
    assert "◀ 30[/green] output" in lines[0]
    assert lines[1].startswith("[dim]  ├─ one:[/dim]")
    assert lines[2].startswith("[dim]  └─ two:[/dim]")


def test_cache_percentage_only_shows_100_when_exact() -> None:
    almost_all = format_regular_turn_usage(
        _usage(input_tokens=454_471, output_tokens=20, cache_percentage=99.9)
    )
    all_cached = format_regular_turn_usage(
        _usage(input_tokens=454_471, output_tokens=20, cache_percentage=100)
    )

    assert "(cache >99%)" in almost_all
    assert "(cache 100%)" in all_cached
