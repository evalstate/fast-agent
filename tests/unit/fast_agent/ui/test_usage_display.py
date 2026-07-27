from types import SimpleNamespace

from rich.console import Console

from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
)
from fast_agent.ui.usage_display import (
    _collect_usage_display_data,
    _format_cache_percentage,
    _usage_table,
)


def _turn(
    *,
    input_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
    tool_calls: int,
    model: str,
) -> TurnUsage:
    return TurnUsage(
        provider=Provider.RESPONSES,
        usage_schema=UsageSchema.OPENAI_RESPONSES,
        model=model,
        prompt=PromptTokenUsage(total=input_tokens, cache_read=cache_read_tokens),
        completion=CompletionTokenUsage(total=output_tokens),
        tool_calls=tool_calls,
    )


def test_usage_breaks_out_subagents_without_double_counting() -> None:
    total = UsageAccumulator()
    total.add_turn(
        _turn(
            input_tokens=100,
            cache_read_tokens=50,
            output_tokens=10,
            tool_calls=1,
            model="parent-model",
        )
    )
    child = UsageAccumulator()
    child.add_turn(
        _turn(
            input_tokens=300,
            cache_read_tokens=270,
            output_tokens=20,
            tool_calls=2,
            model="child-model",
        )
    )
    total.add_turn(child.turns[0].model_copy(deep=True))
    agent = SimpleNamespace(
        usage_accumulator=total,
        subagent_usage_accumulator=child,
        llm=None,
    )

    data = _collect_usage_display_data({"dev": agent})

    assert data is not None
    assert [(row.name, row.input_tokens, row.output_tokens) for row in data.rows] == [
        ("dev", 100, 10),
        ("dev › subagents", 300, 20),
    ]
    assert data.total_input == 400
    assert data.total_cache_read == 320
    assert data.total_output == 30
    assert data.total_tool_calls == 3
    assert data.rows[1].model == "child-model"


def test_usage_uses_turn_summary_cache_language() -> None:
    total = UsageAccumulator()
    total.add_turn(
        _turn(
            input_tokens=1_000,
            cache_read_tokens=850,
            output_tokens=20,
            tool_calls=3,
            model="gpt-test",
        )
    )
    agent = SimpleNamespace(usage_accumulator=total, llm=None)
    data = _collect_usage_display_data({"dev": agent})

    assert data is not None
    console = Console(record=True, width=80, color_system=None)
    console.print(_usage_table(data, subdued_colors=False))
    rendered = console.export_text()
    assert "▶" in rendered
    assert "85%" in rendered
    assert "◀" in rendered
    assert all(len(line) <= 80 for line in rendered.splitlines())
    assert _format_cache_percentage(999, 1_000) == ">99%"
