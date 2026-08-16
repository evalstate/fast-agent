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
    format_usage_markdown,
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
    assert "▶ Input" in rendered
    assert "85%" in rendered
    assert "◀ Output" in rendered
    assert all(len(line) <= 80 for line in rendered.splitlines())
    assert _format_cache_percentage(999, 1_000) == ">99%"


def test_usage_table_keeps_agent_column_compact_and_labels_last_context() -> None:
    total = UsageAccumulator()
    total.set_context_window_size(10_000)
    total.add_turn(
        _turn(
            input_tokens=1_000,
            cache_read_tokens=750,
            output_tokens=200,
            tool_calls=3,
            model="gpt-test",
        )
    )
    agent = SimpleNamespace(usage_accumulator=total, llm=None)
    data = _collect_usage_display_data({"ripgrep_spark": agent})

    assert data is not None
    console = Console(record=True, width=140, color_system=None)
    console.print(_usage_table(data, subdued_colors=False))
    lines = console.export_text().splitlines()

    assert "Last context" in lines[0]
    assert lines[0].index("Input") - lines[0].index("Agent") < 24
    assert lines[1].index("1,000") - lines[1].index("ripgrep_spark") < 24
    assert lines[0].index("Input") + len("Input") == lines[1].index("1,000") + len("1,000")
    assert lines[0].index("Output") + len("Output") == lines[1].index("200") + len("200")
    assert lines[0].index("Last context") + len("Last context") == lines[1].index("12.0%") + len(
        "12.0%"
    )


def test_usage_table_compacts_large_token_counts_to_four_significant_digits() -> None:
    total = UsageAccumulator()
    total.add_turn(
        _turn(
            input_tokens=21_524_724,
            cache_read_tokens=20_000_000,
            output_tokens=51_651,
            tool_calls=385,
            model="gpt-test",
        )
    )
    agent = SimpleNamespace(usage_accumulator=total, llm=None)
    data = _collect_usage_display_data({"dev": agent})

    assert data is not None
    console = Console(record=True, width=120, color_system=None)
    console.print(_usage_table(data, subdued_colors=False))
    rendered = console.export_text()

    assert "21.52M" in rendered
    assert "21,524,724" not in rendered
    assert "51,651" in rendered


def test_usage_markdown_contains_model_visible_values() -> None:
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

    rendered = format_usage_markdown({"dev": agent})

    assert "| Agent | Input | Cache hit | Output | Tool calls | Last context | Model |" in rendered
    assert "| dev | 1,000 | 85% | 20 | 3 | - | unknown |" in rendered
